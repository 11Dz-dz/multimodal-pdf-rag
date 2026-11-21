"""Pair the eight-page answer with an optional visual read of candidate pages 9–12."""

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from answer_relay import AnswerAPIError, AnswerRelay
from demo import DEFAULT_MODELS, LocalModels, resolve_mineru_items
from dotenv import dotenv_values
from .judge import judge
from raganything.resume_retrieval import (
    PAGE_ANSWER_PROMPT,
    ResumeRetriever,
    evidence_ids,
    group_pages,
    json_object,
    terms,
)
from rank_bm25 import BM25Okapi


def indexed_rows(path, field):
    with path.open(encoding='utf-8') as stream:
        return {row['id']: row for line in stream if field in (row := json.loads(line))}


def sample_cases(rows, grades):
    targets, negatives = [], []
    for case in rows.values():
        trace = case['result']['trace']
        pages = [row['page'] for row in trace['pages']]
        if len(pages) < 9:
            continue
        verdict = grades[case['id']]['verdict']
        gold = set(case['gold_pages'])
        if (case['answerable'] and case['evidence_status'] == 'as_annotated' and verdict == 'false_refusal'
                and gold <= set(pages[:12]) and not gold <= set(pages[:8])):
            targets.append(('recoverable_refusal', case))
        elif not case['answerable'] and verdict == 'correct':
            negatives.append(('negative_control', case))
    rng = random.Random(512)
    assert len(targets) == 15
    pairs = targets + rng.sample(sorted(negatives, key=lambda item: item[1]['id']), 15)
    rng.shuffle(pairs)
    return pairs


def holdout_cases(rows, pilot):
    used = {case['id'] for _, case in pilot}
    eligible, insufficient, negatives = [], [], []
    for case in sorted(rows.values(), key=lambda case: case['id']):
        if case['id'] in used:
            continue
        pages = [row['page'] for row in case['result']['trace']['pages']]
        if len(pages) < 9:
            continue
        if not case['answerable']:
            negatives.append(('negative_control', case))
        elif case['evidence_status'] == 'as_annotated':
            gold = set(case['gold_pages'])
            if gold <= set(pages[:8]):
                continue
            if gold <= set(pages[:12]):
                eligible.append(('evidence_in_backfill', case))
            else:
                insufficient.append(('evidence_still_incomplete', case))
    rng = random.Random(731)
    pairs = rng.sample(eligible, 10) + rng.sample(insufficient, 5) + rng.sample(negatives, 15)
    rng.shuffle(pairs)
    return pairs


class ReadOnlyReplay(ResumeRetriever):
    def __init__(self, directory, case, models):
        self.doc_id = case['document_id'].replace('-', '')
        self.pdf_path = directory / 'source.pdf'
        self.models = models
        self.answer_page_limit = 8
        meta = json.loads((directory / 'document.json').read_text(encoding='utf-8'))
        self.filename = meta['filename']
        self.items = resolve_mineru_items(directory / meta['content_list'], include_charts=True)
        self.blocks = json.loads((directory / 'resume-index-v1/chunks.json').read_text(encoding='utf-8'))
        self.page_text, self.page_blocks = defaultdict(list), defaultdict(list)
        self.caption_targets = None
        for block in self.blocks:
            self.page_blocks[block['page']].append(block)
            if block['kind'] == 'text':
                self.page_text[block['page']].append(block)
        self.bm25 = BM25Okapi([terms(self.index_text(block)) or ['_'] for block in self.blocks])
        self.original = case['result']['trace']
        by_id = {block['id']: block for block in self.blocks}
        self.ranked_pages = group_pages([dict(by_id[hit['id']], rerank_score=hit['score'])
                                         for hit in self.original['reranked']])

    async def search_question(self, question, vector, page_context=False):
        trace = {key: self.original[key] for key in ('dense', 'bm25', 'rrf', 'reranked')}
        return self.ranked_pages, trace, question

    async def answer_with_backfill(self, question, baseline, *, verify=False):
        if not baseline['refused']:
            return baseline, [], {}
        trace = baseline['trace']
        ranks = [row['page'] for row in trace['pages']]
        extra = ranks[8:12]
        if not extra:
            return baseline, [], {}
        by_page = {row['page']: row for row in self.ranked_pages}
        ordered = [by_page.get(page, {'page': page, 'blocks': []}) for page in ranks[:12]]
        earlier = await asyncio.to_thread(self.page_sources, ordered[:8], question, [])
        later = await asyncio.to_thread(self.page_sources, ordered[8:12], question, [])
        text_context = [dict(item, images=[], text=item['text'][:2400]) for item in earlier]
        # Keep the retry image count bounded while retaining preceding context in text.
        prompt = ('The earlier pages are included as text. The additional pages include screenshots. '
                  'Answer only if the additional pages provide at least one necessary supporting fact. '
                  'Cite an additional page and any earlier page needed for the answer.\n')
        raw = await self.generate(prompt + PAGE_ANSWER_PROMPT + question, text_context + later, 220)
        output = json_object(raw)
        answer = output.get('answer')
        if type(answer) in (int, float):
            answer = str(answer)
        ids = evidence_ids(output.get('pages'), set(ranks[:12]))
        diagnostic = {}
        if verify:
            diagnostic['raw_proposal'] = raw
        if (isinstance(answer, str) and answer.strip() and set(ids).intersection(extra)
                and not answer.strip().lower().startswith(('not answerable', 'cannot answer',
                                                          'insufficient evidence', '证据不足', '无法回答'))):
            proposal = dict(baseline, answer=answer.strip(), refused=False,
                            citations=[{'page': page, 'doc_id': self.doc_id} for page in ids])
            if verify:
                from .visual_backfill_verifier import verify_backfill

                start = time.monotonic()
                diagnostic['verification'] = await verify_backfill(self, question, proposal, earlier, later)
                diagnostic['verification_seconds'] = round(time.monotonic() - start, 3)
            return proposal, extra, diagnostic
        return baseline, extra, diagnostic


async def evaluate_case(group, case, relay, site, semaphore, verify):
    async with semaphore:
        result = {'id': case['id'], 'group': group}
        directory = site / 'uploads' / case['document_id'].replace('-', '')
        models = type('RelayModel', (), {})()

        async def complete(messages, limit):
            for attempt in range(2):
                try:
                    return await asyncio.to_thread(relay.complete, messages, limit)
                except AnswerAPIError as error:
                    if attempt or error.status not in {429, 502, 504}:
                        raise
                    await asyncio.sleep(1)

        models.complete = complete
        try:
            replay = await asyncio.to_thread(ReadOnlyReplay, directory, case, models)
            start = time.monotonic()
            phase = 'eight_page_answer'
            baseline = await replay.page_answer(case['question'], None)
            old_pages = [page['page'] for page in case['result']['trace']['pages']]
            new_pages = [page['page'] for page in baseline['trace']['pages']]
            if old_pages[:12] != new_pages[:12]:
                raise ValueError('Replayed ranking differs from the frozen first twelve pages')
            candidate = baseline.pop('_unverified_candidate', None)
            if baseline['refused']:
                phase = 'text_review'
                baseline = await replay.review_refusal(case['question'], baseline, candidate=candidate)
            result.update(baseline=baseline, baseline_seconds=round(time.monotonic() - start, 3))
            start = time.monotonic()
            phase = 'visual_backfill'
            trial, pages, diagnostic = await replay.answer_with_backfill(case['question'], baseline, verify=verify)
            result.update(trial=trial, extra_pages=pages, extra_seconds=round(time.monotonic() - start, 3), **diagnostic)
        except AnswerAPIError as error:
            result.update(http_error=error.status, http_phase=phase)
        return case, result


async def run(args):
    root = args.root / 'full-run-haiku'
    rows = indexed_rows(root / 'responses.jsonl', 'result')
    cases = sample_cases(rows, indexed_rows(root / 'judgments.jsonl', 'verdict'))
    if args.holdout:
        cases = holdout_cases(rows, cases)
    if args.prior_trial:
        prior = json.loads(args.prior_trial.read_text(encoding='utf-8'))
        focus = {row['id'] for row in prior['cases'] if row.get('before') == 'false_refusal' and row.get('after') == 'wrong'}
        cases.sort(key=lambda item: item[1]['id'] not in focus)
        if args.previous_errors_only:
            cases = [(group, case) for group, case in cases if case['id'] in focus]
    if args.retry_from:
        previous = json.loads(args.retry_from.read_text(encoding='utf-8'))
        successful = {case['id'] for case in previous['cases'] if 'after' in case and 'http_error' not in case}
        cases = [(group, case) for group, case in cases if case['id'] not in successful]
        if not cases:
            return
    config = dotenv_values(args.env_file, interpolate=False)
    relay = AnswerRelay(config['RAG_ANSWER_API_BASE'], config['RAG_ANSWER_API_MODEL'], args.env_file,
                        compact_images=True, timeout=args.timeout)
    models = LocalModels(args.models, args.device)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(evaluate_case(group, case, relay, args.site, semaphore, args.verify))
             for group, case in cases]
    summaries, traces, extra_times = defaultdict(Counter), [], []
    for task in asyncio.as_completed(tasks):
        case, data = await task
        group = data['group']
        summary = summaries[group]
        summary['sampled'] += 1
        if 'http_error' in data:
            summary['http_error'] += 1
        if 'trial' in data:
            old, new = data['baseline'], data['trial']
            old_verdict, _ = await asyncio.to_thread(judge, dict(case, result=old), models)
            new_verdict, _ = (await asyncio.to_thread(judge, dict(case, result=new), models)
                              if new is not old else (old_verdict, None))
            summary['paired'] += 1
            summary['retry_triggered'] += bool(data['extra_pages'])
            summary['old_' + old_verdict] += 1
            summary['new_' + new_verdict] += 1
            summary['gained_correct'] += old_verdict != 'correct' and new_verdict == 'correct'
            summary['lost_correct'] += old_verdict == 'correct' and new_verdict != 'correct'
            if data['extra_pages']:
                extra_times.append(data['extra_seconds'])
            data.update(before=old_verdict, after=new_verdict, baseline_refused=old['refused'],
                        trial_refused=new['refused'], baseline_pages=old['trace']['answer_pages'],
                        trial_citations=[item['page'] for item in new['citations']])
            if args.verify:
                verification = data.get('verification', {})
                verified_verdict = old_verdict if verification.get('outcome') != 'verified' else new_verdict
                summary['verified_' + verified_verdict] += 1
                data['verified_verdict'] = verified_verdict
                data['verification_outcome'] = verification.get('outcome', 'no_proposal')
                if new is not old:
                    data.update(question=case['question'], gold=case['gold'], gold_pages=case['gold_pages'],
                                backfill_answer=new['answer'])
        data.pop('baseline', None)
        data.pop('trial', None)
        traces.append(data)
        output = {'groups': summaries, 'completed': len(traces), 'planned': len(tasks), 'cases': traces,
                  'retry_seconds_p50': statistics.median(extra_times) if extra_times else None}
        pending = args.report.with_suffix('.tmp')
        pending.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        pending.replace(args.report)
        print(f"{len(traces)}/{len(tasks)} {group} {data.get('before')} -> {data.get('after')} "
              f"HTTP={data.get('http_error')}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--retry-from', type=Path, help='Repeat cases without a completed pair in an earlier report')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--prior-trial', type=Path, help='Prior paired report used to prioritize its wrong answers')
    parser.add_argument('--previous-errors-only', action='store_true')
    parser.add_argument('--holdout', action='store_true', help='Use a seeded sample of questions outside the pilot')
    asyncio.run(run(parser.parse_args()))
