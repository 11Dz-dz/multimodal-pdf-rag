"""Compare static and tool-based PDF QA on the same frozen sample."""
import argparse
import asyncio
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from dotenv import dotenv_values

from answer_relay import AnswerAPIError, AnswerRelay
from demo import LocalModels
from experiments.judge import judge
from experiments.replay_visual_backfill import ReadOnlyReplay, indexed_rows
from raganything.rank_fusion import PAGE_FUSION_METHOD, PAGE_RRF_K, PAGE_TEXT_HEADS, PAGE_VISUAL_HEADS, PAGE_VISUAL_WEIGHT
from raganything.visual_index import VisualPageIndex, VisualWorker


STRATA = {'single_page': 30, 'cross_page': 30, 'unanswerable': 25}


def sample(rows):
    rng = random.Random(11422026)
    seen = Counter()
    selected = []
    for category, count in STRATA.items():
        documents = defaultdict(list)
        for row in rows:
            if category == 'unanswerable':
                eligible = not row['answerable']
            else:
                eligible = (row['answerable'] and row['category'] == category
                            and row['evidence_status'] == 'as_annotated')
            if eligible:
                documents[row['document_id']].append(row)
        names = sorted(documents)
        rng.shuffle(names)
        names.sort(key=seen.__getitem__)
        if len(names) < count:
            raise ValueError(f'Not enough distinct PDFs in {category}')
        for name in names[:count]:
            row = rng.choice(sorted(documents[name], key=lambda r: r['id']))
            selected.append({'id': row['id'], 'document_id': name, 'category': category})
            seen[name] += 1
    rng.shuffle(selected)
    return selected


def full_sample(path, archive):
    cases = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(cases) != 1071 or len({row['id'] for row in cases}) != len(cases):
        raise ValueError('The canonical full evaluation must contain 1,071 unique questions')
    for row in cases:
        previous = archive[row['id']]
        for key in ('question', 'document_id', 'answerable', 'gold_pages'):
            if previous[key] != row[key]:
                raise ValueError(f'Canonical sample and replay archive differ for {row["id"]}: {key}')
    selected = [{'id': row['id'], 'document_id': row['document_id'],
                 'category': row['category'] if row['answerable'] else 'unanswerable',
                 'evidence_status': row['evidence_status']} for row in cases]
    random.Random(11422026).shuffle(selected)
    return selected


def summarize(sample_rows, outputs, model, mode):
    groups = {}
    full = any(row['category'] == 'evidence_unannotated' for row in sample_rows)
    group_names = ('all', 'answerable', *STRATA, 'evidence_unannotated') if full else ('all', *STRATA)
    for key in group_names:
        members = [row for row in sample_rows if key in ('all', row['category'])
                   or (key == 'answerable' and row['category'] != 'unanswerable')]
        ids = [row['id'] for row in members]
        rows = [outputs[case_id] for case_id in ids if case_id in outputs]
        verdicts = Counter(row['verdict'] for row in rows)
        measured = [r for r in rows if r['verdict'] != 'api_error']
        groups[key] = {'total': len(ids), 'verdicts': verdicts, 'correct': verdicts['correct'],
                       'refused': sum(r['refused'] for r in measured),
                       'first_round_complete': sum(bool(r.get('initial_evidence_complete')) for r in measured),
                       'evidence_complete': sum(bool(r.get('evidence_complete')) for r in measured),
                       'mean_seconds': statistics.mean(r['elapsed_seconds'] for r in measured) if measured else None,
                       'median_seconds': statistics.median(r['elapsed_seconds'] for r in measured) if measured else None,
                       'mean_turns': statistics.mean(r['turns'] for r in measured) if measured else None,
                       'mean_images': statistics.mean(r['images'] for r in measured) if measured else None,
                       'one_turn': sum(r['turns'] == 1 for r in measured)}
        if full:
            evidence_ids = {row['id'] for row in members
                            if row['category'] != 'unanswerable' and row['evidence_status'] == 'as_annotated'}
            evidence = [r for r in measured if r['id'] in evidence_ids]
            groups[key].update(evidence_eligible=len(evidence_ids), evidence_scored=len(evidence),
                               first_round_complete=sum(bool(r.get('initial_evidence_complete')) for r in evidence),
                               evidence_complete=sum(bool(r.get('evidence_complete')) for r in evidence))
    return {'mode': mode, 'model': model, 'sample_documents': len({r['document_id'] for r in sample_rows}),
            'groups': groups}


def paired_summary(sample_rows, static, agent):
    output = {}
    for key in ('all', *STRATA):
        rows = [row for row in sample_rows if key in ('all', row['category'])]
        pairs = [(static.get(row['id']), agent.get(row['id'])) for row in rows]
        comparable = [(a, b) for a, b in pairs if a and b
                      and not {a['verdict'], b['verdict']} & {'api_error', 'judge_error'}]
        output[key] = {
            'comparable': len(comparable),
            'agent_only_correct': sum(a['verdict'] != 'correct' and b['verdict'] == 'correct' for a, b in comparable),
            'static_only_correct': sum(b['verdict'] != 'correct' and a['verdict'] == 'correct' for a, b in comparable),
            'both_correct': sum(a['verdict'] == b['verdict'] == 'correct' for a, b in comparable),
        }
    tools = Counter()
    for row in agent.values():
        tools.update(entry['tool'] for entry in row.get('tools', []) if not entry['error'])
    return {'groups': output, 'successful_agent_tools': tools}


def write_json(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


async def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    split = args.out / 'sample.json'
    rows = indexed_rows(args.root / 'full-run-haiku' / 'responses.jsonl', 'id')
    canonical = full_sample(args.root / 'test-set.jsonl', rows) if args.full else None
    if not split.exists():
        if canonical is not None:
            write_json(split, canonical)
        elif args.compare_to:
            split.write_bytes((args.compare_to / 'sample.json').read_bytes())
        else:
            write_json(split, sample(rows.values()))
    cases = json.loads(split.read_text(encoding='utf-8'))
    if canonical is not None and cases != canonical:
        raise ValueError('The existing output directory does not match the canonical full evaluation')
    if args.compare_to and cases != json.loads((args.compare_to / 'sample.json').read_text(encoding='utf-8')):
        raise ValueError('Comparison requires the same frozen cases and case order')
    settings = dotenv_values(args.env_file, interpolate=False)
    if args.bailian_thinking:
        base_url = (settings.get('DASHSCOPE_BASE_URL') or os.environ.get('DASHSCOPE_BASE_URL')
                    or 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        model = args.answer_api_model or 'qwen3.7-plus'
        thinking = args.bailian_thinking == 'on'
    else:
        base_url = settings['RAG_ANSWER_API_BASE']
        model = args.answer_api_model or settings['RAG_ANSWER_API_MODEL']
        thinking = None
    relay = AnswerRelay(base_url, model, args.env_file, compact_images=True, timeout=args.timeout,
                        native_claude=args.native_claude, bailian_thinking=thinking)
    relay.key()
    models = LocalModels(args.models, args.text_device)
    worker = VisualWorker(args.visual_python, args.visual_model, device=args.visual_device,
                          base_model=args.visual_base_model)
    grade_lock, capacity = asyncio.Lock(), asyncio.Semaphore(args.concurrency)
    document_locks = defaultdict(asyncio.Lock)
    path = args.out / f'{args.mode}.jsonl'
    outputs = indexed_rows(path, 'id') if path.exists() else {}
    completed = sum(row['verdict'] not in {'api_error', 'judge_error'} for row in outputs.values())
    agent = None
    if args.mode == 'agent':
        from raganything.agent_qa import agent_answer
        agent = agent_answer

    async def complete(messages, limit):
        return await asyncio.to_thread(relay.complete, messages, limit)

    async def evaluate(item):
        row = rows[item['id']]
        directory = args.site / 'uploads' / row['document_id'].replace('-', '')
        async with capacity, document_locks[row['document_id']]:
            replay = await asyncio.to_thread(ReadOnlyReplay, directory, row, models, rerun_ranking=True)
            replay.visual_index = VisualPageIndex(worker, replay.pdf_path, directory / 'resume-index-v1', replay.doc_id)
            replay.visual_weight, replay.visual_fusion = PAGE_VISUAL_WEIGHT, PAGE_FUSION_METHOD
            replay.visual_rrf_k, replay.visual_text_heads, replay.visual_heads = PAGE_RRF_K, PAGE_TEXT_HEADS, PAGE_VISUAL_HEADS
            replay.models = SimpleNamespace(embed=models.embed, rerank=models.rerank, complete=complete)
            start = time.monotonic()
            try:
                for attempt in range(2):
                    try:
                        if agent:
                            result = await agent(replay, row['question'], relay)
                        else:
                            vector = (await models.embed([row['question']]))[0]
                            result = await replay.page_answer(row['question'], vector)
                        break
                    except AnswerAPIError as exc:
                        if attempt or exc.status not in {429, 502, 504}:
                            return dict(item, verdict='api_error', status=exc.status, reason=str(exc),
                                        elapsed_seconds=time.monotonic() - start)
                        await asyncio.sleep(2)
                elapsed = time.monotonic() - start
                trace = result['trace']
                if trace.get('visual_index_status') != 'ready':
                    raise RuntimeError(f'Visual retrieval unavailable for {row["id"]}')
                async with grade_lock:
                    verdict, judge_method = await asyncio.to_thread(judge, dict(row, result=result), models)
                gold = row['gold_pages']
                inspected = trace.get('verification_pages', [])
                initial = trace.get('initial_pages', inspected)
                return dict(item, elapsed_seconds=elapsed, verdict=verdict, judge_method=judge_method,
                            gold=gold, gold_answer=row['gold'], question=row['question'], answer=result['answer'],
                            refused=result['refused'], citations=result['citations'],
                            turns=trace.get('agent_turns', int(bool(trace.get('answer_pages')))),
                            images=result['images_seen_by_model'], initial_pages=initial, inspected_pages=inspected,
                            initial_evidence_complete=bool(gold) and set(gold) <= set(initial),
                            evidence_complete=bool(gold) and set(gold) <= set(inspected),
                            tools=trace.get('agent_tools', []), output_status=trace.get('agent_output_status'))
            finally:
                await replay.close()

    try:
        with path.open('a', encoding='utf-8', buffering=1) as stream:
            tasks = [asyncio.create_task(evaluate(case)) for case in cases
                     if outputs.get(case['id'], {}).get('verdict') in (None, 'api_error', 'judge_error')]
            for future in asyncio.as_completed(tasks):
                result = await future
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                outputs[result['id']] = result
                completed += result['verdict'] not in {'api_error', 'judge_error'}
                print(args.mode, completed, '/', len(cases), result['category'], result['verdict'], flush=True)
                if result.get('status') == 402:
                    raise AnswerAPIError(402, result['reason'])
        report = summarize(cases, outputs, relay.model, args.mode)
        if thinking is not None:
            report.update(provider='bailian', enable_thinking=thinking)
        if args.mode == 'agent':
            report['output_statuses'] = Counter(row['output_status'] for row in outputs.values()
                                                if row.get('output_status'))
        write_json(args.out / f'{args.mode}-summary.json', report)
        other = 'static' if args.mode == 'agent' else 'agent'
        other_path = args.out / f'{other}.jsonl'
        if (args.out / f'{other}-summary.json').is_file():
            by_mode = {args.mode: outputs, other: indexed_rows(other_path, 'id')}
            write_json(args.out / 'paired-summary.json', paired_summary(cases, by_mode['static'], by_mode['agent']))
        if args.compare_to:
            archived = {}
            for reference in ('static', 'agent'):
                prior_path = args.compare_to / f'{reference}.jsonl'
                prior = indexed_rows(prior_path, 'id')
                archived[reference] = paired_summary(cases, prior, outputs)
            write_json(args.out / 'comparison.json', archived)
    finally:
        await worker.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'site', 'out', 'env-file', 'models', 'visual-python'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--visual-model', required=True)
    cli.add_argument('--visual-base-model', required=True)
    cli.add_argument('--visual-device', default='cuda:1')
    cli.add_argument('--text-device', default='cuda:0')
    cli.add_argument('--mode', choices=('static', 'agent'), required=True)
    cli.add_argument('--full', action='store_true', help='Use all 1,071 cases from the canonical test set')
    cli.add_argument('--answer-api-model', help='Override the dotenv model for this benchmark run')
    cli.add_argument('--bailian-thinking', choices=('off', 'on'), help='Call Bailian directly using the requested thinking mode')
    cli.add_argument('--compare-to', type=Path, help='Compare with an earlier static/agent run on the same sample')
    cli.add_argument('--native-claude', action='store_true')
    cli.add_argument('--timeout', type=int, default=75)
    cli.add_argument('--concurrency', type=int, default=3)
    asyncio.run(run(cli.parse_args()))
