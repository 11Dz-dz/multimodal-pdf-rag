"""Compare text, native page imagery and fused rankings without answer generation."""
import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import pymupdf

from demo import LocalModels
from experiments.replay_visual_backfill import ReadOnlyReplay, indexed_rows
from raganything.resume_retrieval import FUSION_LIMIT
from raganything.resume_references import requested_pages, requested_references
from raganything.visual_index import MODEL, VisualWorker, catalog_documents, fuse_pages
if __package__:
    from .retrieve_mmlongbench_corpus import CorpusIndex, read_vectors
else:
    from retrieve_mmlongbench_corpus import CorpusIndex, read_vectors


WEIGHTS = (0.8, 1.0, 1.2, 1.5)
METHODS = ('text', 'visual', *(f'rrf_{weight}' for weight in WEIGHTS))


def baselines_and_weights(text, visual):
    return dict(zip(METHODS, [text, visual, *(fuse_pages(text, visual, w) for w in WEIGHTS)]))


def coverage(retrieved, gold):
    pages = set(retrieved)
    return {'complete': gold <= pages, 'hit': bool(gold & pages), 'recall': len(pages & gold) / len(gold)}


def compact_pages(ranking, *, corpus=False):
    if corpus:
        return [[r['doc_id'], r['page'], r['score']] for r in ranking]
    return [[r['page'], r['score']] for r in ranking]


def case_results(args):
    cases = indexed_rows(args.root / 'test-set.jsonl', 'id')
    archive = indexed_rows(args.root / 'full-run-haiku' / 'responses.jsonl', 'id')
    baseline_known = {row['id']: row for row in json.loads(args.baseline.read_text(encoding='utf-8'))['cases']}
    baseline_corpus = indexed_rows(args.root / 'corpus-haiku' / 'routes.jsonl', 'id')
    assert len(baseline_known) == 842
    for row in cases.values():
        if not row['answerable']:
            continue
        case_id = row['id']
        expected_document = row['document_id']
        baseline_route = baseline_corpus[case_id]['ranked_documents'][:1]
        result = {**row, 'archive_doc_hit_1': baseline_route == [expected_document]}
        if case_id in baseline_known:
            result['archive'] = archive[case_id]
            result['baseline_known'] = baseline_known[case_id]
        yield result


def summary(rows):
    selected = [row for row in rows if 'document' in row]
    count = len(selected)
    document, corpus = {}, {}
    for name in METHODS:
        known = [row['document'][name] for row in selected]
        route = [row['corpus'][name] for row in rows]
        complete = sum(row['complete'] for row in known)
        hit1 = sum(row['hit1'] for row in route)
        document[name] = {
            'complete': complete, 'complete_rate': complete / count,
            'mean_recall': statistics.mean(row['recall'] for row in known),
            'hit_rate': statistics.mean(row['hit'] for row in known),
            'gained_vs_text': sum(rd['complete'] and not row['document']['text']['complete']
                                  for row, rd in zip(selected, known)),
            'lost_vs_text': sum(not rd['complete'] and row['document']['text']['complete']
                                for row, rd in zip(selected, known)),
        }
        corpus[name] = {
            'hit1': hit1, 'hit1_rate': hit1 / len(rows),
            'hit5': sum(rd['hit5'] for rd in route),
            'gained_vs_text': sum(rd['hit1'] and not row['corpus']['text']['hit1'] for row, rd in zip(rows, route)),
            'lost_vs_text': sum(not rd['hit1'] and row['corpus']['text']['hit1'] for row, rd in zip(rows, route)),
        }
    by_category = {}
    for category, total in Counter(row['category'] for row in selected).items():
        group = [row for row in selected if row['category'] == category]
        by_category[category] = {'questions': total,
                                 'complete': {name: sum(r['document'][name]['complete'] for r in group)
                                              for name in METHODS}}
    by_source = {}
    for source in sorted({source for row in selected for source in row['gold_sources']}):
        group = [row for row in selected if source in row['gold_sources']]
        by_source[source] = {'questions': len(group),
                             'complete': {name: sum(r['document'][name]['complete'] for r in group)
                                          for name in METHODS}}
    archive_known = sum(row['archive_known_complete'] for row in selected)
    archive_corpus = sum(row['archive_doc_hit_1'] for row in rows)
    return {'known_pdf': {'questions': count, 'archive_complete': archive_known, 'methods': document,
                          'category_breakdown': by_category, 'gold_source_breakdown': by_source},
            'open_corpus': {'questions': len(rows), 'archive_hit1': archive_corpus, 'methods': corpus,
                            'visual_top200_doc_hit': sum(r['visual_coarse_doc_hit'] for r in rows),
                            'visual_top200_all_gold_pages': sum(r['visual_coarse_all_gold_pages'] for r in selected)}}


async def run(args):
    catalog = json.loads((args.root / 'documents.json').read_text(encoding='utf-8'))
    documents = {row['doc_id']: row for row in catalog_documents(args.site, catalog)}
    cases = list(case_results(args))
    assert len(documents) == 135 and len(cases) == 863
    cases.sort(key=lambda case: (case['document_id'], case['id']))
    if args.limit:
        cases = cases[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / 'retrieval.jsonl'
    completed = indexed_rows(results_path, 'id') if results_path.exists() else {}
    worker = VisualWorker(args.visual_python, args.visual_model, device=args.visual_device,
                          base_model=args.visual_base_model)
    replay, active_document = None, None
    session_start = time.monotonic()
    try:
        page_count = await worker.request('configure_corpus', documents=list(documents.values()))
        blocks, vectors = read_vectors(args.site, catalog)
        corpus = CorpusIndex(blocks, vectors, args.text_device)
        del vectors
        models = LocalModels(args.models, args.text_device)
        print(f'Loaded {len(documents)} PDFs, {page_count} visual pages, {len(blocks)} text chunks', flush=True)
        remaining = [case for case in cases if case['id'] not in completed]
        with results_path.open('a', encoding='utf-8', buffering=1) as stream:
            for offset in range(0, len(remaining), 16):
                batch = remaining[offset:offset + 16]
                queries = [row['question'] for row in batch]
                requests = []
                for row, query in zip(batch, queries):
                    request = {'question': query}
                    if 'baseline_known' in row:
                        doc = documents[row['document_id']]
                        request['document'] = dict(doc, doc_id=doc['doc_id'].replace('-', ''))
                    requests.append(request)
                visual_task = asyncio.create_task(worker.request(
                    'search_many', requests=requests, top_k=24, corpus=True, coarse_k=200,
                    include_candidates=True, corpus_top_k=200 if args.capture_ranks else 24))
                embeddings = await models.embed(queries)
                neighbors = (torch.as_tensor(embeddings, device=args.text_device) @ corpus.vectors.T)
                neighbors = neighbors.topk(FUSION_LIMIT, dim=1).indices.cpu().tolist()
                text_rankings = [await corpus.rank_pages(query, dense, models)
                                 for query, dense in zip(queries, neighbors)]
                rankings = await visual_task
                for case, visual, vector, full_text in zip(batch, rankings, embeddings, text_rankings):
                    question, document = case['question'], case['document_id']
                    coarse = {page for doc, page in visual['corpus_candidates'] if doc == document}
                    entry = {'id': case['id'], 'category': case['category'], 'document_id': document,
                             'gold_sources': case['gold_sources'], 'archive_doc_hit_1': case['archive_doc_hit_1'],
                             'visual_coarse_doc_hit': bool(coarse), 'corpus': {}}
                    if args.capture_ranks:
                        entry['corpus_raw'] = {'text': compact_pages(full_text, corpus=True),
                                               'visual': compact_pages(visual['corpus'], corpus=True)}
                    for name, pages in baselines_and_weights(full_text, visual['corpus'][:24]).items():
                        ranked = list(dict.fromkeys(page['doc_id'] for page in pages))[:5]
                        entry['corpus'][name] = {'ranked_documents': ranked, 'hit1': ranked[:1] == [document],
                                                  'hit5': document in ranked}
                    if 'baseline_known' in case:
                        if active_document != document:
                            if replay is not None:
                                await replay.close()
                            replay = ReadOnlyReplay(args.site / 'uploads' / document.replace('-', ''),
                                                    case['archive'], models, rerun_ranking=True)
                            active_document = document
                        text, trace, query = await replay.search_question(question, vector, page_context=True)
                        gold = set(case['archive']['gold_pages'])
                        entry['gold_pages'] = sorted(gold)
                        entry['visual_coarse_all_gold_pages'] = gold <= coarse
                        entry['document'] = {}
                        entry['archive_known_complete'] = case['baseline_known']['new_complete']
                        if args.capture_ranks:
                            references = requested_references(question)
                            matches = []
                            if references:
                                matches, _, scoped = await asyncio.to_thread(replay.referenced_evidence, references, True)
                                if not scoped:
                                    matches = []
                            with pymupdf.open(replay.pdf_path) as source:
                                page_count = source.page_count
                            entry['document_raw'] = {
                                'text': compact_pages(text), 'visual': compact_pages(visual['document']),
                                'anchors': requested_pages(question, page_count=page_count)[:replay.answer_page_limit],
                                'references': [match['page'] for match in matches], 'page_count': page_count}
                        for name, candidates in baselines_and_weights(text, visual['document']).items():
                            selected, _, _, _ = await replay.select_answer_pages(question, candidates, dict(trace), query)
                            numbers = [page['page'] for page in selected[:8]]
                            entry['document'][name] = {'selected': numbers, **coverage(numbers, gold)}
                    stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    completed[case['id']] = entry
                print(f'Retrieved {len(completed)}/{len(cases)} questions in {time.monotonic()-session_start:.1f}s', flush=True)
        report = {'pdfs': len(documents), 'visual_pages': page_count, 'text_chunks': len(blocks),
                  'visual_index_bytes': sum(Path(doc['index']).stat().st_size for doc in documents.values()),
                  'inputs': 'Only the original question is used in corpus retrieval.',
                  'selection': 'All single-PDF rankings use the same page anchors, references and neighbor rules.',
                  'parameters': {'coarse_top_k': 200, 'visual_top_k': 24, 'rrf_k': 60, 'rrf_weights': list(WEIGHTS)},
                  **summary([completed[row['id']] for row in cases])}
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    finally:
        if replay is not None:
            await replay.close()
        await worker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--visual-python', type=Path, required=True)
    parser.add_argument('--visual-model', default=MODEL)
    parser.add_argument('--visual-base-model')
    parser.add_argument('--visual-device', default='cuda:1')
    parser.add_argument('--text-device', default='cuda:0')
    parser.add_argument('--limit', type=int, help='Smoke-test the first N positive questions')
    parser.add_argument('--capture-ranks', action='store_true',
                        help='Save scores for offline sweeps, including every MaxSim-ranked coarse corpus candidate')
    asyncio.run(run(parser.parse_args()))
