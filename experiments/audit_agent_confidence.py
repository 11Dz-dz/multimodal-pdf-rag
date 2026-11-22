"""Recover raw first-round scores and citation text with frozen local indexes."""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pymupdf

from demo import LocalModels
from experiments.replay_visual_backfill import ReadOnlyReplay, indexed_rows
from raganything.rank_fusion import PAGE_FUSION_METHOD, PAGE_RRF_K, PAGE_TEXT_HEADS, PAGE_VISUAL_HEADS, PAGE_VISUAL_WEIGHT
from raganything.resume_retrieval import ResumeRetriever
from raganything.visual_index import VisualPageIndex, VisualWorker


def cohort(rows):
    return {
        case['id']: case
        for case in rows.values()
        if ((case['category'] == 'unanswerable' and case['verdict'] == 'wrong')
            or (case['category'] != 'unanswerable' and case['verdict'] == 'correct'))
    }


def citation_text(pdf, blocks, number):
    native = pdf[number - 1].get_text(sort=True)
    indexed = blocks.get(number, [])
    # Avoid including automatically generated visual summaries in literal evidence.
    extracted = [block['text'] for block in indexed if block['kind'] == 'text']
    extracted.extend(block['text'] for block in indexed if block['kind'] == 'table' and block.get('table_method'))
    extracted.extend(block['ocr'] for block in indexed if block.get('ocr'))
    return '\n'.join(dict.fromkeys([native, *extracted]))


async def run(args):
    rows = cohort(indexed_rows(args.eval / 'agent.jsonl', 'id'))
    assert sum(row['category'] == 'unanswerable' for row in rows.values()) == 89
    assert len(rows) == 668
    archive = indexed_rows(args.root / 'full-run-haiku' / 'responses.jsonl', 'id')
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / 'first-round.jsonl'
    saved = indexed_rows(raw, 'id') if raw.exists() else {}
    grouped = defaultdict(list)
    for row in rows.values():
        if row['id'] not in saved:
            grouped[row['document_id']].append(row)
    if not grouped:
        return
    local = LocalModels(args.models, args.text_device)
    models = SimpleNamespace(embed=local.embed, rerank=local.rerank)
    worker = VisualWorker(args.visual_python, args.visual_model, device=args.visual_device,
                          base_model=args.visual_base_model)
    count = len(saved)
    try:
        with raw.open('a', encoding='utf-8', buffering=1) as stream:
            for document_id, group in grouped.items():
                directory = args.site / 'uploads' / document_id.replace('-', '')
                replay = ReadOnlyReplay(directory, archive[group[0]['id']], models, rerun_ranking=True)
                replay.visual_index = VisualPageIndex(worker, replay.pdf_path, directory / 'resume-index-v1', replay.doc_id)
                replay.visual_weight, replay.visual_fusion = PAGE_VISUAL_WEIGHT, PAGE_FUSION_METHOD
                replay.visual_rrf_k, replay.visual_text_heads, replay.visual_heads = (
                    PAGE_RRF_K, PAGE_TEXT_HEADS, PAGE_VISUAL_HEADS)
                cached = {}
                try:
                    with pymupdf.open(replay.pdf_path) as pdf:
                        for row in group:
                            query = row['question']
                            vector = (await models.embed([query]))[0]
                            ranked, trace, _, _ = await ResumeRetriever.rank_answer_pages(
                                replay, query, vector, translate=False)
                            assert trace.get('visual_index_status') == 'ready', row['id']
                            reranked = trace['reranked']
                            visual = trace['visual_pages']
                            pages = sorted({c['page'] for c in row['citations']})
                            for number in pages:
                                if number not in cached:
                                    cached[number] = citation_text(pdf, replay.page_blocks, number)
                            data = {'id': row['id'], 'document_id': document_id, 'category': row['category'],
                                    'question': query, 'answer': row['answer'], 'turns': row['turns'],
                                    'tools': row['tools'],
                                    'text_rerank_max': reranked[0]['score'] if reranked else None,
                                    'visual_maxsim_max': visual[0]['score'] if visual else None,
                                    'text_best_page': reranked[0]['page'] if reranked else None,
                                    'visual_best_page': visual[0]['page'] if visual else None,
                                    'original_initial_pages': row['initial_pages'],
                                    'replayed_initial_pages': [p['page'] for p in ranked[:3]],
                                    'cited_page_text': {number: cached[number] for number in pages}}
                            stream.write(json.dumps(data, ensure_ascii=False) + '\n')
                            count += 1
                            if count % 40 == 0 or count == len(rows):
                                print('Replayed', count, '/', len(rows), flush=True)
                finally:
                    await replay.close()
    finally:
        await worker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'site', 'eval', 'out', 'models', 'visual-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--visual-model', required=True)
    parser.add_argument('--visual-base-model', required=True)
    parser.add_argument('--visual-device', default='cuda:1')
    parser.add_argument('--text-device', default='cuda:0')
    asyncio.run(run(parser.parse_args()))
