"""Rank PDFs by chunk relevance across existing indexes, with question-only queries."""
import argparse
import asyncio
import json
import math
import random
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import torch
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from demo import DEFAULT_MODELS, LocalModels
from raganything.rank_fusion import (
    DOC_FUSION_METHOD, DOC_RRF_K, DOC_SECOND_PAGE_WEIGHT, DOC_VISUAL_LIMIT, DOC_VISUAL_WEIGHT,
    PAGE_FUSION_METHOD, PAGE_RRF_K, PAGE_TEXT_HEADS, PAGE_VISUAL_HEADS, PAGE_VISUAL_WEIGHT,
    fuse_documents, fuse_pages, rank_documents,
)
from raganything.resume_retrieval import FUSION_LIMIT, ResumeRetriever, fuse, group_pages, terms
from raganything.visual_index import MODEL, VisualWorker, catalog_documents


def read_log(path, key):
    if not path.is_file():
        return {}
    with path.open(encoding='utf-8') as stream:
        return {row[key]: row for row in map(json.loads, stream)}


def write_status(path, **status):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_vectors(site, catalog):
    blocks, vectors = [], []
    for document in catalog:
        directory = site / 'uploads' / document['id'].replace('-', '') / 'resume-index-v1'
        with closing(QdrantClient(path=str(directory / 'qdrant'))) as index:
            offset = None
            while True:
                entries, offset = index.scroll('chunks', limit=256, offset=offset, with_vectors=True)
                for point in entries:
                    assert point.payload['doc_id'] == document['id'].replace('-', '')
                    blocks.append(dict(point.payload, doc_id=document['id']))
                    vectors.append(point.vector)
                if offset is None:
                    break
    return blocks, np.asarray(vectors, dtype=np.float32)


class CorpusIndex:
    def __init__(self, blocks, vectors, device):
        self.blocks = blocks
        self.texts = [ResumeRetriever.index_text(block) for block in blocks]
        self.vectors = torch.as_tensor(vectors, device=device)
        lex = BM25Okapi([terms(text) or ['_'] for text in self.texts])
        norm = lex.k1 * (1 - lex.b + lex.b * np.asarray(lex.doc_len) / lex.avgdl)
        posting = defaultdict(list)
        for index, frequencies in enumerate(lex.doc_freqs):
            for token, count in frequencies.items():
                weight = lex.idf[token] * count * (lex.k1 + 1) / (count + norm[index])
                posting[token].append((index, weight))
        self.postings = {
            token: (np.fromiter((idx for idx, _ in values), dtype=np.int32),
                    np.fromiter((weight for _, weight in values), dtype=np.float64))
            for token, values in posting.items()
        }

    async def rank_pages(self, question, dense, models):
        sparse_scores = np.zeros(len(self.blocks))
        matched = np.zeros(len(self.blocks), dtype=bool)
        for token in terms(question):
            if token in self.postings:
                indexes, weights = self.postings[token]
                sparse_scores[indexes] += weights
                matched[indexes] = True
        indexes = np.flatnonzero(matched)
        order = indexes[np.argsort(-sparse_scores[indexes], kind='stable')[:FUSION_LIMIT]]
        ids = [index for index, _ in fuse(dense, order.tolist())]
        scores = await models.rerank(question, [self.texts[i] for i in ids])
        candidates = [dict(self.blocks[index], rerank_score=float(score)) for index, score in zip(ids, scores)]
        return group_pages(candidates)

    async def rank(self, question, dense, models, *, visual_pages=None, routing=None):
        pages = await self.rank_pages(question, dense, models)
        if visual_pages is None:
            documents = list(dict.fromkeys(page['doc_id'] for page in pages))[:5]
        else:
            text_docs = rank_documents(pages, routing.doc_second_page_weight)
            visual_docs = rank_documents(visual_pages, routing.doc_second_page_weight)
            ranked_docs = fuse_documents(text_docs, visual_docs, routing.doc_visual_weight,
                                        k=routing.doc_rrf_k, method=routing.doc_fusion)
            documents = [row['doc_id'] for row in ranked_docs[:5]]
            document_rank = {doc_id: rank for rank, doc_id in enumerate(documents)}
            pages = fuse_pages(pages, visual_pages, routing.doc_visual_weight, k=routing.doc_rrf_k)
            pages = sorted((row for row in pages if row['doc_id'] in document_rank),
                           key=lambda row: document_rank[row['doc_id']])
        return {'ranked_documents': documents, 'ranked_pages': [
            {'doc_id': p['doc_id'], 'page': p['page'], 'score': p['score']} for p in pages[:5]]}


async def visual_rankings(worker, queries, limit):
    if not worker:
        return [None] * len(queries)
    results = await worker.request('search_many', requests=[{'question': query} for query in queries],
                                   corpus=True, coarse_k=200, corpus_top_k=limit)
    return [row['corpus'] for row in results]


async def routed_question(args, index, models, worker):
    from serve import PdfLibrary

    query = args.question
    vector = (await models.embed([query]))[0]
    dense = (index.vectors @ torch.as_tensor(vector, device=args.device)).topk(FUSION_LIMIT).indices.cpu().tolist()
    [visual] = await visual_rankings(worker, [query], args.doc_visual_limit)
    result = await index.rank(query, dense, models, visual_pages=visual, routing=args)
    document = UUID(result['ranked_documents'][0])
    library = PdfLibrary(SimpleNamespace(
        backend='resume', data=args.site, mineru=Path('.mineru-venv/bin/mineru'), models=args.models,
        device=args.device, ocr_python=Path('.ocr-venv/bin/python'), answer_page_limit=8,
        visual_weight=args.page_visual_weight, visual_fusion=args.page_fusion, visual_rrf_k=args.page_rrf_k,
        visual_text_heads=args.page_text_heads, visual_heads=args.page_visual_heads,
    ))
    library.models, library.visual_worker = models, worker
    try:
        retriever = await library.activate(document, library.metadata(document))
        selected, _, _, _ = await retriever.rank_answer_pages(query, vector)
        result.update(document=str(document), selected_pages=[row['page'] for row in selected[:8]])
    finally:
        await library.close()
    print(json.dumps(result, ensure_ascii=False))


async def run_corpus(args, catalog, worker):
    if args.question:
        blocks, vectors = read_vectors(args.site, catalog)
        await routed_question(args, CorpusIndex(blocks, vectors, args.device),
                              LocalModels(args.models, args.device), worker)
        return
    cases = [json.loads(line) for line in (args.root / 'test-set.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(catalog) == 135 and len(cases) == 1071
    args.run.mkdir(exist_ok=True, parents=True)
    path = args.run / 'routes.jsonl'
    completed = read_log(path, 'id')
    remaining = [case for case in cases if case['id'] not in completed]
    paired_path = args.run / 'paired-routes.jsonl'
    if not remaining and paired_path.is_file() and len(read_log(paired_path, 'id')) == 30:
        return
    write_status(args.run / 'status.json', phase='building_corpus', complete=len(completed), questions=len(cases))
    blocks, vectors = read_vectors(args.site, catalog)
    index = CorpusIndex(blocks, vectors, args.device)
    models = LocalModels(args.models, args.device)
    print('Corpus index', len(catalog), 'PDFs', len(blocks), 'chunks', flush=True)
    with path.open('a', encoding='utf-8', buffering=1) as log:
        for start in range(0, len(remaining), 16):
            batch = remaining[start:start + 16]
            queries = [row['question'] for row in batch]
            embeddings = torch.as_tensor(await models.embed(queries), device=args.device)
            neighbors = (embeddings @ index.vectors.T).topk(FUSION_LIMIT, dim=1).indices.cpu().tolist()
            visual = await visual_rankings(worker, queries, args.doc_visual_limit)
            for row, query, dense, visual_pages in zip(batch, queries, neighbors, visual):
                result = await index.rank(query, dense, models, visual_pages=visual_pages, routing=args)
                log.write(json.dumps({'id': row['id'], **result}, ensure_ascii=False) + '\n')
            complete = len(completed) + start + len(batch)
            write_status(args.run / 'status.json', phase='retrieving', complete=complete, questions=len(cases))
            print('Retrieved', complete, '/', len(cases), flush=True)
    # Freeze a diagnostic of 30 independently verifiable two-document questions.
    # The two components are passed as one query or searched separately and fused.
    sampler = random.Random(2350)
    eligible = [case for case in cases if case['category'] == 'single_page'
                and case['evidence_status'] == 'as_annotated']
    candidates = sorted(eligible, key=lambda row: row['id'])
    routes = read_log(path, 'id')
    paired = read_log(paired_path, 'id')
    with paired_path.open('a', encoding='utf-8', buffering=1) as log:
        for number in range(30):
            first = sampler.choice(candidates)
            second = sampler.choice([case for case in candidates if case['document_id'] != first['document_id']])
            case_id = f'paired-{number + 1:02}'
            if case_id in paired:
                continue
            question = f"A: {first['question']}\nB: {second['question']}"
            vector = torch.as_tensor((await models.embed([question]))[0], device=args.device)
            dense = (index.vectors @ vector).topk(FUSION_LIMIT).indices.cpu().tolist()
            [visual] = await visual_rankings(worker, [question], args.doc_visual_limit)
            direct = await index.rank(question, dense, models, visual_pages=visual, routing=args)
            split = fuse(routes[first['id']]['ranked_documents'], routes[second['id']]['ranked_documents'],
                         limit=5, dense_floor=0)
            row = {'id': case_id, 'question': question, 'components': [first['id'], second['id']],
                   'evidence_documents': [first['document_id'], second['document_id']],
                   'direct_documents': direct['ranked_documents'], 'split_documents': [doc for doc, _ in split]}
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_status(args.run / 'status.json', phase='retrieval_complete', complete=len(cases), questions=len(cases),
                 corpus_pdfs=len(catalog), corpus_chunks=len(blocks))


async def main(args):
    catalog = json.loads((args.root / 'documents.json').read_text(encoding='utf-8'))
    worker = None
    try:
        if args.visual_python:
            worker = VisualWorker(args.visual_python, args.visual_model, device=args.visual_device,
                                  base_model=args.visual_base_model)
            await worker.request('configure_corpus', documents=list(catalog_documents(args.site, catalog)))
        await run_corpus(args, catalog, worker)
    finally:
        if worker:
            await worker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--question', help='Route one question, then select eight pages in the predicted PDF')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--visual-python', type=Path)
    parser.add_argument('--visual-model', default=MODEL)
    parser.add_argument('--visual-base-model')
    parser.add_argument('--visual-device', default='cuda:1')
    parser.add_argument('--doc-visual-limit', type=int, choices=(24, 60, 200), default=DOC_VISUAL_LIMIT)
    parser.add_argument('--doc-second-page-weight', type=float, default=DOC_SECOND_PAGE_WEIGHT)
    parser.add_argument('--doc-visual-weight', type=float, default=DOC_VISUAL_WEIGHT)
    parser.add_argument('--doc-rrf-k', type=int, default=DOC_RRF_K)
    parser.add_argument('--doc-fusion', choices=('rrf', 'minmax'), default=DOC_FUSION_METHOD)
    parser.add_argument('--page-visual-weight', type=float, default=PAGE_VISUAL_WEIGHT)
    parser.add_argument('--page-rrf-k', type=int, default=PAGE_RRF_K)
    parser.add_argument('--page-fusion', choices=('rrf', 'minmax'), default=PAGE_FUSION_METHOD)
    parser.add_argument('--page-text-heads', type=int, choices=range(9), default=PAGE_TEXT_HEADS)
    parser.add_argument('--page-visual-heads', type=int, choices=range(9), default=PAGE_VISUAL_HEADS)
    args = parser.parse_args()
    if not (args.question or args.run):
        parser.error('Provide --run for benchmark retrieval or --question for one question')
    if not all(math.isfinite(value) and value >= 0 for value in
               (args.doc_visual_weight, args.doc_second_page_weight, args.page_visual_weight)):
        parser.error('Fusion weights must be finite and nonnegative')
    if min(args.doc_rrf_k, args.page_rrf_k) < 1:
        parser.error('RRF constants must be positive')
    asyncio.run(main(args))
