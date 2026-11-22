"""Fuse text and visual evidence at the page or PDF level."""
from .resume_multistep import interleave_pages


PAGE_FUSION_METHOD = 'minmax'
PAGE_VISUAL_WEIGHT = 0.8
PAGE_RRF_K = 10
PAGE_TEXT_HEADS = 2
PAGE_VISUAL_HEADS = 3
DOC_FUSION_METHOD = 'minmax'
DOC_VISUAL_WEIGHT = 0.8
DOC_RRF_K = 10
DOC_SECOND_PAGE_WEIGHT = 0.0
DOC_VISUAL_LIMIT = 24


def _fuse(text, visual, key, visual_weight, *, k, method):
    rows, scores = {}, {}
    for ranking, weight in ((text, 1.0), (visual, visual_weight)):
        if not ranking or weight == 0:
            continue
        if method == 'minmax':
            low = min(row['score'] for row in ranking)
            span = max(row['score'] for row in ranking) - low
        seen = set()
        for rank, row in enumerate(ranking, 1):
            identity = key(row)
            if identity in seen:
                continue
            seen.add(identity)
            if method == 'minmax':
                score = weight * ((row['score'] - low) / span if span else 1.0)
            else:
                score = weight / (k + rank)
            rows.setdefault(identity, row)
            scores[identity] = scores.get(identity, 0.0) + score
    return [dict(rows[identity], score=scores[identity])
            for identity in sorted(scores, key=scores.__getitem__, reverse=True)]


def fuse_pages(text_pages, visual_pages, visual_weight=1.0, *, k=60, method='rrf'):
    if method == 'minmax':
        text_pages, visual_pages = text_pages[:24], visual_pages[:24]
    pages = _fuse(text_pages, visual_pages, lambda row: (row.get('doc_id'), row['page']),
                  visual_weight, k=k, method=method)
    return [dict(row, blocks=row.get('blocks', [])) for row in pages]


def page_heads(text_pages, visual_pages, text_count, visual_count):
    return list(dict.fromkeys(row['page'] for row in [*visual_pages[:visual_count], *text_pages[:text_count]]))


def select_pages(pages, *, page_count, limit, anchors=(), references=(), protected=()):
    if references:
        pages = interleave_pages([references, pages])
    if anchors:
        ranked = {page['page']: page for page in pages}
        forced = [ranked.get(number, {'page': number, 'score': 0.0, 'blocks': []}) for number in anchors]
        pages = forced + [page for page in pages if page['page'] not in anchors]
    if limit >= 6 and len(pages) > limit:
        boundary = limit - 2
        neighbors = {row['page'] + offset for row in pages[:2] for offset in (-1, 1)
                     if 1 <= row['page'] + offset <= page_count}
        promoted = [page for page in pages[boundary:] if page['page'] in neighbors][:2]
        promoted_ids = {page['page'] for page in promoted}
        pages = pages[:boundary] + promoted + [page for page in pages[boundary:] if page['page'] not in promoted_ids]
    if not protected or len(pages) <= limit:
        return pages

    front, rest = pages[:limit], pages[limit:]
    rest_by_number = {page['page']: page for page in rest}
    preserve = {*anchors, *protected, *(row['page'] for row in references)}
    slots = [i for i in reversed(range(limit)) if front[i]['page'] not in preserve]
    missing = [rest_by_number[number] for number in protected if number in rest_by_number][:len(slots)]
    if not missing:
        return pages
    missing_ids = {row['page'] for row in missing}
    displaced = set(slots[:len(missing)])
    return ([row for i, row in enumerate(front) if i not in displaced] + missing
            + [row for i, row in enumerate(front) if i in displaced]
            + [row for row in rest if row['page'] not in missing_ids])


def rank_documents(pages, second_weight=0.15):
    by_document = {}
    for page in pages:
        by_document.setdefault(page['doc_id'], []).append(page['score'])
    docs = []
    for document, scores in by_document.items():
        scores.sort(reverse=True)
        score = scores[0] + second_weight * (scores[1] if len(scores) > 1 else 0.0)
        docs.append({'doc_id': document, 'score': score})
    return sorted(docs, key=lambda row: row['score'], reverse=True)


def fuse_documents(text_docs, visual_docs, visual_weight=1.0, *, k=10, method='rrf'):
    return _fuse(text_docs, visual_docs, lambda row: row['doc_id'], visual_weight, k=k, method=method)
