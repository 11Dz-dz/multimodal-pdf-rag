"""Page retrieval and refusal evaluation for the resume profile."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import UUID


def evaluate(url, document_id, cases):
    report = []
    for case in cases:
        payload = json.dumps({'question': case['question'], 'visual': True}).encode()
        request = Request(f'{url.rstrip("/")}/api/documents/{document_id}/ask', data=payload,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=240) as response:
            output = json.load(response)
        trace = output['trace']
        gold = set(case['pages'])
        pages = [candidate['page'] for candidate in trace['pages']]
        rank = next((i for i, page in enumerate(pages, 1) if page in gold), None)
        cause = None
        if gold and (rank is None or rank > 5):
            fused_pages = {candidate['page'] for candidate in trace['rrf']}
            cause = 'ranking_after_fusion' if gold & fused_pages else 'not_in_fusion_candidates'
        report.append({'question': case['question'], 'gold_pages': sorted(gold), 'rank': rank,
                       'first_five': pages[:5], 'retrieval_diagnosis': cause,
                       'refused': output['refused'], 'answer': output['answer'], 'citations': output['citations']})
    positives = [row for row in report if row['gold_pages']]
    negatives = [row for row in report if not row['gold_pages']]
    ratio = lambda number, rows: number / len(rows) if rows else None
    return {'answerable': len(positives), 'unanswerable': len(negatives),
            'hit_at_5': ratio(sum(row['rank'] is not None and row['rank'] <= 5 for row in positives), positives),
            'mrr_at_returned_pages': ratio(sum(1 / row['rank'] for row in positives if row['rank']), positives),
            'false_refusal_rate': ratio(sum(row['refused'] for row in positives), positives),
            'correct_refusal_rate': ratio(sum(row['refused'] for row in negatives), negatives), 'cases': report}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8094')
    parser.add_argument('--doc-id', type=UUID, required=True)
    parser.add_argument('--cases', type=Path, required=True, help='JSON array with question and one-based pages')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    print(json.dumps(evaluate(args.url, args.doc_id, cases), ensure_ascii=False, indent=2))
