"""Replay page and document fusion from frozen text and visual ranks."""
import argparse
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

from raganything.rank_fusion import fuse_documents, fuse_pages, page_heads, rank_documents, select_pages


PAGE_WEIGHTS = (0.8, 1.0, 1.2, 1.5)
DOC_WEIGHTS = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
METHODS = [('rrf', k) for k in (5, 10, 20, 60)] + [('minmax', None)]


def page_profiles():
    for (method, k), weight, (text_heads, visual_heads) in product(
            METHODS, PAGE_WEIGHTS, ((0, 0), (2, 3), (2, 2))):
        yield dict(method=method, k=k, weight=weight, text_heads=text_heads, visual_heads=visual_heads)


def document_profiles():
    for (method, k), weight, second_weight, visual_limit in product(
            METHODS, DOC_WEIGHTS, (0.0, 0.15), (24, 60, 200)):
        yield dict(method=method, k=k, weight=weight, second_weight=second_weight, visual_limit=visual_limit)


def pages(data, *, corpus=False):
    if corpus:
        return [{'doc_id': document, 'page': number, 'score': score} for document, number, score in data]
    return [{'page': number, 'score': score} for number, score in data]


def numbers(ranking):
    return [row['page'] for row in ranking]


def choose_pages(text, visual, profile, context):
    ranked = fuse_pages(text, visual, profile['weight'], k=profile['k'], method=profile['method'])
    heads = page_heads(text, visual, profile['text_heads'], profile['visual_heads'])
    return numbers(select_pages(ranked, **context, protected=heads)[:8])


def holdout_document(document):
    return hashlib.sha256(document.encode()).digest()[0] % 4 == 0


def audit_page_failure(row, text, visual, context, gold):
    if row['document']['rrf_1.2']['complete']:
        return None
    union = set(row['document']['text']['selected']) | set(row['document']['visual']['selected'])
    if not gold <= union:
        return None
    plain = fuse_pages(text, visual, 1.2)
    before_rules = set(numbers(plain[:8]))
    final = row['document']['rrf_1.2']['selected']
    missing = gold - set(final)
    text_top, visual_top = set(numbers(text[:8])), set(numbers(visual[:8]))
    text_all, visual_all = set(numbers(text)), set(numbers(visual))
    strong_single = (text_top - visual_all) | (visual_top - text_all)
    shared = text_all & visual_all
    return {'id': row['id'], 'gold': sorted(gold), 'missing': sorted(missing),
            'raw_top8': numbers(plain[:8]), 'final_top8': final,
            'raw_fusion_loss': bool(missing - before_rules),
            'selection_only_loss': bool(missing & before_rules),
            'strong_single_route_missing': sorted(missing & strong_single),
            'shared_pages_in_raw_top8': len(before_rules & shared),
            'anchors': context['anchors'], 'reference_pages': numbers(context['references'])}


def evaluate(rows, archive):
    profiles, routes = list(page_profiles()), list(document_profiles())
    page_counts = [Counter() for _ in profiles]
    route_counts = [Counter() for _ in routes]
    audits, route_audits = [], []
    page_total, page_holdout, doc_holdout = 0, 0, 0
    archived_metrics = Counter()
    for row in rows:
        previous = archive[row['id']]
        target = row['document_id']
        holdout = holdout_document(target)
        fold = 'holdout' if holdout else 'tuning'
        doc_holdout += holdout
        text = pages(row['corpus_raw']['text'], corpus=True)
        visual = pages(row['corpus_raw']['visual'], corpus=True)
        old_text_top = previous['corpus']['text']['hit1']
        old_visual_top = previous['corpus']['visual']['hit1']
        old_fused_top = previous['corpus']['rrf_1.0']['hit1']
        archived_metrics.update(text_doc_hit1=old_text_top, old_fusion_doc_hit1=old_fused_top,
                                **{f'{fold}_text_doc_hit1': old_text_top})
        assert text[0]['doc_id'] == previous['corpus']['text']['ranked_documents'][0]
        assert visual[0]['doc_id'] == previous['corpus']['visual']['ranked_documents'][0]
        old_consensus_lost = old_text_top and old_visual_top and not old_fused_top
        if old_consensus_lost:
            route_audits.append({'id': row['id'], 'target': target, 'text_first_page': text[0]['page'],
                                 'visual_first_page': visual[0]['page'],
                                 'old_first_document': previous['corpus']['rrf_1.0']['ranked_documents'][0]})
        text_docs = {s: rank_documents(text, s) for s in (0.0, 0.15)}
        visual_docs = {(s, cap): rank_documents(visual[:cap], s) for s, cap in product((0.0, 0.15), (24, 60, 200))}
        for counts, profile in zip(route_counts, routes):
            t_docs = text_docs[profile['second_weight']]
            v_docs = visual_docs[profile['second_weight'], profile['visual_limit']]
            ranked = fuse_documents(t_docs, v_docs, profile['weight'], k=profile['k'], method=profile['method'])
            correct = ranked[0]['doc_id'] == target
            consensus = t_docs[0]['doc_id'] == v_docs[0]['doc_id'] == target
            counts.update(hit1=correct, hit5=target in {r['doc_id'] for r in ranked[:5]},
                          gained_vs_text=correct and not old_text_top, lost_vs_text=old_text_top and not correct,
                          consensus=consensus, lost_consensus=consensus and not correct,
                          rescued_old_six=old_consensus_lost and correct, **{f'{fold}_hit1': correct})
            assert not consensus or correct
        if 'document_raw' not in row:
            continue
        page_total += 1
        page_holdout += holdout
        raw = row['document_raw']
        text, visual = pages(raw['text']), pages(raw['visual'])
        refs = [{'page': n, 'score': 0.0} for n in raw['references']]
        context = dict(page_count=raw['page_count'], limit=8, anchors=raw['anchors'], references=refs)
        gold = set(row['gold_pages'])
        old_page = previous['document']['rrf_1.2']['complete']
        archived_metrics.update(old_fusion_page_complete=old_page,
                                text_page_complete=previous['document']['text']['complete'],
                                **{f'{fold}_old_fusion_page_complete': old_page})
        reproduced = choose_pages(text, visual, dict(method='rrf', k=60, weight=1.2, text_heads=0, visual_heads=0),
                                  context)
        assert reproduced == row['document']['rrf_1.2']['selected'] == previous['document']['rrf_1.2']['selected'], row['id']
        audit = audit_page_failure(row, text, visual, context, gold)
        if audit:
            audits.append(audit)
        for counts, profile in zip(page_counts, profiles):
            selected = choose_pages(text, visual, profile, context)
            complete = gold <= set(selected)
            counts.update(complete=complete, recovered_46=audit is not None and complete,
                          gained_vs_old=complete and not old_page, lost_vs_old=old_page and not complete,
                          cross_page_complete=complete and row['category'] == 'cross_page',
                          **{f'{fold}_complete': complete})

    assert len(rows) == 863 and page_total == 842 and len(audits) == 46 and len(route_audits) == 6
    page_results = [dict(config=config, **counts) for config, counts in zip(profiles, page_counts)]
    doc_results = [dict(config=config, **counts) for config, counts in zip(routes, route_counts)]
    return {
        'known_pdf': {'questions': page_total, 'holdout_questions': page_holdout,
                      'tuning_winner': max(page_results, key=lambda r: r['tuning_complete']),
                      'profiles': sorted(page_results, key=lambda r: (-r['complete'], r['lost_vs_old'])),
                      'audit': {'tracked_misses': len(audits),
                                'raw_fusion_loss': sum(row['raw_fusion_loss'] for row in audits),
                                'selection_only_loss': sum(row['selection_only_loss'] for row in audits),
                                'strong_single_route_missing': sum(bool(row['strong_single_route_missing']) for row in audits),
                                'cases': audits}},
        'open_corpus': {'questions': len(rows), 'holdout_questions': doc_holdout,
                        'tuning_winner': max(doc_results, key=lambda r: r['tuning_hit1']),
                        'profiles': sorted(doc_results, key=lambda r: (-r['hit1'], -r['hit5'])),
                        'original_consensus_misses': route_audits},
        'archived_metrics': archived_metrics,
    }


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(read_rows(args.ranks), {row['id']: row for row in read_rows(args.archive)})
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: result[name]['profiles'][0] for name in ('known_pdf', 'open_corpus')}))
