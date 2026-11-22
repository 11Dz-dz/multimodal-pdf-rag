"""Compare two QA cohorts using offline retrieval scores and literal citation markers."""

import argparse
import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path


YEAR = re.compile(r'(?<!\d)(?:19|20)\d{2}(?!\d)')
WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
LABEL = re.compile(r'"([^"\n]{1,65})"|(?<![\w])\x27([^\x27\n]{2,65})\x27(?![\w])')
ANSWER_FORMAT = re.compile(
    r'(?:[?\n]\s*|(?<=[.])\s+)(?:directly\s+)?'
    r'(?:answer|write|give|please|provide|respond|round|format|state|return)\b', re.IGNORECASE)
NO_NAME = set(('A An The What Which Who Whose When Where Why How If Is Are Does Do Did Can Could Would '
               'Will In On As Of At To For By Or And From With According Given Between Among Figure Fig '
               'Table Page Pages Section Chapter Appendix Part Wave Total Please List Name Write Using '
               'Calculate Provide Answer Yes No None Example Excluding State True False Compared Suppose '
               'First Second Last Note Diagram Plot Image Slide Year FY For Example Main Include Correct '
               'Count Refer Return Assume Shown Based Looking Read Document Report Directly Tell After Before '
               'January February March April May June July August September October November December '
               'Mon Tue Wed Thu Fri Sat Sun Monday Tuesday Wednesday Thursday Friday Saturday Sunday '
               'English Chinese Japanese German French YYYY MM DD').split())
LINKERS = {'of', 'and', 'the', 'de', 'van', 'von'}
FISCAL_YEAR = re.compile(r'fy\s*(?:19|20)?\d{2}', re.IGNORECASE)
METRIC_GROUPS = [
    ('earnings per diluted share', 'diluted earnings per share', 'diluted eps'),
    ('basic earnings per share', 'basic earnings per ordinary share', 'basic eps'),
    ('earnings per share', 'earnings per ordinary share', 'eps'),
    ('debt to ebitda ratio', 'debt to ebitda', 'debt ebitda ratio'),
    ('ebitda', 'earnings before interest taxes depreciation and amortization'),
    ('gdp growth', 'gross domestic product growth'),
    ('gdp', 'gross domestic product'),
    ('return on equity', 'roe'), ('return on assets', 'roa'), ('return on investment', 'roi'),
    ('operating margin', 'operating profit margin'),
    ('gross margin', 'gross profit margin'),
    ('net profit margin', 'net margin'), ('market capitalization', 'market cap'),
    ('free cash flow', 'fcf'), ('net income', 'net earnings'),
    ('operating income', 'operating profit'), ('gross profit',),
    ('total revenue', 'net revenue', 'net sales'), ('total debt',), ('long term debt',),
    ('depreciation expense',), ('total assets',), ('market share',), ('dividend yield',),
    ('dividends paid', 'dividend paid'), ('total operating expenses', 'operating expenses'),
    ('research and development expenses', 'r d expenses'),
    ('cost of revenue', 'cost of sales', 'cost of goods sold', 'cogs'),
    ('operating cash flow', 'cash flow from operating activities'), ('capital expenditure', 'capex'),
    ('compound annual growth rate', 'cagr'),
    ('r d to asset ratio', 'research and development to asset ratio'),
    ('mean average precision', 'map'), ('average precision',),
    ('mean squared error', 'mse'), ('root mean squared error', 'rmse'), ('mean absolute error', 'mae'),
    ('intersection over union', 'iou'), ('f1 score', 'f1'),
    ('false positive rate', 'fpr'), ('true positive rate', 'tpr'),
    ('area under the curve', 'auc', 'auroc'), ('exact match',),
    ('top 1 accuracy', 'top1 accuracy'), ('top 5 accuracy', 'top5 accuracy'),
    ('inference time', 'inference latency'), ('training time',),
    ('token throughput', 'tokens per second'), ('error rate',), ('success rate',),
    ('mortality rate',), ('unemployment rate',), ('inflation rate',), ('interest rate',),
    ('response rate',), ('conversion rate',), ('participation rate',),
    ('population density',), ('total population',), ('total expenditure',),
    ('human development index', 'hdi'), ('body mass index', 'bmi'),
    ('absolute humidity',), ('relative humidity',), ('mean temperature',), ('average temperature',),
    ('accuracy',), ('precision',), ('recall',), ('perplexity',), ('relative error',), ('absolute error',),
]


def normalized(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    value = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', value)
    value = re.sub(r'[^\w]+|_', ' ', value)
    return ' '.join(value.split())


def contains(text, term):
    return f' {term} ' in f' {text} '


def metric_markers(words, question):
    active = []
    for group in METRIC_GROUPS:
        display = next((name for name in group if contains(words, name)
                        and (name != 'map' or re.search(r'\bmAP\b', question))), None)
        if display:
            active.append((display, group))
    return [(display, aliases) for display, aliases in active
            if not any(display != other and contains(other, display) for other, _ in active)]


def proper_names(question):
    names = []
    tokens = list(WORD.finditer(question))
    chain = []
    previous = None
    for i, match in enumerate(tokens):
        word = match.group()
        title = word[:1].isupper() and word not in NO_NAME and len(word) > 1 and not FISCAL_YEAR.fullmatch(word)
        connected = previous is not None and bool(re.fullmatch(r'[\s.\-–]*', question[previous.end():match.start()]))
        following = tokens[i + 1].group() if i + 1 < len(tokens) else ''
        internal = (connected and bool(chain) and word in LINKERS
                    and following[:1].isupper() and following not in NO_NAME
                    and not (word == 'and' and (chain[-1].isupper() or following.isupper())))
        if chain and not (connected and (title or internal)):
            names.append(' '.join(chain))
            chain = []
        if title or internal:
            if i != 0 or not word.istitle():
                chain.append(word)
        elif (re.search(r'[a-z][A-Z]|[A-Za-z]\d', word) and word.casefold() not in {'top1', 'top5'}
              and not re.fullmatch(r'(?:table|fig|figure|page|chapter|section)\d+', word, re.IGNORECASE)):
            names.append(word)
        previous = match
    if chain:
        names.append(' '.join(chain))
    for match in LABEL.finditer(question):
        label = match[1] or match[2]
        compact = normalized(label)
        if any(c.isalpha() for c in compact) and compact not in {
                'yes', 'no', 'none', 'not answerable', 'true', 'false', 'left', 'right', 'top', 'middle', 'bottom'}:
            names.append(label)
    names.extend(match.group() for match in re.finditer(r'\b(?:U\.S\.|U\.K\.)(?=[\s,]|$)', question))
    return list(dict.fromkeys(normalized(name) for name in names if normalized(name)
                              and not re.fullmatch(r'(?:table|fig|figure|page|chapter|section)\s*\d+',
                                                   name, re.IGNORECASE)))


def page_markers(question, page_text):
    question = ANSWER_FORMAT.split(question, maxsplit=1)[0]
    words = normalized(question)
    metrics = metric_markers(words, question)
    metric_aliases = {alias for aliases in METRIC_GROUPS for alias in aliases if contains(words, alias)}
    common_names = {'u s': ('u s', 'united states'), 'u k': ('u k', 'united kingdom')}
    names = [(name, common_names.get(name, (name,))) for name in proper_names(question) if name not in metric_aliases
             and not YEAR.fullmatch(name) and not FISCAL_YEAR.fullmatch(name)]
    years = [(year, (year,)) for year in dict.fromkeys(YEAR.findall(question))]
    sources = {int(page): (str(text), normalized(text)) for page, text in page_text.items()}
    result = {}
    for kind, terms in [('name_or_label', names), ('year', years), ('metric', metrics)]:
        matches = {}
        for display, aliases in terms:
            if kind == 'year':
                pages = [page for page, (original, _) in sources.items()
                         if display in YEAR.findall(original)]
            else:
                pages = [page for page, (original, content) in sources.items()
                         if any(contains(content, alias) and (alias != 'map' or re.search(r'\bmAP\b', original))
                                for alias in aliases)]
            matches[display] = sorted(pages)
        result[kind] = matches
    return result


def index(path):
    with path.open(encoding='utf-8') as file:
        return {row['id']: row for row in map(json.loads, file)}


def score_stats(values):
    cuts = statistics.quantiles(values, n=100, method='inclusive')
    return {'n': len(values), 'minimum': min(values), 'p10': cuts[9], 'p25': cuts[24],
            'median': cuts[49], 'mean': statistics.mean(values), 'p75': cuts[74], 'p90': cuts[89],
            'maximum': max(values)}


def marker_stats(rows, key):
    eligible = []
    for row in rows:
        values = ([pages for by_kind in row['markers'].values() for pages in by_kind.values()]
                  if key == 'combined' else list(row['markers'][key].values()))
        if values:
            eligible.append(values)
    return {'eligible': len(eligible), 'no_marker': len(rows) - len(eligible),
            'any_match': sum(any(values) for values in eligible),
            'all_match': sum(all(values) for values in eligible)}


def group_stats(rows):
    return {
        'questions': len(rows), 'turns': Counter(row['turns'] for row in rows),
        'turns_gte_2': sum(row['turns'] >= 2 for row in rows),
        'actually_opened_additional_pages': sum(row['opened_new_page'] for row in rows),
        'first_three_exact_order': sum(row['same_initial_order'] for row in rows),
        'first_three_same_set': sum(row['same_initial_pages'] for row in rows),
        'text': score_stats([row['text_rerank_max'] for row in rows]),
        'text_at_least_0_99': sum(row['text_rerank_max'] >= 0.99 for row in rows),
        'text_saturated_at_1': sum(row['text_rerank_max'] == 1 for row in rows),
        'visual': score_stats([row['visual_maxsim_max'] for row in rows]),
        'markers': {key: marker_stats(rows, key) for key in ('combined', 'name_or_label', 'year', 'metric')},
    }


def ranking_auc(pos, neg, key):
    return sum((a[key] > b[key]) + 0.5 * (a[key] == b[key]) for a in pos for b in neg) / (len(pos) * len(neg))


def analyze(replay, final):
    rows = []
    for case_id, raw in replay.items():
        record = final[case_id]
        negative = record['category'] == 'unanswerable'
        expected = 'wrong' if negative else 'correct'
        assert record['verdict'] == expected and set(raw['cited_page_text']) == {
            str(c['page']) for c in record['citations']}
        assert raw['question'] == record['question'] and raw['turns'] == record['turns']
        assert raw['text_rerank_max'] is not None and raw['visual_maxsim_max'] is not None
        row = {key: raw[key] for key in raw if key != 'cited_page_text'}
        row.update(verdict=record['verdict'], cited_pages=sorted(map(int, raw['cited_page_text'])),
                   opened_new_page=bool(set(record['inspected_pages']) - set(record['initial_pages'])),
                   same_initial_order=raw['original_initial_pages'] == raw['replayed_initial_pages'],
                   same_initial_pages=set(raw['original_initial_pages']) == set(raw['replayed_initial_pages']),
                   markers=page_markers(raw['question'], raw['cited_page_text']))
        rows.append(row)
    negative = [r for r in rows if r['verdict'] == 'wrong']
    positive = [r for r in rows if r['verdict'] == 'correct']
    assert len(negative) == 89 and len(positive) == 579
    report = {'unanswerable_wrong': group_stats(negative), 'answerable_correct': group_stats(positive),
              'auc_higher_score_predicts_answerable_correct': {
                  key: ranking_auc(positive, negative, key) for key in ('text_rerank_max', 'visual_maxsim_max')},
              'method': {'scores': 'Local replay of frozen first-round chunk reranking and page MaxSim.',
                         'markers': ('Rule-extracted title-case names, acronyms and quoted phrases; explicit 4-digit years; '
                                     'named metrics and listed aliases. Matched on the union of cited PDF page text, '
                                     'indexed text, structured tables and OCR. No generated image summaries.'),
                         'turns': 'Multi-turn means turns >= 2. New full-page openings are measured separately.',
                         'does_not_establish_support': True}}
    return report, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--final', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report, rows = analyze(index(args.replay), index(args.final))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'confidence-summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (args.out / 'confidence-cases.jsonl').open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
