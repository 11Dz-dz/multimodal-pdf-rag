"""Table representation for the PDF retrieval pipeline."""
import re
import unicodedata
from decimal import Decimal
from html.parser import HTMLParser
from statistics import median


class HtmlCells(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.parts = []

    def handle_starttag(self, tag, attributes):
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            attrs = dict(attributes)
            try:
                rowspan = max(1, min(200, int(attrs.get('rowspan', 1))))
                colspan = max(1, min(200, int(attrs.get('colspan', 1))))
            except (ValueError, TypeError):
                rowspan, colspan = 1, 1
            self.cell = (rowspan, colspan, tag == 'th')
            self.parts = []
        elif tag == 'br' and self.cell:
            self.parts.append(' ')

    def handle_data(self, data):
        if self.cell:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell and self.row is not None:
            self.row.append((' '.join(''.join(self.parts).split()), *self.cell))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None


def structured_html(html):
    parser = HtmlCells()
    parser.feed(html if isinstance(html, str) else '')
    grid, header_count, merged = {}, 0, False
    for row_index, row in enumerate(parser.rows):
        column = 0
        if row_index == header_count and all(cell[3] for cell in row):
            header_count += 1
        for text, down, across, _ in row:
            while (row_index, column) in grid:
                column += 1
            for y in range(row_index, min(row_index + down, len(parser.rows))):
                for x in range(column, column + across):
                    if (y, x) in grid or x >= 200:
                        return [], 0, False
                    grid[y, x] = text
            merged |= down > 1 or across > 1
            # A spanning top header combined with a following subheader row.
            if row_index == 0 and across > 1 and len(parser.rows) > 2:
                header_count = max(header_count, 2)
            column += across
    if not grid:
        return [], 0, False
    width = max(x for _, x in grid) + 1
    rows = [[grid.get((y, x), '') for x in range(width)] for y in range(len(parser.rows))]
    return rows, min(max(1, header_count), len(rows) - 1), merged


def geometric_grid(boxes, expected_columns=None):
    if len(boxes) < 4:
        return []
    centers = lambda cell: (cell['bbox'][1] + cell['bbox'][3]) / 2
    height = median(max(1, b['bbox'][3] - b['bbox'][1]) for b in boxes)
    lines = []
    for box in sorted(boxes, key=centers):
        if not lines or abs(centers(box) - median(map(centers, lines[-1]))) > max(2, height * 0.55):
            lines.append([])
        lines[-1].append(box)
    lines = [sorted(line, key=lambda b: b['bbox'][0]) for line in lines]
    width = expected_columns or len(lines[0])
    if len(lines) < 2 or width < 2 or any(len(line) != width for line in lines):
        return []
    # Establish one global set of column intervals across all rows.
    limits = [(median(line[i]['bbox'][0] for line in lines), median(line[i]['bbox'][2] for line in lines))
              for i in range(width)]
    boundaries = [(limits[i][1] + limits[i + 1][0]) / 2 for i in range(width - 1)]
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        return []
    result = []
    for line in lines:
        cells = [''] * width
        for box in line:
            x0, _, x1, _ = box['bbox']
            column = sum((x0 + x1) / 2 > boundary for boundary in boundaries)
            if cells[column]:
                return []
            cells[column] = box['text']
        result.append(cells)
    return result


def table_text(boxes, html):
    structural, headers, merged = structured_html(html)
    geometry = geometric_grid(boxes, expected_columns=len(structural[0]) if structural else None)
    # MinerU supplies explicit spans. Geometry alone cannot distinguish spans from missed OCR.
    if geometry and not merged and (not structural or len(geometry) == len(structural)):
        rows, method = (geometry, 'paddleocr-geometry')
        headers = headers or 1
    elif len(structural) >= 2 and len(structural[0]) >= 2:
        rows, method = (structural, 'mineru-spans' if merged else 'mineru-html')
    else:
        return '', 'image-only'
    labels = []
    for index in range(len(rows[0])):
        titles = dict.fromkeys(row[index].strip() for row in rows[:headers] if row[index].strip())
        labels.append(' / '.join(titles) or f'第{index + 1}列')
    return '\n'.join(' | '.join(f'{key}：{value}' for key, value in zip(labels, row) if value)
                     for row in rows[headers:] if any(row)), method


def table_fields(text):
    rows = []
    for line in text.splitlines():
        cells = [part.partition('：') for part in line.split(' | ')]
        if len(cells) >= 2 and all(label and separator and value for label, separator, value in cells):
            rows.append({label.strip(): value.strip() for label, _, value in cells})
    return rows


def field_name(text):
    return re.sub(r'\s*\[\d+\]|[↑↓]', '', text).strip()


def mention(question, text):
    if not text or re.fullmatch(r'[-+.\d]+', text):
        return None
    # ASCII names need word boundaries even when the surrounding question is Chinese.
    return re.search(r'(?<![a-z0-9])' + re.escape(text) + r'(?![a-z0-9])', question, re.I)


def table_comparison(question, text):
    """Answer an explicit lookup or subtraction when named cells resolve uniquely."""
    difference = re.search(r'高多少|低多少|多多少|少多少|差多少|差值|\bdifference\b', question, re.I)
    lookup = re.search(r'分别|各(?:是|为|多少)|\brespectively\b', question, re.I)
    if not (difference or lookup) or re.search(r'%|百分比|倍数|平均|方差|比率', question):
        return None
    rows = table_fields(text)
    if not rows or any(row.keys() != rows[0].keys() for row in rows):
        return None

    answers = []
    for metric in rows[0]:
        name = field_name(metric)
        header = unicodedata.normalize('NFKC', name)
        base_name = re.sub(r'\s*\((?:元|美元|港元|人民币|万元|亿元)\)\s*$', '', header)
        if not (mention(question, name) or mention(question, base_name)):
            continue
        if not all(re.fullmatch(r'-?\d+(?:\.\d+)?', row[metric]) for row in rows):
            continue
        for field in rows[0].keys() - {metric}:
            matches = [row for row in rows if mention(question, field_name(row[field]))]
            # A mentioned category, such as a dataset or object, must match the same row.
            for other in rows[0].keys() - {field, metric}:
                allowed = {row[other] for row in rows if mention(question, field_name(row[other]))}
                if allowed:
                    matches = [row for row in matches if row[other] in allowed]
            if len(matches) >= 2 and len({row[field] for row in matches}) == len(matches):
                answers.append((name, [(field_name(row[field]), row[metric]) for row in matches]))

    # Some tables put metrics down the first column and methods across the header.
    first_column = next(iter(rows[0]))
    for row in rows:
        name = field_name(row[first_column])
        if mention(question, name):
            cells = [(field_name(label), value) for label, value in row.items() if label != first_column
                     and mention(question, field_name(label)) and re.fullmatch(r'-?\d+(?:\.\d+)?', value)]
            if len(cells) >= 2:
                answers.append((name, cells))
    if len(answers) != 1:
        return None
    name, cells = answers[0]
    cells.sort(key=lambda cell: mention(question, cell[0]).start())
    if difference and len(cells) != 2:
        return None
    if difference and difference[0] in {'低多少', '少多少'}:
        cells.reverse()
    return name, cells, bool(difference)


def format_table_comparison(comparison, *, chinese):
    metric, cells, difference = comparison
    if chinese:
        answer = '，'.join(f'{name} 的 {metric} 为 {value}' for name, value in cells) + '。'
    else:
        answer = ', '.join(f'{name} has a {metric} of {value}' for name, value in cells) + '.'
    if difference:
        first, second = (value for _, value in cells)
        delta = format(Decimal(first) - Decimal(second), 'f')
        equation = f'{first} − {second} = {delta}'
        answer += f'按上述顺序相减，{equation}。' if chinese else f' The difference is {equation}.'
    return answer
