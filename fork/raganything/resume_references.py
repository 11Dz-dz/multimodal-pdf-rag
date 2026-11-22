"""Resolve numbered figure and table captions to PDF regions."""
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pymupdf

from .utils import normalize_caption_list


REFERENCE = re.compile(
    r'(图|表|(?<!\w)(?:fig(?:ure)?|tab(?:le)?)\.?)\s*([a-z]?\d+(?:[.-]\d+)*[a-z]?)(?![a-z0-9])',
    re.IGNORECASE,
)
NAMES = {'image': '图', 'table': '表'}
IDENTIFIER = re.compile(r'(?<![A-Za-z0-9])[A-Z][A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*(?![A-Za-z0-9])')
PAGE_REFERENCE = re.compile(
    r'(?<!\w)(?:PDF\s+)?pages?\s+(\d+)(?:\s*(?:and|to|[-–])\s*(\d+))?(?!\w|\.\d)'
    r'|(?:PDF\s*)?第\s*(\d+)\s*页', re.I,
)
FRONT_PAGE = re.compile(r'\b(?:cover|title\s+page|first\s+page|front\s+page|logo)\b|封面|首页|标志', re.I)
CONTENTS_PAGE = re.compile(r'\b(?:table\s+of\s+contents|contents\s+page)\b|目录', re.I)
BACK_PAGE = re.compile(r'\b(?:back\s+cover|last\s+page|final\s+page)\b|封底|最后一页', re.I)


def reference_key(match):
    label, number = match.groups()
    return ('image' if label == '图' or label.lower().startswith('fig') else 'table', number.upper())


def requested_references(question):
    query = unicodedata.normalize('NFKC', question)
    return list(dict.fromkeys(reference_key(match) for match in REFERENCE.finditer(query)))


def requested_pages(question, page_count=None):
    query = unicodedata.normalize('NFKC', question)
    pages = {}
    for match in PAGE_REFERENCE.finditer(query):
        start, end, chinese = match.groups()
        first = int(start or chinese)
        pages[first] = None
        if end:
            if re.search(r'\band\b', match[0], re.I):
                pages[int(end)] = None
            elif 0 < int(end) - first <= 3:
                pages.update(dict.fromkeys(range(first + 1, int(end) + 1)))
    if FRONT_PAGE.search(query):
        pages[1] = None
    if CONTENTS_PAGE.search(query):
        pages.update(dict.fromkeys((1, 2, 3)))
    if page_count is not None:
        if BACK_PAGE.search(query):
            pages[page_count] = None
        return [page for page in pages if 1 <= page <= page_count]
    return list(pages)


def label_for(key):
    kind, number = key
    return f'{NAMES[kind]} {number}'


def pdf_rect(page, bbox):
    x0, y0, x1, y1 = bbox
    width, height = page.rect.width / 1000, page.rect.height / 1000
    return pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height) & page.rect


def nearest_region(page, caption_rect, blocks):
    candidates = []
    for block in blocks:
        if not block.get('bbox'):
            continue
        region = pdf_rect(page, block['bbox'])
        overlap = min(caption_rect.x1, region.x1) - max(caption_rect.x0, region.x0)
        if overlap < min(caption_rect.width, region.width) * 0.20:
            continue
        gap = max(caption_rect.y0 - region.y1, region.y0 - caption_rect.y1, 0)
        if gap <= page.rect.height * 0.12:
            candidates.append((gap, block))
    candidates.sort(key=lambda candidate: candidate[0])
    if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] > 2):
        return candidates[0][1]
    return None


def caption_targets(pdf, blocks, items):
    by_page_kind = defaultdict(list)
    by_asset = {}
    for block in blocks:
        if block['kind'] in NAMES and block.get('image_path'):
            by_page_kind[block['page'], block['kind']].append(block)
            by_asset[block['page'], Path(block['image_path']).name] = block

    native, parsed, descriptions = defaultdict(dict), defaultdict(dict), {}
    for item in items:
        item_type = item['type']
        kind = 'image' if item_type == 'chart' else item_type
        if kind not in NAMES or not item.get('img_path'):
            continue
        page_number = item['page_idx'] + 1
        block = by_asset.get((page_number, Path(item['img_path']).name))
        if not block:
            continue
        captions = normalize_caption_list(item.get(item_type + '_caption', []))
        keys = set()
        for caption in captions:
            match = REFERENCE.match(unicodedata.normalize('NFKC', caption.strip()))
            if match:
                key = reference_key(match)
                descriptions.setdefault((page_number, key), caption)
                if key[0] == kind:
                    keys.add(key)
        if len(keys) == 1:
            parsed[keys.pop()][block['id']] = block

    for page_number in sorted({number for number, _ in by_page_kind}):
        page = pdf[page_number - 1]
        for entry in page.get_text('dict', flags=pymupdf.TEXTFLAGS_TEXT)['blocks']:
            for line in entry.get('lines', []):
                text = unicodedata.normalize('NFKC', ''.join(span['text'] for span in line['spans']).strip())
                match = REFERENCE.match(text)
                if not match or not re.match(r'\s*[.:：。]', text[match.end():]):
                    continue
                key = reference_key(match)
                target = nearest_region(page, pymupdf.Rect(line['bbox']), by_page_kind[page_number, key[0]])
                if target:
                    native[key][target['id']] = target

    return {**parsed, **native}, descriptions


def scoped_evidence(pdf, block, key, caption, visual):
    label = label_for(key)
    text = block['text'] if key[0] == 'table' else block.get('ocr', '')
    images = [block['image_path']] if visual else []
    native = ''
    if block.get('bbox'):
        page = pdf[block['page'] - 1]
        bbox = list(block['bbox'])
        if key[0] == 'image':
            # Figure labels can sit immediately above the image in the PDF text layer.
            bbox[1] = max(0, bbox[1] - 24)
        region = pdf_rect(page, bbox)
        native = page.get_text(clip=region, sort=True).strip()
        if native:
            if key[0] == 'image' or len(native.splitlines()) >= 3:
                text = native
        if visual and key[0] == 'image':
            top = pymupdf.Rect(region.x0, region.y0, region.x1, pdf_rect(page, block['bbox']).y0)
            if top.height > 0 and page.get_text(clip=top).strip():
                scale = min(3, 1400 / max(region.width, region.height))
                images = [page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=region).tobytes('jpeg')]
    structured = block['text'] if block.get('table_method') in {'mineru-html', 'paddleocr-geometry'} else ''
    excerpt = '\n'.join(part for part in (label, caption, text[:6000]) if part)
    return {'page': block['page'], 'text': excerpt, 'reference': label,
            'images': images, 'has_structure': key[0] == 'table', 'structured': structured,
            'figure_text': native if key[0] == 'image' else ''}


def figure_column_labels(question, native):
    if not re.search(r'从左到右|left to right', question, re.I):
        return []
    if not re.search(r'标题|标签|headers?|labels?|titles?', question, re.I):
        return []
    count = re.search(r'([2-9二三四五六七八九])\s*列|\b([2-9])\s+columns?\b', question, re.I)
    if not count or '\n' in native:
        return []
    requested = next(number for number in count.groups() if number)
    amount = '零一二三四五六七八九'.find(requested) if not requested.isascii() else int(requested)
    labels = re.split(r'\s{2,}', native) if native else []
    return labels if len(labels) == amount and len(set(labels)) == amount else []


def missing_table_identifiers(question, text):
    """A named model or dataset must occur in the specified table or its caption."""
    if not re.search(r'多少|分别|各(?:是|为)|\b(?:value|values|how much)\b', question, re.I):
        return []
    identifiers = dict.fromkeys(IDENTIFIER.findall(unicodedata.normalize('NFKC', question)))
    missing = []
    for name in identifiers:
        if not re.search(r'\d|[a-z][A-Z]', name):
            continue
        parts = re.split(r'[-_.]', name)
        expression = r'(?<![A-Za-z0-9])' + r'[-_.\s–‑]*'.join(map(re.escape, parts)) + r'(?![A-Za-z0-9])'
        if not re.search(expression, text, re.I):
            missing.append(name)
    return missing
