"""Run PaddleOCR in its own environment and return box-level results as JSON."""
import json
import sys
from pathlib import Path

from paddleocr import PaddleOCR


def main():
    request, output = map(Path, sys.argv[1:])
    paths = json.loads(request.read_text(encoding='utf-8'))
    ocr = PaddleOCR(
        device='cpu', lang='ch', ocr_version='PP-OCRv5',
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='PP-OCRv5_mobile_rec',
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, enable_mkldnn=False, cpu_threads=4,
    )
    records = {}
    for path in paths:
        result = next(iter(ocr.predict(path)), None)
        cells = []
        if result is not None:
            for polygon, text, score in zip(result['rec_polys'], result['rec_texts'], result['rec_scores']):
                if float(score) < 0.50 or not text.strip():
                    continue
                points = [[float(x), float(y)] for x, y in polygon]
                xs, ys = zip(*points)
                cells.append({'text': text, 'score': float(score),
                              'bbox': [min(xs), min(ys), max(xs), max(ys)]})
        records[path] = cells
    output.write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
