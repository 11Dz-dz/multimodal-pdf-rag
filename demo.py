"""Run upstream RAG-Anything with local BGE-M3 and Qwen2.5-VL."""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import pymupdf
import torch
from PIL import Image
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
sys.path.insert(0, str(Path(__file__).resolve().parent / 'fork'))
from raganything import RAGAnything, RAGAnythingConfig
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'multimodal-rag-demo'
DEFAULT_MODELS = Path(os.getenv('DEMO_MODEL_DIR', str(LEGACY / 'models')))


def resolve_mineru_items(content_list, *, include_charts=False):
    """Load MinerU legacy content-list JSON and resolve its exported image paths."""
    items = json.loads(content_list.read_text(encoding='utf-8'))
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError('请传入 MinerU 的 *_content_list.json（平铺版）')
    supported = {'text', 'image', 'table'}
    if include_charts:
        supported.add('chart')
    items = [item for item in items if item.get('type') in supported]
    for item in items:
        if item.get('img_path'):
            asset = (content_list.parent / item['img_path']).resolve()
            if not asset.is_relative_to(content_list.parent.resolve()) or not asset.is_file():
                raise ValueError(f'MinerU 图片资源不存在或在输出目录之外: {item["img_path"]}')
            item['img_path'] = str(asset)
    return items


def recognize_image(image, ocr):
    """OCR and a deliberately narrow row/column heuristic for the regular sample table."""
    detected, _ = ocr(str(image))
    boxes = []
    for quad, label, confidence in detected or []:
        if confidence >= 0.55:
            x = min(point[0] for point in quad)
            y0, y1 = min(point[1] for point in quad), max(point[1] for point in quad)
            boxes.append((x, (y0 + y1) / 2, y1 - y0, label))
    if not boxes:
        return [], None
    tolerance = max(5, sorted(box[2] for box in boxes)[len(boxes) // 2] * 0.5)
    rows = []
    for box in sorted(boxes, key=lambda item: item[1]):
        if not rows or abs(box[1] - sum(x[1] for x in rows[-1]) / len(rows[-1])) > tolerance:
            rows.append([])
        rows[-1].append(box)
    rows = [sorted(row, key=lambda item: item[0]) for row in rows]
    labels = [[item[3] for item in row] for row in rows]
    columns = len(rows[0])
    table = None
    if len(rows) >= 3 and columns >= 2 and all(len(row) == columns for row in rows):
        width = max(rows[0][-1][0] - rows[0][0][0], 1)
        aligned = all(abs(cell[0] - header[0]) < width * 0.12
                      for row in rows[1:] for cell, header in zip(row, rows[0]))
        if aligned:
            lines = ['| ' + ' | '.join(value.replace('|', '\\|') for value in row) + ' |' for row in labels]
            lines.insert(1, '| ' + ' | '.join(['---'] * columns) + ' |')
            table = '\n'.join(lines)
    return [' '.join(row) for row in labels], table


def extract_pdf(pdf, assets):
    """Optional light parser. Real text/image extraction and OCR, no MinerU dependency."""
    from rapidocr_onnxruntime import RapidOCR

    assets.mkdir(parents=True, exist_ok=True)
    ocr = RapidOCR(intra_op_num_threads=3, inter_op_num_threads=2)
    items = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            for block in page.get_text('blocks', sort=True):
                if block[6] == 0 and block[4].strip():
                    items.append({'type': 'text', 'text': block[4].strip(), 'page_idx': page.number})
            seen = set()
            for info in page.get_images(full=True):
                xref = info[0]
                if xref in seen:
                    continue
                seen.add(xref)
                image = Image.open(io.BytesIO(doc.extract_image(xref)['image'])).convert('RGB')
                if image.width < 100 or image.height < 70:
                    continue
                image.thumbnail((1600, 1600))
                asset = assets / f'page-{page.number+1}-image-{xref}.jpg'
                image.save(asset, quality=93)
                lines, table = recognize_image(asset, ocr)
                item = {'page_idx': page.number, 'img_path': str(asset.resolve())}
                if table:
                    item.update(type='table', table_body=table, table_caption=[], table_footnote=[])
                else:
                    item.update(type='image', image_caption=lines, image_footnote=[])
                items.append(item)
    return items


class LocalModels:
    def __init__(self, directory, device, answer_model=None, answer_batch_size=1, answer_relay=None):
        self.directory = directory
        self.device = device
        self.answer_model = Path(answer_model or directory / 'qwen-vl').resolve()
        self.answer_relay = answer_relay
        self.answer_identity = (f'{answer_relay.base_url}/{answer_relay.model}' if answer_relay
                                else str(self.answer_model))
        self.embed_model = SentenceTransformer(str(directory / 'bge-m3'), device=device, local_files_only=True)
        self.embed_model.max_seq_length = 1024
        if device.startswith('cuda'):
            self.embed_model.half()
        self.processor = None
        self.model = None
        self.reranker = None
        self.generation_lock = asyncio.Lock()
        self.answer_batch_size = answer_batch_size
        self.pending_answers = []
        self.answer_task = None
        self.image_events = []

    async def embed(self, texts, **_):
        return self.embed_model.encode(texts, normalize_embeddings=True, batch_size=8).astype('float32')

    async def rerank(self, question, texts):
        if self.reranker is None:
            self.reranker = CrossEncoder(str(self.directory / 'reranker'), device=self.device,
                                       max_length=1024, local_files_only=True)
            if self.device.startswith('cuda'):
                self.reranker.model.half()
        scores = await asyncio.to_thread(self.reranker.predict, [(question, text) for text in texts],
                                         batch_size=8, show_progress_bar=False, activation_fn=torch.nn.Sigmoid())
        return scores.reshape(-1)

    async def complete(self, messages, limit):
        if self.answer_relay:
            return await asyncio.to_thread(self.generate, messages, limit)
        if self.answer_batch_size > 1:
            result = asyncio.get_running_loop().create_future()
            self.pending_answers.append((messages, limit, result))
            if self.answer_task is None or self.answer_task.done():
                self.answer_task = asyncio.create_task(self.flush_answers())
            return await result
        async with self.generation_lock:
            return await asyncio.to_thread(self.generate, messages, limit)

    async def flush_answers(self):
        while self.pending_answers:
            if len(self.pending_answers) < self.answer_batch_size:
                await asyncio.sleep(0.10)
            first_limit = self.pending_answers[0][1]
            count = 1
            for _, limit, _ in self.pending_answers[1:self.answer_batch_size]:
                if limit != first_limit:
                    break
                count += 1
            batch = self.pending_answers[:count]
            del self.pending_answers[:count]
            try:
                values = await asyncio.to_thread(self.generate_batch, [messages for messages, _, _ in batch],
                                                 first_limit)
            except Exception as exc:
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
            else:
                for (_, _, future), value in zip(batch, values):
                    if not future.done():
                        future.set_result(value)

    def generate(self, messages, limit):
        if self.answer_relay:
            result = self.answer_relay.complete(messages, limit)
            self.image_events.append(sum(part.get('type') == 'image_url'
                                         for message in messages if isinstance(message['content'], list)
                                         for part in message['content']))
            return result
        return self.generate_batch([messages], limit)[0]

    def generate_batch(self, batch, limit):
        if self.model is None:
            path = str(self.answer_model)
            model_type = AutoConfig.from_pretrained(path, local_files_only=True).model_type
            if model_type == 'qwen3_vl':
                model_class, patch_size = Qwen3VLForConditionalGeneration, 32
                min_tokens, max_tokens = 16, 1200
            else:
                model_class, patch_size = Qwen2_5_VLForConditionalGeneration, 28
                min_tokens, max_tokens = 256, 768
            self.processor = AutoProcessor.from_pretrained(path, min_pixels=min_tokens * patch_size**2,
                                                            max_pixels=max_tokens * patch_size**2,
                                                            local_files_only=True)
            self.processor.tokenizer.padding_side = 'left'
            dtype = torch.bfloat16 if self.device.startswith('cuda') else torch.float32
            self.model = model_class.from_pretrained(
                path, dtype=dtype, device_map={'': self.device}, attn_implementation='sdpa', local_files_only=True).eval()
        texts, images = [], []
        for messages in batch:
            template, sample_images = self.prepare_messages(messages)
            self.image_events.append(len(sample_images))
            images.extend(sample_images)
            texts.append(self.processor.apply_chat_template(template, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=images or None, padding=len(batch) > 1,
                                return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        return [value.strip() for value in self.processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)]

    @staticmethod
    def prepare_messages(messages):
        template_messages, images = [], []
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str):
                content = [{'type': 'text', 'text': content}]
            parts = []
            for part in content:
                if part['type'] == 'text':
                    parts.append({'type': 'text', 'text': part['text']})
                elif part['type'] == 'image_url':
                    url = part['image_url']['url']
                    if not url.startswith('data:image/') or ';base64,' not in url:
                        raise ValueError('只读取 RAG-Anything 提供的本地图像数据')
                    im = Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')
                    images.append(im)
                    parts.append({'type': 'image'})
            template_messages.append({'role': message['role'], 'content': parts})
        return template_messages, images

    async def text(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        if kwargs.get('keyword_extraction'):
            prompt += '\n只返回要求的 JSON 对象。'
        messages = ([{'role': 'system', 'content': system_prompt}] if system_prompt else [])
        messages.extend(history_messages or [])
        messages.append({'role': 'user', 'content': prompt})
        return await self.complete(messages, 640)

    async def vision(self, prompt='', system_prompt=None, history_messages=None,
                     image_data=None, messages=None, **kwargs):
        if messages is None:
            messages = ([{'role': 'system', 'content': system_prompt}] if system_prompt else [])
            messages.extend(history_messages or [])
            content = [{'type': 'text', 'text': prompt}]
            if image_data:
                content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image_data}})
            messages.append({'role': 'user', 'content': content})
        return await self.complete(messages, 520)


class Demo:
    def __init__(self, pdf, model_dir, device, data_dir, content_list=None, *, models=None,
                 workspace='', filename=None):
        self.pdf = pdf.resolve()
        self.filename = filename or self.pdf.name
        self.content_list = content_list.resolve() if content_list else None
        raw = self.pdf.read_bytes() + (self.content_list.read_bytes() if self.content_list else b'ocr-v1')
        self.doc_dir = data_dir.resolve() / hashlib.sha256(raw).hexdigest()[:16]
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.mode = 'MinerU' if self.content_list else 'PyMuPDF + RapidOCR'
        torch.set_num_threads(4)
        self.models = models if models is not None else LocalModels(model_dir, device)
        embed = EmbeddingFunc(embedding_dim=1024, max_token_size=1024, func=self.models.embed,
                              model_name='BAAI/bge-m3')
        self.light = LightRAG(working_dir=str(self.doc_dir / 'index'), workspace=workspace,
                              embedding_func=embed,
                              llm_model_func=self.models.text, llm_model_name='qwen2.5-vl',
                              chunk_token_size=512, chunk_overlap_token_size=48,
                              max_parallel_insert=1, llm_model_max_async=1,
                              entity_extract_max_gleaning=0, default_llm_timeout=240,
                              addon_params={'language': 'Chinese'})
        asset_dir = self.content_list.parent if self.content_list else self.doc_dir / 'assets'
        config = RAGAnythingConfig(working_dir=str(self.doc_dir / 'index'), parser_output_dir=str(asset_dir),
                                   enable_equation_processing=False,
                                   enable_image_processing=True, enable_table_processing=True,
                                   context_mode='page', context_window=0, max_context_tokens=300)
        self.rag = RAGAnything(lightrag=self.light, config=config, embedding_func=embed,
                               llm_model_func=self.models.text, vision_model_func=self.models.vision)
        self.lock = asyncio.Lock()

    async def close(self):
        await self.light.finalize_storages()

    async def initialize(self):
        await self.light.initialize_storages()
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status
            await initialize_pipeline_status()
        except ImportError:
            pass
        complete = self.doc_dir / 'indexed.json'
        if complete.exists():
            self.items = json.loads(complete.read_text(encoding='utf-8'))
            return
        self.items = (resolve_mineru_items(self.content_list) if self.content_list
                      else extract_pdf(self.pdf, self.doc_dir / 'assets'))
        if not self.items:
            raise ValueError('解析结果为空')
        await self.rag.insert_content_list(self.items, file_path=self.filename)
        if await self.light.text_chunks.is_empty():
            raise RuntimeError('LightRAG 没有写入检索块，请检查入库日志')
        complete.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding='utf-8')

    async def answer(self, question, visual=True):
        async with self.lock:
            opts = dict(mode='naive', top_k=8, chunk_top_k=5, max_total_tokens=3300, enable_rerank=False)
            retrieved = await self.light.aquery_data(question, QueryParam(**opts))
            chunks = retrieved.get('data', {}).get('chunks', [])
            if chunks:
                stored = await self.light.text_chunks.get_by_ids([chunk['chunk_id'] for chunk in chunks])
                for entry, metadata in zip(chunks, stored):
                    if metadata and isinstance(metadata.get('page_idx'), int):
                        entry['page'] = metadata['page_idx'] + 1
                        if isinstance(metadata.get('page_idx_end'), int):
                            entry['page_end'] = metadata['page_idx_end'] + 1
            start = len(self.models.image_events)
            answer = await self.rag.aquery(
                question, mode=opts.pop('mode'), vlm_enhanced=visual,
                user_prompt='用中文简洁回答。只使用给定证据，证据不足时明确说明。', **opts)
            return {'question': question, 'answer': answer,
                    'images_seen_by_model': sum(self.models.image_events[start:]),
                    'retrieval': chunks, 'parser': self.mode}


def model_arguments(parser):
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--data', type=Path, default=ROOT / 'runtime')
    parser.add_argument('--device', default='cuda:0')


async def main(args):
    demo = Demo(args.pdf, args.models, args.device, args.data, args.content_list)
    try:
        await demo.initialize()
        result = await demo.answer(args.question, not args.text_only)
        if args.output:
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        await demo.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    model_arguments(cli)
    cli.add_argument('--pdf', type=Path, required=True)
    cli.add_argument('--content-list', type=Path, help='MinerU 解包后的 *_content_list.json')
    cli.add_argument('--question', required=True)
    cli.add_argument('--text-only', action='store_true', help='关闭原图输入，仍使用同一个已入库的索引')
    cli.add_argument('--output', type=Path)
    asyncio.run(main(cli.parse_args()))
