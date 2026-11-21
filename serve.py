"""Upload a PDF and query its RAG-Anything index."""
import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pymupdf
import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from answer_relay import AnswerAPIError, AnswerRelay
from demo import Demo, LocalModels, ROOT, model_arguments, resolve_mineru_items
from raganything.resume_retrieval import ResumeRetriever


MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 40
DEFAULT_ANSWER_PAGE_LIMIT = 8
LOG = logging.getLogger('uvicorn.error')
BenchmarkRun = Literal['full-run-v1', 'full-run-v2', 'full-run-v3', 'full-run-v4', 'full-run-haiku']


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=400)
    visual: bool = True


class QuestionBatch(BaseModel):
    questions: list[Question] = Field(min_length=1, max_length=8)


class PdfLibrary:
    def __init__(self, args):
        self.args = args
        self.root = args.data.resolve() / 'uploads'
        self.root.mkdir(parents=True, exist_ok=True)
        self.mineru = args.mineru.resolve()
        self.lock = asyncio.Lock()
        self.active_id = None
        self.active = None
        self.models = None
        self.benchmark = getattr(args, 'benchmark', None)
        self.mineru_api = getattr(args, 'mineru_api', None)
        self.answer_relay = getattr(args, 'answer_relay', None)

    def metadata(self, doc_id):
        path = self.root / doc_id.hex / 'document.json'
        if not path.is_file():
            raise HTTPException(404, '没有找到已入库的 PDF')
        return json.loads(path.read_text(encoding='utf-8'))

    def pdf_path(self, doc_id):
        return self.root / doc_id.hex / 'source.pdf'

    def save_metadata(self, doc_id, metadata):
        path = self.root / doc_id.hex / 'document.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
        temporary.replace(path)

    async def close(self):
        if self.active is not None:
            demo, self.active = self.active, None
            self.active_id = None
            await demo.close()

    async def activate(self, doc_id, metadata):
        if self.active_id == doc_id:
            return self.active
        await self.close()
        if self.models is None:
            self.models = await self.load_models()
        directory = self.root / doc_id.hex
        metadata = dict(metadata)
        if self.benchmark and not metadata.get('content_list'):
            metadata['content_list'] = await self.parse(directory)
            self.save_metadata(doc_id, metadata)
        content_list = directory / metadata['content_list']
        if self.args.backend == 'baseline':
            demo = Demo(self.pdf_path(doc_id), self.args.models, self.args.device, directory / 'index',
                        content_list, models=self.models, workspace=doc_id.hex, filename=metadata['filename'])
        else:
            demo = ResumeRetriever(items=resolve_mineru_items(content_list, include_charts=True),
                                   directory=directory / 'resume-index-v1',
                                   doc_id=doc_id.hex, filename=metadata['filename'], models=self.models,
                                   ocr_python=self.args.ocr_python, ocr_script=ROOT / 'paddle_worker.py',
                                   ocr_timeout=7200 if self.benchmark else 300,
                                   ocr_workers=getattr(self.args, 'ocr_workers', 1),
                                   review_on_refusal=getattr(self.args, 'review_on_refusal', False),
                                   answer_page_limit=getattr(self.args, 'answer_page_limit', DEFAULT_ANSWER_PAGE_LIMIT))
        try:
            await demo.initialize()
        except BaseException:
            await demo.close()
            raise
        self.active_id, self.active = doc_id, demo
        return demo

    async def load_models(self):
        return await asyncio.to_thread(LocalModels, self.args.models, self.args.device,
                                       self.args.answer_model, self.args.answer_batch_size, self.answer_relay)

    async def parse(self, directory):
        if not self.mineru.is_file():
            raise HTTPException(503, '服务器暂未配置 MinerU')
        output = directory / 'parsed'
        log_path = directory / 'mineru.log'
        env = os.environ.copy()
        env.setdefault('MINERU_MODEL_SOURCE', 'modelscope')
        env.setdefault('MODELSCOPE_CACHE', str(ROOT / 'model-cache/ms'))
        env.setdefault('HF_HOME', str(ROOT / 'model-cache/hf'))
        api_args = ['--api-url', self.mineru_api] if self.mineru_api else []
        with log_path.open('wb') as log:
            process = await asyncio.create_subprocess_exec(
                str(self.mineru), '-p', str(directory / 'source.pdf'), '-o', str(output),
                '-b', 'pipeline', '-l', 'ch', *api_args, cwd=directory, env=env,
                stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=7200 if self.benchmark else 300)
            except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise HTTPException(504, 'PDF 解析超时。可以尝试拆成更短的 PDF。') from exc
        matches = sorted(output.rglob('source_content_list.json'))
        if process.returncode or len(matches) != 1:
            LOG.error('MinerU failed: exit=%s, %s', process.returncode,
                      log_path.read_text(encoding='utf-8', errors='replace')[-1600:])
            raise HTTPException(422, '没有解析出可用内容，请检查 PDF 后重试。')
        return matches[0].relative_to(directory).as_posix()

    async def ingest(self, upload):
        filename = (upload.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(400, '请选择 PDF 文件')
        doc_id = uuid4()
        directory = self.root / doc_id.hex
        directory.mkdir()
        try:
            path = self.pdf_path(doc_id)
            size = 0
            with path.open('wb') as target:
                while part := await upload.read(1024 * 1024):
                    size += len(part)
                    if size > MAX_BYTES:
                        raise HTTPException(413, '单个 PDF 最大 20 MB')
                    target.write(part)
            with path.open('rb') as source:
                if b'%PDF-' not in source.read(1024):
                    raise HTTPException(400, '文件内容不是有效的 PDF')
            try:
                with pymupdf.open(path, filetype='pdf') as pdf:
                    if pdf.needs_pass or pdf.is_encrypted:
                        raise HTTPException(400, '请先移除 PDF 密码保护')
                    page_count = pdf.page_count
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(400, '无法打开这个 PDF，请检查文件是否损坏') from exc
            if not 1 <= page_count <= MAX_PAGES:
                raise HTTPException(400, 'PDF 须为 1–40 页')
            async with self.lock:
                metadata = {'id': str(doc_id), 'filename': filename, 'pages': page_count}
                metadata['content_list'] = await self.parse(directory)
                demo = await self.activate(doc_id, metadata)
                metadata.update(items=len(demo.items),
                                tables=sum(item['type'] == 'table' for item in demo.items),
                                images=sum(item['type'] == 'image' for item in demo.items))
                self.save_metadata(doc_id, metadata)
            return metadata
        except BaseException:
            if self.active_id == doc_id:
                await self.close()
            shutil.rmtree(directory, ignore_errors=True)
            raise


def make_app(args):
    library = PdfLibrary(args)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await library.close()

    app = FastAPI(title='多模态检索', lifespan=lifespan)

    @app.exception_handler(AnswerAPIError)
    async def answer_api_error(_request, exc):
        return JSONResponse(status_code=exc.status, content={'detail': str(exc)})

    @app.get('/', include_in_schema=False)
    def home():
        return FileResponse(ROOT / 'index.html', headers={'Cache-Control': 'no-store'})

    @app.get('/api/status')
    def status():
        return {'ready': library.mineru.is_file(), 'parser': 'MinerU', 'accepts_uploads': True,
                'backend': args.backend, 'ocr_ready': args.ocr_python.is_file(),
                'benchmark': library.benchmark is not None}

    if library.benchmark:
        index_slots = asyncio.Semaphore(args.index_workers)
        index_locks = {}

        @app.get('/api/benchmark/documents')
        def benchmark_documents():
            catalog = json.loads((library.benchmark / 'documents.json').read_text(encoding='utf-8'))
            for row in catalog:
                directory = library.root / UUID(row['id']).hex
                row['indexed'] = (directory / 'resume-index-v1' / 'charts-v1').is_file()
            return catalog

        @app.get('/api/benchmark/test-set')
        def benchmark_test_set():
            return FileResponse(library.benchmark / 'test-set.jsonl', media_type='application/x-ndjson',
                                filename='mmlongbench-doc-v2.jsonl')

        @app.get('/api/benchmark/progress')
        def benchmark_progress(run: BenchmarkRun = 'full-run-v1'):
            path = library.benchmark / run / 'status.json'
            if not path.is_file():
                return {'phase': 'pending'}
            return json.loads(path.read_text(encoding='utf-8'))

        @app.get('/api/benchmark/report')
        def benchmark_report(run: BenchmarkRun = 'full-run-v1'):
            path = library.benchmark / run / 'report.json'
            if not path.is_file():
                raise HTTPException(404, '全量评测报告尚未生成')
            return FileResponse(path, media_type='application/json', filename='mmlongbench-doc-v2-report.json')

        @app.post('/api/benchmark/documents/{doc_id}/index')
        async def benchmark_index(doc_id: UUID):
            lock = index_locks.setdefault(doc_id, asyncio.Lock())
            async with index_slots, lock:
                async with library.lock:
                    if library.models is None:
                        library.models = await library.load_models()
                    if library.active_id == doc_id:
                        await library.close()
                worker = PdfLibrary(args)
                worker.models = library.models
                try:
                    demo = await worker.activate(doc_id, worker.metadata(doc_id))
                    count = len(demo.blocks)
                    if not count or demo.vector_store.count('chunks', exact=True).count != count:
                        raise HTTPException(500, '索引条数与文档块数量不一致')
                    return {'id': str(doc_id), 'chunks': count}
                finally:
                    await worker.close()

        @app.post('/api/benchmark/documents/{doc_id}/evaluate')
        async def benchmark_evaluate(doc_id: UUID, query: Question):
            lock = index_locks.setdefault(doc_id, asyncio.Lock())
            async with lock, library.lock:
                demo = await library.activate(doc_id, library.metadata(doc_id))
                return await demo.answer(query.question.strip(), visual=query.visual, use_cache=False)

        @app.post('/api/benchmark/documents/{doc_id}/evaluate-batch')
        async def benchmark_evaluate_batch(doc_id: UUID, batch: QuestionBatch):
            lock = index_locks.setdefault(doc_id, asyncio.Lock())
            async with lock, library.lock:
                demo = await library.activate(doc_id, library.metadata(doc_id))
                questions = [query.question.strip() for query in batch.questions]
                vectors = await demo.models.embed(questions)
                results = await asyncio.gather(*(demo.answer(question, visual=query.visual, use_cache=False,
                                                              vector=vector)
                                                 for question, query, vector in zip(questions, batch.questions, vectors)),
                                               return_exceptions=True)
                return [{'error': str(result)} if isinstance(result, Exception) else {'result': result}
                        for result in results]

    @app.post('/api/documents', status_code=201)
    async def upload(file: UploadFile = File(...)):
        try:
            return await library.ingest(file)
        finally:
            await file.close()

    @app.get('/api/documents/{doc_id}')
    def document_info(doc_id: UUID):
        return library.metadata(doc_id)

    @app.get('/api/documents/{doc_id}/pdf')
    def document(doc_id: UUID):
        metadata = library.metadata(doc_id)
        return FileResponse(library.pdf_path(doc_id), media_type='application/pdf',
                            filename=metadata['filename'], content_disposition_type='inline')

    @app.get('/api/documents/{doc_id}/pages/{number}')
    def page(doc_id: UUID, number: int):
        metadata = library.metadata(doc_id)
        if not 1 <= number <= metadata['pages']:
            raise HTTPException(404, '没有这一页')
        with pymupdf.open(library.pdf_path(doc_id)) as doc:
            pdf_page = doc[number - 1]
            scale = min(1.6, 1600 / max(pdf_page.rect.width, pdf_page.rect.height, 1))
            image = pdf_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            return Response(image.tobytes('jpeg'), media_type='image/jpeg')

    @app.post('/api/documents/{doc_id}/ask')
    async def ask(doc_id: UUID, query: Question):
        question = query.question.strip()
        if not question:
            raise HTTPException(400, '请输入问题')
        lock = index_locks.setdefault(doc_id, asyncio.Lock()) if library.benchmark else nullcontext()
        async with lock, library.lock:
            metadata = library.metadata(doc_id)
            demo = await library.activate(doc_id, metadata)
            return await demo.answer(question, visual=query.visual)

    return app


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    model_arguments(cli)
    cli.add_argument('--mineru', type=Path, default=ROOT / '.mineru-venv/bin/mineru')
    cli.add_argument('--ocr-python', type=Path, default=ROOT / '.ocr-venv/bin/python')
    cli.add_argument('--backend', choices=('resume', 'baseline'), default='resume')
    cli.add_argument('--port', type=int, default=8094)
    cli.add_argument('--benchmark', type=Path)
    cli.add_argument('--answer-model', type=Path, help='Local Qwen vision-language checkpoint')
    cli.add_argument('--answer-api-base', help='HTTPS base URL for an OpenAI-compatible model gateway')
    cli.add_argument('--answer-api-model', help='Model ID recognized by the gateway')
    cli.add_argument('--answer-env-file', type=Path, help='Dotenv file containing the gateway configuration')
    cli.add_argument('--answer-compact-images', action='store_true', help='Compress page images for the relay')
    cli.add_argument('--answer-api-timeout', type=int, choices=range(10, 181), default=100)
    cli.add_argument('--answer-native-claude', action='store_true', help='Use the gateway Claude Messages endpoint')
    cli.add_argument('--index-workers', type=int, choices=range(1, 9), default=2)
    cli.add_argument('--answer-batch-size', type=int, choices=range(1, 9), default=1)
    cli.add_argument('--answer-page-limit', type=int, choices=range(1, 13), default=DEFAULT_ANSWER_PAGE_LIMIT,
                     help='Maximum number of PDF pages sent in the first visual answer request')
    cli.add_argument('--review-on-refusal', action='store_true', help='Reread focused PDF passages after a refusal')
    cli.add_argument('--mineru-api')
    cli.add_argument('--ocr-workers', type=int, default=1, choices=range(1, 5))
    args = cli.parse_args()
    relay_settings = dotenv_values(args.answer_env_file, interpolate=False) if args.answer_env_file else {}
    relay_base = args.answer_api_base or relay_settings.get('RAG_ANSWER_API_BASE') or os.getenv('RAG_ANSWER_API_BASE')
    args.answer_relay = None
    if relay_base:
        relay_model = (args.answer_api_model or relay_settings.get('RAG_ANSWER_API_MODEL')
                       or os.getenv('RAG_ANSWER_API_MODEL') or 'claude-haiku-5-5')
        try:
            args.answer_relay = AnswerRelay(relay_base, relay_model, args.answer_env_file,
                                           compact_images=args.answer_compact_images, timeout=args.answer_api_timeout,
                                           native_claude=args.answer_native_claude)
        except ValueError as exc:
            cli.error(str(exc))
    elif args.answer_env_file:
        cli.error('RAG_ANSWER_API_BASE is required in the answer configuration')
    args.mineru = args.mineru.resolve()
    args.ocr_python = args.ocr_python.absolute()
    os.environ['PATH'] = str(args.mineru.parent) + os.pathsep + os.environ.get('PATH', '')
    uvicorn.run(make_app(args), host='127.0.0.1', port=args.port)
