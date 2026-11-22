"""Visual PDF page retrieval with a separate ColQwen2.5 model process."""

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import pymupdf
import torch
from PIL import Image
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

if __package__:
    from .rank_fusion import fuse_pages


MODEL = 'vidore/colqwen2.5-v0.2'
MODEL_REVISION = 'dcbe8d9cede518bce830488364ba0e40c873645b'
INDEX_FILENAME = 'colqwen25-pages.pt'
RENDER_EDGE = 1024
INDEX_SCHEMA = 1


class VisualIndexError(RuntimeError):
    pass


class VisualWorker:
    """One persistent encoder shared across document activations."""

    def __init__(self, python, model=MODEL, *, device='cuda:0', batch_size=8, base_model=None):
        self.command = [str(python), '-u', str(Path(__file__).resolve()), '--worker', '--model', str(model),
                        '--device', device, '--batch-size', str(batch_size)]
        if base_model:
            self.command.extend(('--base-model', str(base_model)))
        self.identity = model_id(str(model), str(base_model) if base_model else None)
        self.process = None
        self.lock = asyncio.Lock()

    async def request(self, operation, **payload):
        async with self.lock:
            if self.process is None or self.process.returncode is not None:
                self.process = await asyncio.create_subprocess_exec(
                    *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    limit=4 * 1024 * 1024,
                )
            try:
                message = json.dumps({'op': operation, **payload}, ensure_ascii=False)
                self.process.stdin.write((message + '\n').encode())
                await self.process.stdin.drain()
                raw = await asyncio.wait_for(self.process.stdout.readline(), timeout=180)
                if not raw:
                    raise VisualIndexError('ColQwen worker exited before returning results')
                try:
                    response = json.loads(raw)
                except ValueError as exc:
                    raise VisualIndexError('ColQwen worker returned an invalid response') from exc
            except asyncio.TimeoutError as exc:
                await self._close()
                raise VisualIndexError('ColQwen worker timed out') from exc
            except BaseException:
                await self._close()
                raise
            if 'error' in response:
                raise VisualIndexError(response['error'])
            return response['result']

    async def _close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    async def close(self):
        async with self.lock:
            await self._close()


class VisualPageIndex:
    def __init__(self, worker, pdf, directory, doc_id):
        self.worker = worker
        self.document = {'doc_id': doc_id, 'pdf': str(Path(pdf).resolve()),
                         'index': str((Path(directory) / INDEX_FILENAME).absolute())}

    async def search(self, question, top_k=24):
        return await self.worker.request('search', document=self.document, question=question, top_k=top_k)

    async def build(self):
        return await self.worker.request('build', document=self.document)


def catalog_documents(site, catalog):
    for row in catalog:
        doc_id = row['id']
        directory = Path(site) / 'uploads' / doc_id.replace('-', '')
        yield {'doc_id': doc_id, 'pdf': str((directory / 'source.pdf').resolve()),
               'index': str((directory / 'resume-index-v1' / INDEX_FILENAME).absolute())}


def render_batch(pdf, start, stop):
    images = []
    for number in range(start, stop):
        page = pdf[number]
        scale = RENDER_EDGE / max(page.rect.width, page.rect.height)
        pixels = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
        images.append(Image.frombytes('RGB', (pixels.width, pixels.height), pixels.samples))
    return images


def calculate_maxsim(query, pages, device, batch_size=24):
    """Score variable-length page vectors, masking padding before the maximum."""
    if not pages:
        return []
    mode = torch.float16 if torch.device(device).type == 'cuda' else torch.float32
    query = query.to(device=device, dtype=mode).T.contiguous()
    results = []
    with torch.inference_mode():
        for offset in range(0, len(pages), batch_size):
            group = pages[offset:offset + batch_size]
            tokens = pad_sequence(group, batch_first=True).to(device=device, dtype=mode)
            lengths = torch.tensor([len(page) for page in group], device=device)
            cosine = tokens @ query
            padding = torch.arange(tokens.shape[1], device=device)[None, :] >= lengths[:, None]
            cosine.masked_fill_(padding[..., None], -torch.inf)
            results.extend(cosine.amax(dim=1).sum(dim=1, dtype=torch.float32).tolist())
    return results


def model_id(model, base_model):
    identity = [model, base_model, RENDER_EDGE, INDEX_SCHEMA]
    if model == MODEL:
        identity.append(MODEL_REVISION)
    for value in (model, base_model):
        if value and Path(value).is_dir():
            directory = Path(value).resolve()
            paths = {*directory.rglob('*.safetensors'), *directory.rglob('*config.json')}
            for file in sorted(paths):
                stat = file.stat()
                identity.append((str(file.relative_to(directory)), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def load_checkpoint(path, mtime_ns):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


load_checkpoint = lru_cache(maxsize=32)(load_checkpoint)


class VisualEngine:
    def __init__(self, model, device, batch_size, base_model=None):
        from sentence_transformers import MultiVectorEncoder

        self.signature = model_id(model, base_model)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.corpus = None
        with contextlib.ExitStack() as stack:
            if base_model:
                source = Path(model).resolve()
                if not source.is_dir():
                    raise ValueError('--base-model requires a local adapter snapshot')
                overlay = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='colqwen25-')))
                for item in source.iterdir():
                    if item.name != 'adapter_config.json':
                        (overlay / item.name).symlink_to(item, target_is_directory=item.is_dir())
                adapter = json.loads((source / 'adapter_config.json').read_text(encoding='utf-8'))
                adapter['base_model_name_or_path'] = str(Path(base_model).resolve())
                (overlay / 'adapter_config.json').write_text(json.dumps(adapter), encoding='utf-8')
                model = str(overlay)
            self.encoder = MultiVectorEncoder(
                model, device=device, local_files_only=Path(model).is_dir(),
                revision=MODEL_REVISION if model == MODEL else None,
                model_kwargs={'dtype': torch.bfloat16 if self.device.type == 'cuda' else torch.float32},
            )

    def fingerprint(self, pdf):
        stat = Path(pdf).stat()
        return [self.signature, stat.st_size, stat.st_mtime_ns]

    def checkpoint(self, document, *, build=False):
        pdf, path = document['pdf'], Path(document['index'])
        expected = self.fingerprint(pdf)
        try:
            existing = load_checkpoint(path, path.stat().st_mtime_ns)
            if existing.get('fingerprint') == expected:
                return existing
        except FileNotFoundError:
            pass
        if not build:
            raise VisualIndexError(f'Missing or stale visual index for {document["doc_id"]}')
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self.checkpoint(document)
            except VisualIndexError:
                pass
            embeddings = []
            with pymupdf.open(pdf) as source:
                for start in range(0, source.page_count, self.batch_size):
                    images = render_batch(source, start, min(start + self.batch_size, source.page_count))
                    encoded = self.encoder.encode_document(images, batch_size=self.batch_size, show_progress_bar=False)
                    for page in encoded:
                        tokens = torch.as_tensor(page).to(dtype=torch.float16, device='cpu').contiguous()
                        if tokens.ndim != 2 or tokens.shape[1] != 128 or not len(tokens):
                            raise VisualIndexError(f'Invalid ColQwen page vectors {tuple(tokens.shape)}')
                        embeddings.append(tokens)
            if not embeddings or self.fingerprint(pdf) != expected:
                raise VisualIndexError('PDF has no pages or changed during visual indexing')
            means = F.normalize(torch.stack([page.float().mean(dim=0) for page in embeddings]), dim=1).half()
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem, suffix='.tmp', delete=False) as target:
                temporary = Path(target.name)
            try:
                torch.save({'fingerprint': expected, 'page_embeddings': embeddings, 'mean_embeddings': means}, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return self.checkpoint(document)

    def configure_corpus(self, documents):
        identities, vectors, means = [], [], []
        for document in documents:
            bundle = self.checkpoint(document)
            entries = bundle['page_embeddings']
            identities.extend((document['doc_id'], index + 1) for index in range(len(entries)))
            vectors.extend(entries)
            means.append(bundle['mean_embeddings'])
        if not vectors:
            raise VisualIndexError('No visual pages in the requested corpus')
        self.corpus = identities, vectors, torch.cat(means).to(device=self.device, dtype=torch.float32)
        return len(identities)

    def rank(self, question_vectors, identities, vectors, top_k):
        scores = calculate_maxsim(question_vectors, vectors, self.device)
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:top_k]
        return [{'doc_id': identities[i][0], 'page': identities[i][1], 'score': scores[i], 'blocks': []}
                for i in order]

    def search(self, requests, top_k=24, corpus=False, coarse_k=200, include_candidates=False, corpus_top_k=None):
        if corpus and self.corpus is None:
            raise VisualIndexError('Configure a visual corpus before searching it')
        queries = self.encoder.encode_query([row['question'] for row in requests], batch_size=16,
                                            show_progress_bar=False)
        output = []
        for row, query in zip(requests, queries):
            query = torch.as_tensor(query)
            result = {}
            if document := row.get('document'):
                vectors = self.checkpoint(document, build=row.get('build', False))['page_embeddings']
                identities = [(document['doc_id'], number + 1) for number in range(len(vectors))]
                result['document'] = self.rank(query, identities, vectors, top_k)
            if corpus:
                identities, vectors, means = self.corpus
                mean = F.normalize(query.float().mean(dim=0).to(self.device), dim=0)
                pool = (means @ mean).topk(min(coarse_k, len(vectors))).indices.tolist()
                result['corpus'] = self.rank(query, [identities[i] for i in pool], [vectors[i] for i in pool],
                                             top_k if corpus_top_k is None else corpus_top_k)
                if include_candidates:
                    result['corpus_candidates'] = [identities[i] for i in pool]
            output.append(result)
        return output


def serve_worker(engine):
    # Reserve stdout for the JSON-line protocol, even when a model prints diagnostics.
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request.pop('op')
                if op == 'search':
                    results = engine.search([{'question': request['question'], 'document': request['document'],
                                              'build': True}], top_k=request['top_k'])
                    result = results[0]['document']
                elif op == 'build':
                    result = len(engine.checkpoint(request['document'], build=True)['page_embeddings'])
                elif op == 'search_many':
                    result = engine.search(**request)
                elif op == 'configure_corpus':
                    result = engine.configure_corpus(request['documents'])
                else:
                    raise ValueError(f'Unknown visual worker operation {op}')
                response = {'result': result}
            except Exception as exc:
                response = {'error': f'{type(exc).__name__}: {exc}'}
            print(json.dumps(response, ensure_ascii=False, allow_nan=False), file=output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--worker', action='store_true')
    mode.add_argument('--build-catalog', type=Path)
    parser.add_argument('--site', type=Path)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--base-model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.shard < args.shards or (args.build_catalog and not args.site):
        parser.error('Provide a positive batch size, a valid shard and --site when building a catalog')
    with contextlib.redirect_stdout(sys.stderr):
        engine = VisualEngine(args.model, args.device, args.batch_size, args.base_model)
    if args.worker:
        serve_worker(engine)
    else:
        catalog = json.loads(args.build_catalog.read_text(encoding='utf-8'))
        documents = list(catalog_documents(args.site, catalog))[args.shard::args.shards]
        total_pages = 0
        for position, document in enumerate(documents, 1):
            total_pages += len(engine.checkpoint(document, build=True)['page_embeddings'])
            print(json.dumps({'documents': position, 'total': len(documents), 'pages': total_pages}), flush=True)
