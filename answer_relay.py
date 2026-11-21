"""OpenAI-compatible multimodal answer generation through a configured gateway."""
import base64
import io
import json
import os
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import dotenv_values
from PIL import Image


class AnswerAPIError(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def relay_messages(messages, *, compact_images=False):
    """Keep text and page images in order and label each image with its actual format."""
    normalized = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            normalized.append(message)
            continue
        parts = []
        for part in content:
            if part['type'] != 'image_url':
                parts.append(part)
                continue
            scheme, separator, encoded = part['image_url']['url'].partition(',')
            if not separator or not scheme.startswith('data:image/') or not scheme.endswith(';base64'):
                raise AnswerAPIError(422, '视觉证据必须是本地 PDF 图像。')
            try:
                signature = base64.b64decode(encoded[:24], validate=True)
            except ValueError:
                raise AnswerAPIError(422, '视觉证据的图像编码无效。') from None
            if signature.startswith(b'\xff\xd8\xff'):
                mime = 'image/jpeg'
            elif signature.startswith(b'\x89PNG\r\n\x1a\n'):
                mime = 'image/png'
            elif signature.startswith((b'GIF87a', b'GIF89a')):
                mime = 'image/gif'
            elif signature.startswith(b'RIFF') and signature[8:12] == b'WEBP':
                mime = 'image/webp'
            else:
                raise AnswerAPIError(422, '视觉证据的图像格式无法发送给中转模型。')
            if compact_images:
                try:
                    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                        rgb = image.convert('RGB')
                        rgb.thumbnail((1100, 1100))
                        buffer = io.BytesIO()
                        rgb.save(buffer, format='JPEG', quality=67, optimize=True)
                except (OSError, ValueError):
                    raise AnswerAPIError(422, '视觉证据的图像无法压缩。') from None
                mime = 'image/jpeg'
                encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            parts.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}})
        normalized.append(dict(message, content=parts))
    return normalized


def claude_body(messages):
    """Convert normalized chat messages to the relay's native Claude format."""
    system = []
    conversation = []
    for message in messages:
        content = message['content']
        if message['role'] == 'system':
            system.append(content if isinstance(content, str)
                          else '\n'.join(part['text'] for part in content if part['type'] == 'text'))
            continue
        if isinstance(content, list):
            blocks = []
            for part in content:
                if part['type'] == 'image_url':
                    info, encoded = part['image_url']['url'].split(',', 1)
                    mime = info.removeprefix('data:').removesuffix(';base64')
                    blocks.append({'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': encoded}})
                else:
                    blocks.append(part)
            content = blocks
        conversation.append(dict(message, content=content))
    body = {'messages': conversation}
    if system:
        body['system'] = '\n'.join(system)
    return body


class AnswerRelay:
    def __init__(self, base_url, model, env_file=None, *, compact_images=False, timeout=100, native_claude=False):
        url = urlsplit(base_url.rstrip('/'))
        if (url.scheme != 'https' or not url.hostname or url.username is not None or url.password is not None
                or url.query or url.fragment):
            raise ValueError('中转 API 地址须为不含凭据的 HTTPS URL。')
        self.base_url = url.geturl()
        if not url.path.endswith('/v1'):
            self.base_url += '/v1'
        self.model = model
        self.env_file = env_file
        self.compact_images = compact_images
        self.timeout = timeout
        self.native_claude = native_claude

    @staticmethod
    def valid_key(key):
        return 0 < len(key) <= 1024 and all(33 <= ord(character) <= 126 for character in key)

    def key(self):
        try:
            settings = dotenv_values(self.env_file, interpolate=False) if self.env_file else {}
            key = (settings.get('RAG_ANSWER_API_KEY', os.environ.get('RAG_ANSWER_API_KEY')) or '').strip()
        except (OSError, UnicodeError):
            key = ''
        if not self.valid_key(key):
            raise AnswerAPIError(503, '作答服务尚未配置。')
        return key

    def complete(self, messages, limit):
        key = self.key()
        messages = relay_messages(messages, compact_images=self.compact_images)
        body = {'model': self.model, 'stream': False, 'max_tokens': limit + 1536}
        headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
        endpoint = '/chat/completions'
        if self.native_claude:
            endpoint = '/messages'
            headers['anthropic-version'] = '2023-06-01'
            body.update(claude_body(messages))
            # Leave extra room for adaptive thinking on requests retried after truncation.
            body['max_tokens'] = limit + 4096
        else:
            body['messages'] = messages
        request = Request(self.base_url + endpoint, json.dumps(body).encode('utf-8'), headers=headers, method='POST')
        try:
            with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            exc.close()
            if exc.code in {401, 403}:
                raise AnswerAPIError(502, '中转作答服务认证失败。') from None
            if exc.code == 429:
                raise AnswerAPIError(429, '中转已限流或额度不足。') from None
            if exc.code == 404:
                raise AnswerAPIError(502, f'中转未找到接口或模型 {self.model}。') from None
            raise AnswerAPIError(502, f'中转请求失败（HTTP {exc.code}）。') from None
        except (OSError, ValueError) as exc:
            if isinstance(exc, TimeoutError) or isinstance(getattr(exc, 'reason', None), TimeoutError):
                raise AnswerAPIError(504, '中转模型请求超时。') from None
            raise AnswerAPIError(502, '中转连接失败或返回了无效数据。') from None
        try:
            if self.native_claude:
                answer = '\n'.join(block['text'] for block in payload['content'] if block['type'] == 'text')
                truncated = payload.get('stop_reason') == 'max_tokens'
            else:
                choice = payload['choices'][0]
                answer = choice['message']['content']
                truncated = choice.get('finish_reason') == 'length'
        except (KeyError, IndexError, TypeError):
            raise AnswerAPIError(502, '中转未返回有效的作答结果。') from None
        if truncated:
            raise AnswerAPIError(502, '中转模型的输出被截断。')
        if not isinstance(answer, str) or not answer.strip():
            raise AnswerAPIError(502, '中转模型未返回答案文本。')
        return answer.strip()
