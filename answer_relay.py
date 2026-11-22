"""Multimodal answer generation and client tools through a configured gateway."""
import base64
import io
import json
import os
from dataclasses import dataclass
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
        content = message.get('content')
        if not isinstance(content, list) or message.get('native_assistant'):
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
            if compact_images and not part.get('lossless'):
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
        if message.get('native_assistant'):
            conversation.append({'role': 'assistant', 'content': content})
            continue
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
        if message['role'] == 'tool':
            result = {'type': 'tool_result', 'tool_use_id': message['tool_call_id'], 'content': content}
            if message.get('is_error'):
                result['is_error'] = True
            if conversation[-1]['role'] == 'user':
                conversation[-1]['content'].append(result)
            else:
                conversation.append({'role': 'user', 'content': [result]})
        elif (message['role'] == 'user' and conversation and conversation[-1]['role'] == 'user'
              and isinstance(conversation[-1]['content'], list)
              and conversation[-1]['content']
              and conversation[-1]['content'][0]['type'] == 'tool_result'):
            conversation[-1]['content'].extend([{'type': 'text', 'text': content}]
                                               if isinstance(content, str) else content)
        else:
            conversation.append(dict(message, content=content))
    body = {'messages': conversation}
    if system:
        body['system'] = '\n'.join(system)
    return body


def openai_messages(messages):
    """Chat Completions accepts only text in the tool role."""
    conversation, images = [], []
    for message in messages:
        if message['role'] == 'tool':
            content = message['content']
            if isinstance(content, list):
                images.extend(part for part in content if part['type'] == 'image_url')
                content = '\n'.join(part['text'] for part in content if part['type'] == 'text')
            conversation.append({'role': 'tool', 'tool_call_id': message['tool_call_id'], 'content': content})
            continue
        if images:
            conversation.append({'role': 'user', 'content': images})
            images = []
        conversation.append(message)
    if images:
        conversation.append({'role': 'user', 'content': images})
    return conversation


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: object


@dataclass(frozen=True)
class ToolTurn:
    text: str
    calls: list
    message: dict
    has_reasoning: bool = False


def streamed_completion(response):
    """Assemble OpenAI-compatible text, thinking and fragmented tool calls."""
    text, reasoning, tools = [], [], {}
    finish = None
    for line in response:
        if not line.startswith(b'data:'):
            continue
        value = line[5:].strip()
        if value == b'[DONE]':
            break
        event = json.loads(value)
        if event.get('error'):
            raise AnswerAPIError(502, '百炼流式作答中断。')
        for choice in event.get('choices') or []:
            delta = choice.get('delta') or {}
            text.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or '')
            for fragment in delta.get('tool_calls') or []:
                tool = tools.setdefault(fragment['index'], {
                    'type': 'function', 'function': {'name': '', 'arguments': ''},
                })
                if fragment.get('id'):
                    tool['id'] = fragment['id']
                function = fragment.get('function') or {}
                for name in ('name', 'arguments'):
                    tool['function'][name] += function.get(name) or ''
            finish = choice.get('finish_reason') or finish
    if finish is None:
        raise AnswerAPIError(502, '百炼流式作答缺少结束标记。')
    message = {'content': ''.join(text), 'reasoning_content': ''.join(reasoning),
               'tool_calls': [tools[index] for index in sorted(tools)]}
    return {'choices': [{'message': message, 'finish_reason': finish}]}


def tool_turn(payload, native_claude):
    has_reasoning = False
    try:
        if native_claude:
            blocks = payload['content']
            truncated = payload.get('stop_reason') == 'max_tokens'
            calls = [ToolCall(b['id'], b['name'], b['input']) for b in blocks if b['type'] == 'tool_use']
            text = '\n'.join(b['text'] for b in blocks if b['type'] == 'text')
            message = {'role': 'assistant', 'content': blocks, 'native_assistant': True}
        else:
            choice = payload['choices'][0]
            raw = choice['message']
            has_reasoning = bool(raw.get('reasoning_content'))
            truncated = choice.get('finish_reason') == 'length'
            entries = raw.get('tool_calls') or []
            calls = []
            for entry in entries:
                function = entry['function']
                try:
                    arguments = json.loads(function['arguments'])
                except (TypeError, ValueError):
                    arguments = None
                calls.append(ToolCall(entry['id'], function['name'], arguments))
            text = raw.get('content') or ''
            message = {'role': 'assistant', 'content': text or None}
            if entries:
                message['tool_calls'] = entries
                if raw.get('reasoning_content'):
                    message['reasoning_content'] = raw['reasoning_content']
        if not isinstance(text, str) or any(not isinstance(c.id, str) or not c.id or not isinstance(c.name, str)
                                            for c in calls) or len({c.id for c in calls}) != len(calls):
            raise ValueError('Invalid tool response')
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        raise AnswerAPIError(502, '中转未返回有效的作答结果。') from None
    if truncated:
        raise AnswerAPIError(502, '中转模型的输出被截断。')
    if not text.strip() and not calls:
        raise AnswerAPIError(502, '中转模型未返回答案文本。')
    return ToolTurn(text.strip(), calls, message, has_reasoning)


class AnswerRelay:
    def __init__(self, base_url, model, env_file=None, *, compact_images=False, timeout=100, native_claude=False,
                 bailian_thinking=None):
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
        if native_claude and bailian_thinking is not None:
            raise ValueError('百炼思考模式须使用 OpenAI 兼容接口。')
        self.bailian_thinking = bailian_thinking

    @staticmethod
    def valid_key(key):
        return 0 < len(key) <= 1024 and all(33 <= ord(character) <= 126 for character in key)

    def key(self):
        try:
            settings = dotenv_values(self.env_file, interpolate=False) if self.env_file else {}
            variable = 'DASHSCOPE_API_KEY' if self.bailian_thinking is not None else 'RAG_ANSWER_API_KEY'
            key = (settings.get(variable, os.environ.get(variable)) or '').strip()
        except (OSError, UnicodeError):
            key = ''
        if not self.valid_key(key):
            raise AnswerAPIError(503, '作答服务尚未配置。')
        return key

    def complete(self, messages, limit):
        turn = self.exchange(messages, limit)
        if not turn.text:
            raise AnswerAPIError(502, '中转模型未返回答案文本。')
        return turn.text

    def complete_tools(self, messages, limit, tools, *, allow_tools=True):
        return self.exchange(messages, limit, tools=tools, allow_tools=allow_tools)

    def exchange(self, messages, limit, *, tools=(), allow_tools=True):
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
            if tools:
                body['tools'] = [{'name': item['name'], 'description': item['description'],
                                  'input_schema': item['parameters']} for item in tools]
                body['tool_choice'] = {'type': 'auto' if allow_tools else 'none'}
        else:
            body['messages'] = openai_messages(messages)
            if self.bailian_thinking is not None:
                body.update(stream=True, enable_thinking=self.bailian_thinking)
                headers['Accept'] = 'text/event-stream'
            if tools:
                body['tools'] = [{'type': 'function', 'function': item} for item in tools]
                body['tool_choice'] = 'auto' if allow_tools else 'none'
        request = Request(self.base_url + endpoint, json.dumps(body).encode('utf-8'), headers=headers, method='POST')
        try:
            with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                payload = streamed_completion(response) if body['stream'] else json.load(response)
        except HTTPError as exc:
            thinking_not_supported = False
            billing_blocked = False
            if self.bailian_thinking is not None and exc.code in {400, 402}:
                try:
                    error = json.load(exc)
                    details = error.get('error', error)
                    billing_blocked = str(details.get('code', '')).casefold() == 'arrearage'
                    reason = str(details.get('message', '')).casefold()
                    thinking_not_supported = self.bailian_thinking is False and 'enable_thinking' in reason and (
                        'only support true' in reason or 'must be true' in reason
                        or ('false' in reason and ('not support' in reason or 'cannot' in reason))
                    )
                except (AttributeError, ValueError):
                    pass
            exc.close()
            if billing_blocked or (self.bailian_thinking is not None and exc.code == 402):
                raise AnswerAPIError(402, '百炼计费状态阻止调用（Arrearage）。') from None
            if thinking_not_supported:
                raise AnswerAPIError(400, '百炼当前端点拒绝为该模型关闭思考。') from None
            if exc.code in {401, 403}:
                raise AnswerAPIError(502, '中转作答服务认证失败。') from None
            if exc.code == 429:
                raise AnswerAPIError(429, '中转已限流或额度不足。') from None
            if exc.code == 404:
                raise AnswerAPIError(502, f'中转未找到接口或模型 {self.model}。') from None
            raise AnswerAPIError(502, f'中转请求失败（HTTP {exc.code}）。') from None
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            if isinstance(exc, TimeoutError) or isinstance(getattr(exc, 'reason', None), TimeoutError):
                raise AnswerAPIError(504, '中转模型请求超时。') from None
            raise AnswerAPIError(502, '中转连接失败或返回了无效数据。') from None
        return tool_turn(payload, self.native_claude)
