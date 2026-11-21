# 多模态检索

基于 [RAG-Anything 1.4.4](https://github.com/HKUDS/RAG-Anything/tree/019d6fdcfe635ed66fc096e4cf23376389e9de21) 的 PDF 问答 Demo。上传 PDF 后，系统解析正文、图片和表格，通过 BGE-M3 和 BM25 召回候选块，经 RRF 融合和重排后选取证据页，生成带页码引用的答案。

## 在 Linux 上运行

准备 Python 3.10+ 和适合设备的 PyTorch。将 BGE-M3、Qwen2.5-VL-3B-Instruct、BGE-Reranker-v2-m3 权重分别放在 `models/bge-m3`、`models/qwen-vl` 和 `models/reranker`。

```bash
python3 -m venv .venv
# 在 .venv 中安装适合设备的 PyTorch 和 torchvision。
.venv/bin/pip install -r requirements.txt

python3 -m venv .mineru-venv
# 在 .mineru-venv 中另行安装适合设备的 PyTorch 和 torchvision。
.mineru-venv/bin/pip install 'mineru[pipeline]==3.4.1' 'numpy<2' 'opencv-python==4.11.0.86'

python3 -m venv .ocr-venv
.ocr-venv/bin/pip install 'paddlepaddle==3.2.0' 'paddleocr==3.3.1' 'numpy<2'

.venv/bin/python serve.py --models ./models
```

访问 <http://127.0.0.1:8094> 上传 PDF。用 `--backend baseline` 启动上游检索链路，用 `--review-on-refusal` 启用拒答后的证据补读。

使用兼容 API 作答时，将 `.env.example` 复制为 `.env` 并填入自己的配置，启动时附加 `--answer-env-file .env`。该模式会将候选页图像和证据文字发给配置的服务。
视觉问答首轮默认发送最多 8 张证据页图。可用 `--answer-page-limit 4` 调整页数上限。使用 API 中转时，可加 `--answer-compact-images` 压缩页图。评估接口在 `trace.answer_pages` 返回本次首轮作答发送的 PDF 页码。

## 评测

`evaluate.py` 接收一个 JSON 数组，每题包含 `question` 和 `pages`，页码从 1 开始，空数组表示无答案。先上传 PDF，从接口结果取得文档 ID 再执行。

```bash
.venv/bin/python evaluate.py --doc-id YOUR_DOCUMENT_ID --cases annotated-questions.json
```

`fork/raganything` 包含上游 Python 源码及 Demo 的检索、补读、表格处理等改动。许可证见 [LICENSE](LICENSE)。

## 视觉问答补读实验

[视觉问答补读实验](experiments/README.md)
