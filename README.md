# 多模态 PDF 检索与问答

上传 PDF 后，可查询正文、图表和页面布局信息。系统结合文本检索与原生页图检索，Agent 按需翻页和放大图表，返回带 PDF 页码引用的答案。

## 定版结果

评测对应改提示词前的三工具 Agent（[提交 `5ad549e`](https://github.com/11Dz-dz/multimodal-pdf-rag/commit/5ad549ee95bb8eaa609df96e55cb578a765b0fec)）。作答使用百炼 `qwen3.8-flash` 开思考。以下结果基于 MMLongBench-Doc。

| 场景 | 指标 | 结果 |
| --- | --- | ---: |
| 已知目标 PDF | 页级 Hit@5 / Recall@5 / MRR@5 | 90.62% / 82.02% / 0.7709 |
| 已知目标 PDF | Top-8 全证据页覆盖率 | **81.71%（688/842）** |
| 已知目标 PDF | 端到端问答正确率 | **73.30%（785/1,071）** |
| 135 份 PDF 全库检索 | 文档 Hit@1 / Hit@5 | 74.62% / 83.31% |

73.30% 来自对原始答案的统一重判。完整指标、延迟和复现条件见 [实验与评测](experiments/README.md) 和 [定版指标 JSON](experiments/results/main-5ad549e-metrics.json)。

## 上传和问答流程

1. **入库**。MinerU 解析 PDF，PaddleOCR 补充图表文字。正文、表格和图片保留页码与上下文关系，文本和页图分别建立索引。
2. **检索**。BGE-M3 与 BM25 召回候选块，经 BGE-Reranker 重排并聚合到页。ColQwen2.5 编码整页图像并用 MaxSim 打分。文本页与视觉页使用 Min-Max 归一化融合。
3. **作答**。Agent 先读前 3 个候选页及后续候选摘要，可以继续搜索页、打开整页或裁剪局部。最多 4 轮，最后一轮输出答案和已读证据页码。

Web Demo 在用户选中的 PDF 内问答。全库文档路由由 [单独的检索脚本](experiments/retrieve_mmlongbench_corpus.py) 运行。

## 在 Linux 上运行

准备 Python 3.10+，并按机器的设备和 CUDA 配置，在 `.venv`、`.mineru-venv` 和 `.visual-venv` 中分别安装匹配的 PyTorch 和 torchvision。模型目录使用以下结构。

| 本地目录 | 模型权重 |
| --- | --- |
| `models/bge-m3` | `BAAI/bge-m3` |
| `models/reranker` | `BAAI/bge-reranker-v2-m3` |
| `models/qwen-vl` | `Qwen/Qwen2.5-VL-3B-Instruct` |

```bash
python3 -m venv .venv
python3 -m venv .mineru-venv
python3 -m venv .ocr-venv
python3 -m venv .visual-venv

# 先在对应环境安装 PyTorch 和 torchvision，再安装下列依赖。
.venv/bin/python -m pip install -r requirements.txt
.mineru-venv/bin/python -m pip install 'mineru[pipeline]==3.4.1' 'numpy<2' 'opencv-python==4.11.0.86'
.ocr-venv/bin/python -m pip install 'paddlepaddle==3.2.0' 'paddleocr==3.3.1' 'numpy<2'
.visual-venv/bin/python -m pip install -r requirements-visual.txt
```

将 `.env.example` 复制为 `.env`，在文件中配置支持图片和工具调用的 HTTPS 模型网关。网关会接收问题和用于作答的页图、文字。

```bash
cp .env.example .env
# 填好 .env 后启动 Agent
.venv/bin/python serve.py --models ./models \
  --qa-mode agent --answer-env-file .env --answer-compact-images \
  --visual-python .visual-venv/bin/python --visual-device cuda:0 \
  --visual-base-model ./models/qwen-vl
```

访问 <http://127.0.0.1:8094>，上传 PDF 后提问。普通上传支持不超过 20 MB、40 页的 PDF。ColQwen2.5 默认使用固定版本的 `vidore/colqwen2.5-v0.2`，已有本地快照可通过 `--visual-model` 指定。

要先验证上传、解析和页码引用，也可运行 `.venv/bin/python serve.py --models ./models`，使用本地 Qwen2.5-VL 静态作答。静态模式可通过 `--answer-page-limit` 调整证据页上限，通过 `--review-on-refusal` 开启拒答补读。

## 仓库入口

| 文件 | 用途 |
| --- | --- |
| [serve.py](serve.py) · [index.html](index.html) | 上传、文档索引、问答 API 和 Web 界面 |
| [resume_retrieval.py](fork/raganything/resume_retrieval.py) · [resume_tables.py](fork/raganything/resume_tables.py) | 正文、图表索引，候选块召回与证据页构建 |
| [visual_index.py](fork/raganything/visual_index.py) · [rank_fusion.py](fork/raganything/rank_fusion.py) | 页图编码、视觉检索与排序融合 |
| [agent_qa.py](fork/raganything/agent_qa.py) · [answer_relay.py](answer_relay.py) | 工具调用、证据读取与模型网关 |
| [evaluate.py](evaluate.py) · [experiments/](experiments/) | 单份 PDF 的接口评测、数据集评测和历史实验 |

项目基于 [RAG-Anything 1.4.4](https://github.com/HKUDS/RAG-Anything/tree/019d6fdcfe635ed66fc096e4cf23376389e9de21)。上游源码与本文档问答的扩展位于 [fork/raganything/](fork/raganything/)。许可证见 [LICENSE](LICENSE)。
