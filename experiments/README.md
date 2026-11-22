# 视觉问答补读实验

这些脚本在已指定目标 PDF 的 MMLongBench-Doc 样本上复放固定的检索排序，并实际调用配置的视觉作答模型。它们使用现有索引和基线日志，适合比较同一批题在不同作答策略下的变化。

- `replay_visual_backfill.py` 对比首轮 8 页、拒答后补读第 9–12 名候选页，以及补读加独立核验。补读输入保留最多两张首轮页图和相关文本，并附上新增页图。`--holdout` 使用与首批样本不重叠的 30 题。`--rerun-ranking` 在已有 Qdrant 索引上调用当前检索实现，默认使用归档排名。请用闲置的索引副本运行重新排序，以免争用 Qdrant 本地目录锁。
- `visual_backfill_verifier.py` 在补读产生候选答案后再次读证据，检查答案一致性、可定位的引文，以及可复算的数值结果。
- `replay_answer_support_trigger.py` 对比 8 页基线、仅在拒答时补读，以及引用证据不足时也补读。`sample-manifest.json` 只保存已用样本的题号和分组，用于复现选样。

## 运行

从仓库根目录运行。准备主项目的 Python 环境和模型权重，按 `.env.example` 配置视觉模型中转。脚本需要原评测留下的 `full-run-haiku/responses.jsonl`、`full-run-haiku/judgments.jsonl`，以及包含 `uploads/<doc_id>/` 的索引目录。每份文档目录需要 `source.pdf`、`document.json`、解析结果和 `resume-index-v1/chunks.json`。使用中转时，模型会收到问题及相关页图和文字。

```bash
mkdir -p runtime-experiments

.venv/bin/python -m experiments.replay_visual_backfill \
  --root /path/to/benchmark --site /path/to/indexed-site \
  --models ./models --env-file .env --concurrency 2 --verify \
  --report runtime-experiments/visual-backfill.json

.venv/bin/python -m experiments.replay_visual_backfill \
  --root /path/to/benchmark --site /path/to/indexed-site \
  --models ./models --env-file .env --concurrency 2 --verify --holdout \
  --report runtime-experiments/visual-backfill-holdout.json

.venv/bin/python -m experiments.replay_answer_support_trigger \
  --root /path/to/benchmark --site /path/to/indexed-site \
  --models ./models --env-file .env --concurrency 2 \
  --prior experiments/sample-manifest.json \
  --report runtime-experiments/answer-support-trigger.json
```

## 实验结果

在 15 道定向漏页题上，8 页基线、直接图文补读、补读加核验分别答对 0、6、3 道。另取 15 道有答案题验证时，三组均答对 3 道。引用不足触发实验在 20 道题上比较，8 页基线答对 13 道，两种补读触发策略均答对 14 道。这些结果来自小样本配对评测。

## ColQwen2.5 和三工具 Agent

在仓库根目录设置 `PYTHONPATH=.:fork` 后，可通过对应模块的 `--help` 查看参数。

- `experiments.replay_colqwen25_retrieval` 复放文本、原生页图检索和融合结果。`--capture-ranks` 保存完整候选排名，供 `experiments.sweep_colqwen25_fusion` 离线扫参。
- `experiments.retrieve_mmlongbench_corpus` 跑全库文档路由。`--question` 可以测试先选文档、再在选中文档内选页的流程。
- `experiments.eval_agent_qa` 在固定的已知 PDF 上评估 Agent 或静态 8 页。`--full` 使用评测目录中的全量 `test-set.jsonl`，默认做分层抽样。`--compare-to` 可复用既有样本清单和对照结果。每题结果写入 JSONL，重启时会跳过已有有效判分的题。
- `experiments.audit_agent_confidence` 在本地重放首轮检索。`experiments.compare_agent_confidence` 汇总两组首轮原始分数、打开新页的比例，以及最终引用页中的字面标记匹配情况。

百炼定版评测使用以下方式启动。私有 dotenv 文件包含 `DASHSCOPE_API_KEY` 和对应地域的 `DASHSCOPE_BASE_URL`。公开的报告只保留聚合指标和冻结配置。

```bash
PYTHONPATH=.:fork .venv/bin/python -m experiments.eval_agent_qa \
  --mode agent --full --root /path/to/benchmark --site /path/to/indexed-site \
  --out runtime-experiments/flash-final --models ./models \
  --visual-python .visual-venv/bin/python --visual-model /path/to/colqwen25-snapshot \
  --visual-base-model ./models/qwen-vl --bailian-thinking on \
  --answer-api-model qwen3.8-flash --env-file .bailian.env --concurrency 6
```

输入目录沿用上文的索引和基线日志。全量评测还需要规范的题集清单，以及通过 ColQwen2.5 构建的 `colqwen25-pages.pt` 文件。

### 结果

| 测试配置 | 正确率 | 平均耗时 |
| --- | ---: | ---: |
| Qwen3.8 Flash 开思考，1,071 题定版 | 698/1,071（65.17%） | 19.29 秒 |
| Qwen3.8 Flash 开思考，同批 85 题 | 64/85（75.29%） | 18.04 秒 |
| DeepSeek V4.1 Flash-1017 开思考，同批 85 题 | 60/85（70.59%） | 12.13 秒 |

全量评测给定每题所属 PDF，其中 4 题被服务端拦截，耗时基于 1,067 道有效判分题。DeepSeek 有 1 道题在标准输出上限下反复截断，补跑时将该题上限提高到 8,192 Token，最终判错。

[全量定版报告](results/qwen38-flash-full.json) · [85 题模型对照](results/deepseek-85-comparison.json) · [误答与正确作答的离线统计](results/confidence-audit.json)
