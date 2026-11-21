# 视觉问答补读实验

这些脚本在已指定目标 PDF 的 MMLongBench-Doc 样本上复放固定的检索排序，并实际调用配置的视觉作答模型。它们使用现有索引和基线日志，适合比较同一批题在不同作答策略下的变化。

- `replay_visual_backfill.py` 对比首轮 8 页、拒答后补读第 9–12 名候选页，以及补读加独立核验。补读输入包括新增页的页图和首轮候选页的文本。`--holdout` 使用与首批样本不重叠的 30 题。
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
