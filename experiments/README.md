# 实验与评测

## 定版指标

代码冻结在 [`5ad549e`](https://github.com/11Dz-dz/multimodal-pdf-rag/commit/5ad549ee95bb8eaa609df96e55cb578a765b0fec)。端到端问答使用改提示词前的三工具 Agent、百炼 `qwen3.8-flash` 开思考、首轮 3 页、最多 4 轮、并发 6。机器可读指标、源码哈希和模型配置见 [定版指标 JSON](results/main-5ad549e-metrics.json)。

### 检索

| 指标 | 已知目标 PDF（842 题） | 135 份 PDF 全库路由（863 题） |
| --- | ---: | ---: |
| Hit@1 | — | 74.62%（644/863） |
| Hit@5 | 90.62%（763/842） | 83.31%（719/863） |
| Recall@5 | 82.02% | — |
| MRR@5 | 0.7709 | 0.7792 |
| Top-8 全证据页覆盖率 | **81.71%（688/842）** | — |
| 跨页题 Top-8 全证据页覆盖率 | 65.27%（233/357） | — |

已知目标 PDF 的指标以页为单位，全库路由以文档为单位。Hit@5 表示前 5 个结果至少命中一次，Recall@5 是每题证据页召回比例的平均，MRR@5 是首个命中排名倒数的平均。漏检在 MRR@5 中记 0。全证据页覆盖要求同一题的全部标注页进入前 8。表内指标来自离线排序回放。

### 问答

| 指标 | 结果 |
| --- | ---: |
| 总正确率 | **73.30%（785/1,071）** |
| 有答案题正确率 | 77.17%（666/863） |
| 单页题正确率 | 86.53%（424/490） |
| 跨页题正确率 | 65.10%（235/361） |
| 无答案题判对率 | 57.21%（119/208） |
| 有答案题误拒率 | 4.29%（37/863） |
| 首轮送齐全部证据页 | 65.16%（546/838） |
| 多轮结束后送齐全部证据页 | 79.24%（664/838） |
| 平均 / P50 / P95 耗时 | 19.29 / 12.54 / 55.34 秒 |
| 平均模型调用轮数 / 图片数 | 2.03 轮 / 4.09 张 |

问答评测每题指定目标 PDF，1,071 题来自 134 份 PDF。正确率分母包含 4 道接口失败题。证据送齐率的分母是有证据页标注且成功完成评测的 838 题。耗时按 1,067 道有效请求统计，计入检索、工具执行和生成。

[原始全量报告](results/qwen38-flash-full.json)用本地 Qwen2.5-VL-3B 判分，结果为 698/1,071（65.17%）。定版结果用 `qwen3.8-max` 对这批原始答案重新判分，生成过程和延迟取自原始日志。

## 运行评测

### 验证单份 PDF

按[首页说明](../README.md)启动 Demo，上传 PDF 后取上传接口返回的文档 ID。准备一个标注文件，例如下面的结构。页码从 1 开始，空数组代表预期拒答。

```json
[
  {"question": "文档内示例问题", "pages": [2, 3]},
  {"question": "文档未提供的信息", "pages": []}
]
```

```bash
.venv/bin/python evaluate.py --url http://127.0.0.1:8094 \
  --doc-id YOUR_DOCUMENT_ID --cases annotated-questions.json
```

该脚本通过 `/ask` 检查候选证据页与拒答行为。`mrr_at_returned_pages` 按接口返回的全部候选页计算。

### 在已有评测目录上运行全量 Agent

从仓库根目录执行。`--root` 需要 `test-set.jsonl` 和归档 `full-run-haiku/responses.jsonl`。`--site` 需要已入库的 `uploads/<doc_id>/`，包含 `source.pdf`、`document.json`、MinerU 解析结果、`resume-index-v1/chunks.json`、Qdrant 索引和 `colqwen25-pages.pt`。Git 仓库提供脚本与聚合报告，运行所需的原始归档和索引需自行准备。

将 `DASHSCOPE_API_KEY` 和对应地域的 `DASHSCOPE_BASE_URL` 配置在 `.bailian.env`。以下命令按原始定版配置生成答案和本地 3B 判分结果。

```bash
PYTHONPATH=.:fork .venv/bin/python -m experiments.eval_agent_qa \
  --mode agent --full --root /path/to/benchmark --site /path/to/indexed-site \
  --out runtime-experiments/flash-final --models ./models \
  --visual-python .visual-venv/bin/python \
  --visual-model /path/to/colqwen25-snapshot \
  --visual-base-model ./models/qwen-vl --bailian-thinking on \
  --answer-api-model qwen3.8-flash --env-file .bailian.env --concurrency 6
```

命令默认使用 `cuda:0` 做文本检索、`cuda:1` 做视觉检索。单卡可用 `--visual-device cuda:0`。重新排序请用独立的索引副本，以免争用 Qdrant 的本地目录锁。

`--mode static` 可运行静态 8 页对照。去掉 `--full` 会分层抽样 85 题。`--compare-to` 读取已有同批样本与结果。每题输出写入 JSONL，重启沿用已成功判分的记录。

### 回放检索

| 模块 | 用途 |
| --- | --- |
| [replay_colqwen25_retrieval.py](replay_colqwen25_retrieval.py) | 回放文本和页图检索，`--capture-ranks` 保存排序供离线分析 |
| [sweep_colqwen25_fusion.py](sweep_colqwen25_fusion.py) | 在已保存的候选上比较页级融合与文档级融合 |
| [retrieve_mmlongbench_corpus.py](retrieve_mmlongbench_corpus.py) | 评测全库文档路由，`--question` 可检查单个问题的文档和页级排序 |
| [audit_agent_confidence.py](audit_agent_confidence.py) · [compare_agent_confidence.py](compare_agent_confidence.py) | 分析首轮检索分数、新开页与最终引用证据 |

使用 `PYTHONPATH=.:fork .venv/bin/python -m experiments.<模块名> --help` 查看各脚本参数。

## 历史实验

- [静态拒答补读](replay_visual_backfill.py)比较首轮 8 页、追加第 9–12 名候选页、追加页再做独立核验。在 15 道定向漏页题上分别答对 0、6、3 道。`--holdout` 使用与首批不重叠的 30 题。
- [引用触发补读](replay_answer_support_trigger.py)在 20 道小样本上比较固定 8 页、拒答触发和引用不足触发，分别答对 13、14、14 道。样本号见 [sample-manifest.json](sample-manifest.json)。
- [85 题模型对照](results/deepseek-85-comparison.json)沿用原始判分。Qwen3.8 Flash 开思考答对 64/85，平均 18.04 秒。DeepSeek V4.1 Flash-1017 开思考答对 60/85，平均 12.13 秒。
- [置信度审计](results/confidence-audit.json)对比无答案误答与有答案正确作答的首轮检索分数和引用页。
