# Hybrid 检索实验

## 实验目标

观察 BM25 与 FAISS 融合后，不同 `HYBRID_ALPHA` 对排序结果的影响。

## 实验配置

- 运行时间：2026-05-23
- 运行脚本：`experiments/hybrid_experiment.py`
- 运行命令：`python experiments\hybrid_experiment.py --top-k 3 --alphas 0.3,0.7,1.0`
- 文档数：5
- 查询数：5
- TopK：3
- embedding 模型：`D:\Projects\Codex\Projects\RAG_Project\models\bge-small-zh-v1.5`

## 总体结果

| HYBRID_ALPHA | Hit@1 | 说明 |
| --- | --- | --- |
| 0.3 | 100.00% | 偏关键词，Top1 全命中 |
| 0.7 | 100.00% | 语义与关键词平衡，Top1 全命中 |
| 1.0 | 100.00% | 纯语义，Top1 仍全命中 |

## 分 query 结果记录

### alpha = 0.3

| 测试问题 | Top1 doc_id | 是否符合预期 | Top3 摘要 | 备注 |
| --- | --- | --- | --- | --- |
| RTX4090 显存是多少 | doc_0001 | 是 | doc_0001 > doc_0004 > doc_0002 | Top1 命中显存文档 |
| ERR_CONN_RESET 怎么排查 | doc_0002 | 是 | doc_0002 > doc_0004 > doc_0005 | 术语命中稳定 |
| HYBRID_ALPHA 是什么 | doc_0003 | 是 | doc_0003 > doc_0005 > doc_0004 | 配置项文档排第一 |
| RAG 为什么先检索再回答 | doc_0004 | 是 | doc_0004 > doc_0005 > doc_0003 | 第二第三名有语义相关干扰 |
| 纯关键词检索用什么 | doc_0005 | 是 | doc_0005 > doc_0003 > doc_0004 | BM25 相关文档占优 |

### alpha = 0.7

| 测试问题 | Top1 doc_id | 是否符合预期 | Top3 摘要 | 备注 |
| --- | --- | --- | --- | --- |
| RTX4090 显存是多少 | doc_0001 | 是 | doc_0001 > doc_0004 > doc_0002 | Top1 稳定 |
| ERR_CONN_RESET 怎么排查 | doc_0002 | 是 | doc_0002 > doc_0004 > doc_0005 | Top1 稳定 |
| HYBRID_ALPHA 是什么 | doc_0003 | 是 | doc_0003 > doc_0005 > doc_0004 | Top1 稳定 |
| RAG 为什么先检索再回答 | doc_0004 | 是 | doc_0004 > doc_0005 > doc_0003 | 与 alpha=0.3 相同 |
| 纯关键词检索用什么 | doc_0005 | 是 | doc_0005 > doc_0003 > doc_0004 | 与 alpha=0.3 相同 |

### alpha = 1.0

| 测试问题 | Top1 doc_id | 是否符合预期 | Top3 摘要 | 备注 |
| --- | --- | --- | --- | --- |
| RTX4090 显存是多少 | doc_0001 | 是 | doc_0001 > doc_0004 > doc_0002 | Top1 稳定 |
| ERR_CONN_RESET 怎么排查 | doc_0002 | 是 | doc_0002 > doc_0004 > doc_0005 | Top1 稳定 |
| HYBRID_ALPHA 是什么 | doc_0003 | 是 | doc_0003 > doc_0005 > doc_0004 | Top1 稳定 |
| RAG 为什么先检索再回答 | doc_0004 | 是 | doc_0004 > doc_0005 > doc_0002 | 第三名变为 doc_0002 |
| 纯关键词检索用什么 | doc_0005 | 是 | doc_0005 > doc_0003 > doc_0002 | 第三名变为 doc_0002 |

## 关键观察

1. 当前这组 5 条 query 上，`alpha=0.3/0.7/1.0` 的 **Top1 全部命中**，Hit@1 都是 100%。  
2. `alpha` 变化主要影响 **Top2/Top3 的相对排序**，说明融合参数在这组数据上更多体现在“候选重排”，而不是 Top1 决策。  
3. 当 `alpha=1.0`（纯语义）时，个别 query 的第三名发生变化（如 `RAG 为什么先检索再回答`、`纯关键词检索用什么`），表现为更偏语义邻近结果。  

## 结论

- 在当前小规模教学语料上，`HYBRID_ALPHA=0.7` 是合理默认值：兼顾语义与关键词，且 Top1 稳定。  
- 仅看 Hit@1 无法拉开三组参数差异，下一步应扩充更难 query（同义改写、术语歧义、长尾表达）再做参数选择。  
- 本节完成后可进入第 10 节：在“召回已稳定”的前提下，用 Reranker 优化最终排序质量。  
