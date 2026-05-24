# Rerank 实验

## 实验目标

观察重排序是否能提升候选上下文的相关性。

## 实验配置

- 运行时间：2026-05-23
- 运行脚本：`experiments/reranker_experiment.py`
- 运行命令：`python experiments\reranker_experiment.py --retrieval-top-k 5 --rerank-top-k 3 --methods none,overlap`
- embedding 模型：`D:\Projects\Codex\Projects\RAG_Project\models\bge-small-zh-v1.5`
- 混合检索参数：`HYBRID_ALPHA=0.7`
- 召回数量：`RETRIEVAL_TOP_K=5`
- 重排输出：`RERANK_TOP_K=3`

## 总体结果

| 方法 | Hit@1 | 平均召回耗时(ms) | 平均重排耗时(ms) | 平均总耗时(ms) |
| --- | --- | --- | --- | --- |
| none | 100.00% | 12.19 | 0.02 | 12.21 |
| overlap | 100.00% | 10.70 | 0.21 | 10.92 |

## 实验记录

| 测试问题 | 是否启用 Rerank | Top-K 结果变化 | 回答是否正确 | 响应速度 | 备注 |
| --- | --- | --- | --- | --- | --- |
| RTX4090 显存是多少 | 否（none） | `doc_0001 > doc_0004 > doc_0002` -> 不变 | 正确 | 快 | Top1 保持命中 |
| ERR_CONN_RESET 怎么排查 | 否（none） | `doc_0002 > doc_0004 > doc_0005` -> 不变 | 正确 | 快 | Top1 保持命中 |
| HYBRID_ALPHA 是什么 | 否（none） | `doc_0003 > doc_0005 > doc_0004` -> 不变 | 正确 | 快 | Top1 保持命中 |
| RAG 为什么先检索再回答 | 否（none） | `doc_0004 > doc_0005 > doc_0003` -> 不变 | 正确 | 快 | Top1 保持命中 |
| 纯关键词检索用什么 | 否（none） | `doc_0005 > doc_0003 > doc_0004` -> 不变 | 正确 | 快 | Top1 保持命中 |
| RTX4090 显存是多少 | 是（overlap） | `doc_0001 > doc_0004 > doc_0002` -> 不变 | 正确 | 略慢 | 该 query 已在召回阶段较优 |
| ERR_CONN_RESET 怎么排查 | 是（overlap） | `doc_0002 > doc_0004 > doc_0005` -> 不变 | 正确 | 略慢 | 该 query 已在召回阶段较优 |
| HYBRID_ALPHA 是什么 | 是（overlap） | `doc_0003 > doc_0005 > doc_0004` -> 不变 | 正确 | 略慢 | 该 query 已在召回阶段较优 |
| RAG 为什么先检索再回答 | 是（overlap） | `doc_0004 > doc_0005 > doc_0003` -> 不变 | 正确 | 略慢 | 该 query 已在召回阶段较优 |
| 纯关键词检索用什么 | 是（overlap） | `doc_0005 > doc_0003 > doc_0004` -> 不变 | 正确 | 略慢 | 该 query 已在召回阶段较优 |

## 结论

1. 在当前小规模教学语料中，`none` 与 `overlap` 的 TopK 排序一致，Hit@1 均为 100%。  
2. 说明当前 query 集对 rerank 区分度不足，尚不能体现精排优势。  
3. 下一步应补“更难 query”（同义改写、弱关键词、长句噪声）再评估 rerank 价值。  
4. 代码链路已具备：可对接更强的 `cross_encoder` 方法做进一步实验。  

## 补充：“看不出效果”

当召回阶段已经把目标文档稳定放在 Top1 时，重排序很可能不会改变排序。  
这是数据集难度不足导致的正常现象。

为了解决这个问题，实验脚本新增了 `toy` 模式：

```powershell
python experiments\reranker_experiment.py --mode toy --methods none,overlap
```

该模式使用“故意错序”的候选集，能直观看到：

```text
none:    重排前/后 Hit@1 = 0%
overlap: 重排后 Hit@1 = 100%，Top1 发生变化
```
