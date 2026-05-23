# Chunk 实验

## 实验目标

观察 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 对检索效果的影响。

## 运行命令

在项目根目录运行：

```powershell
python src\test_text_splitter.py
python src\chunk_experiment.py
```

## 当前测试文件

```text
test_files/sample_rag.txt
```

测试文件较短，实验脚本会把文本重复 4 次，方便观察分块参数变化。

## 实验记录

| CHUNK_SIZE | CHUNK_OVERLAP | chunk 数量 | 最短 chunk | 最长 chunk | 平均长度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 0 | 8 | 146 | 197 | 171.4 | 小 chunk，无重叠，数量较多 |
| 200 | 40 | 12 | 101 | 186 | 150.2 | 小 chunk，有重叠，数量更多 |
| 400 | 40 | 4 | 340 | 388 | 374.0 | 当前推荐观察参数 |
| 800 | 40 | 2 | 687 | 735 | 711.0 | chunk 较大，数量少 |
| 800 | 100 | 2 | 735 | 748 | 741.5 | chunk 较大，重叠更多 |

## 结论

```text
1. chunk_size 越小，chunk 数量越多，检索粒度更细，但上下文更容易被切碎。
2. chunk_overlap 会增加重复内容，因此 chunk 数量和平均长度都会变化。
3. 当前测试文本较短，chunk_size=400、overlap=40 比较适合观察边界和内容完整性。
4. 后续进入 embedding 和检索后，需要继续用真实问题验证不同分块参数的检索效果。
```
