
---

## FAISS 实验报错问题记录（WinError 1114 / 1455）

### 现象

运行 `experiments/faiss_experiment.py` 或 `experiments/retrieval_compare.py` 时，报错类似：

```text
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败
Error loading ... torch\lib\shm.dll
```

或：

```text
OSError: [WinError 1455] 页面文件太小，无法完成操作
```

### 根因判断

这类报错不是 FAISS 索引逻辑本身报错。  
失败发生在更前面的模型加载阶段：

```text
embeddings.py -> sentence_transformers -> transformers -> torch DLL 初始化
```

即：向量模型未成功加载，导致后续 FAISS 还没开始执行就中断。

### 关键验证命令

```powershell
python -c "import torch; print(torch.__version__)"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(r'models\bge-small-zh-v1.5'); print('ok')"
```

如果两条命令都成功，说明当前环境可继续运行 FAISS 实验。

### 解决方案

1. 优先使用新开的 PowerShell 窗口重试。  
2. 运行前限制线程，降低资源冲突概率：

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
```

3. 关闭占内存进程（IDE 里的旧 Python 进程、浏览器等）。  
4. 若仍频繁报错，调大 Windows 页面文件并重启系统。  
5. 可先运行 BM25-only：

```powershell
python experiments\retrieval_compare.py --bm25-only
```

### 排查顺序（推荐）

1. 先测 `import torch` 是否成功。  
2. 再测 `SentenceTransformer(...)` 是否成功。  
3. 最后再跑 `faiss_experiment.py` 或 `retrieval_compare.py`。  

## Reranker 模型加载失败

最开始下载并尝试加载的是 `bge-reranker-v2-m3`，但这个模型较大（`model.safetensors` 约 2GB+），加载时报：

```
OSError: 页面文件太小，无法完成操作。 (os error 1455)
```

因此后续改用体量更小的 `bge-reranker-base` 做验证。  
用底层 `transformers` 测试：

```
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; p=r'models\bge-reranker-base'; tokenizer=AutoTokenizer.from_pretrained(p); print('tokenizer ok'); model=AutoModelForSequenceClassification.from_pretrained(p); print('model ok')"
```

输出到了：

```
tokenizer ok
```

然后报页面文件错误。

这说明（针对 `bge-reranker-base` 的验证）：

```
tokenizer 能加载
模型路径没问题
模型文件基本存在
真正失败的是 model.safetensors 权重加载
```

根因是：**当时系统可用虚拟内存不足**。

查到：FreeVirtualMemory = 2880804 KB.约等于：2.8GB

这个可用空间对 reranker 模型加载来说太少。即使模型权重文件只有 1GB～2GB，实际加载时还需要 PyTorch、Transformers、safetensors、中间缓存、连续虚拟内存空间等额外开销。

------

### 解决方案

有效方案是：

```
关闭占内存程序 / 重启电脑
释放可用虚拟内存
再重新加载 reranker 模型
```

如果仍然不够，可以调整 Windows 虚拟内存（建议按阶段逐步上调，而不是一次性写死）：

```
Win + R
输入 sysdm.cpl
高级
性能 → 设置
高级
虚拟内存 → 更改
```

建议值（可按机器情况选择其一）：

```
方案 A（常用）：
初始大小：16384 MB
最大大小：32768 MB

方案 B（模型较大或并发较高）：
初始大小：32768 MB
最大大小：65536 MB
```

优先从方案 A 开始；若仍报 1455，再升到方案 B。

设置后需要重启。

---

## Gradio 启动 502（startup-events）代理问题

### 现象

运行：

```powershell
python experiments\gradio_app_experiment.py
```

出现报错：

```text
Couldn't start the app because
'http://127.0.0.1:7860/gradio_api/startup-events' failed (code 502)
```

### 根因判断

这不是 Gradio 业务逻辑报错，而是本机代理设置影响了 localhost 回环请求。  
表现为：页面地址已经打印出来，但内部 startup-events 请求被代理链路干扰，返回 502。

### 解决命令（当前项目可用）

```powershell
$env:NO_PROXY="localhost,127.0.0.1,::1"
$env:no_proxy="localhost,127.0.0.1,::1"
python experiments\gradio_app_experiment.py
```

### 验证标准

终端出现类似输出即为修复成功：

```text
* Running on local URL:  http://127.0.0.1:7860
```

并且浏览器可正常打开页面、上传文档和提问。
