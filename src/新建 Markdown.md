main.py：适合命令行或最小项目入口，负责串联流程。 

后续：

src/config.py
src/document_loader.py
src/text_splitter.py
src/embeddings.py
src/vector_store.py
src/retriever.py
src/generator.py
src/main.py

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
5. 只想继续第 8 节学习时，可先运行 BM25-only：

```powershell
python experiments\retrieval_compare.py --bm25-only
```

### 排查顺序（推荐）

1. 先测 `import torch` 是否成功。  
2. 再测 `SentenceTransformer(...)` 是否成功。  
3. 最后再跑 `faiss_experiment.py` 或 `retrieval_compare.py`。  
