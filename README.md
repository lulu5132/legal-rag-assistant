# Legal RAG Assistant

面向法律/专利场景的最小可运行 RAG 实验框架（LlamaIndex + FAISS）。

## 已实现内容

- 本地文档读取与向量化索引
- FAISS 本地持久化索引
- 混合检索（语义 + 关键词）
- Embedding 相似度 rerank
- JSON Schema 结构化输出
- 问答检索与来源片段预览
- DeepSeek API 与 Ollama 本地双模式
- Embedding 双模式（本地 / API）
- Ollama 本地模型支持（默认启用）

## 依赖说明

项目已包含 `requirements.txt`，建议直接安装：

```bash
pip install -r requirements.txt
```

说明：

- 该文件使用了兼容区间版本，避免环境里无关包被 `pip freeze` 一并写入。
- 如需生成“实验快照版”依赖，可另存为 `requirements.lock.txt`，不要覆盖当前 `requirements.txt`。

## 目录结构

- `src/` 核心代码
- `data/raw/` 原始法律/专利文档
- `data/processed/` 清洗与切分后数据（当前索引默认读取这里）
- `indexes/faiss/` 向量索引
- `prompts/` 提示词模板
- `evaluation/datasets/` 评测数据集
- `evaluation/results/` 评测结果
- `config/` 配置文件
- `notebooks/` 实验笔记
- `scripts/` 实用脚本
- `logs/` 运行日志
- `docs/` 项目文档
- `tests/` 测试代码

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，按使用模式填写：

- `DEEPSEEK_API_KEY=...`
- `OLLAMA_API_KEY=ollama`

3. 准备文档

将待检索文档放入：

- `data/processed/`

4. 首次构建索引并提问

```bash
PYTHONPATH=. python scripts/run_pipeline.py --config config/settings.example.yaml --rebuild --query "请总结该专利草案的核心创新点"
```

后续查询无需重建索引（去掉 `--rebuild`）。

### 本地快速模式（推荐先跑通）

当本地模型推理较慢或你在调试流程时，建议先使用快速配置：

```bash
PYTHONPATH=. python scripts/run_pipeline.py --config config/settings.local.fast.yaml --query "请总结该专利草案的核心创新点"
```

该配置做了以下优化：

- 降低检索候选数量（更快）
- 关闭 JSON Schema 结构化输出（更稳）
- 保持本地 Ollama 模式

## 模型切换

在 `config/settings.example.yaml` 修改：

- `model.provider`: `deepseek` 或 `ollama`
- `model.llm_model`: 例如 `deepseek-chat` 或 `deepseek-r1:8b`
- `model.api_base`: DeepSeek 用 `https://api.deepseek.com/v1`；Ollama 用 `http://localhost:11434/v1`

当前默认配置为本地 Ollama：

- `provider: ollama`
- `llm_model: deepseek-r1:8b`
- `api_base: http://localhost:11434/v1`

运行时脚本会打印当前使用的 provider、model 和 api_base。只要看到 `provider: ollama`，就表示使用本地模型，不会调用 DeepSeek 云 API，也不会产生云端 API 费用。

## 超时与稳定性说明

针对本地 Ollama 长推理场景，项目已做如下处理：

- Ollama 请求不再使用固定短超时，避免长思考被提前中断
- 提供 `request_timeout_sec` 与 `max_retries` 配置项
- 结构化输出失败时会返回可解析的兜底 JSON，而不是直接崩溃退出

如果你希望最高稳定性，优先使用 `config/settings.local.fast.yaml`；功能确认后再切回 `config/settings.example.yaml`。

## 效果对比（示例）

以下是推荐放在简历/仓库中的对比维度示例。数值请用你自己的实测结果替换。

| 评估维度 | DeepSeek API (deepseek-chat) | DeepSeek-R1 (Ollama Local) |
| :--- | :--- | :--- |
| 法律术语准确度 | 高 | 高 |
| 推理耗时 | 2.0s - 4.0s | 4.0s - 12.0s (RTX 4090) |
| 单次调用费用 | 按 token 计费 | 0 元 API 费用 |
| 隐私性 | 需传输到云端 | 数据留在本机/本地服务器 |

建议在 `evaluation/results/` 保存对比结果（例如 CSV/Markdown 报告），并在此表中引用真实数据。

## 数据隐私说明

本项目支持通过 Ollama 本地化部署 DeepSeek 模型：

- 法律敏感数据可不出本地服务器
- 适合对隐私与合规要求更高的法律/专利场景
- 在离线或弱网环境下仍可运行核心检索与生成流程

## Ollama 本地部署

安装 Ollama 后，启动服务：

```bash
ollama serve
```

下载模型：

```bash
ollama pull deepseek-r1:8b
```

验证模型是否存在：

```bash
curl -s http://127.0.0.1:11434/api/tags
```

如果输出里能看到 `deepseek-r1:8b`，说明本地模型已准备完成。

## 发布到 GitHub

当前仓库建议不要上传以下内容：

- `.env`
- 本地模型目录 `models/`
- 索引目录 `indexes/`
- 日志目录 `logs/`
- 真实业务文档 `data/raw/` 和 `data/processed/`

这些内容已经写入 `.gitignore`。

## 什么是“本地 Embedding”

本地 Embedding 指的是：向量化模型在你自己的机器上运行，不把原文传给第三方 API。

当前默认配置：

- `embedding_type: local`
- `embedding_model: models/bge-small-zh-v1.5`

优点：

- 更好数据隐私（法律场景很重要）
- 降低长期 API 成本

代价：

- 首次会下载模型
- CPU 速度通常慢于云端高性能服务

## 本地 Embedding 部署（你当前环境）

由于你环境里的 PyTorch 是 2.1.2，建议保持现有 Torch，不要强制重装。按下面执行：

```bash
pip install -r requirements.txt
```

然后通过镜像下载到项目本地目录（首次会较慢）：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5')"
```

下载后 `config/settings.example.yaml` 默认已经指向本地目录：

- `embedding_model: models/bge-small-zh-v1.5`

## 报错说明（你遇到的 NameError: nn）

这是 `sentence-transformers/transformers` 与当前环境版本组合不兼容导致。当前仓库已通过两种方式修复：

- 依赖版本锁定为兼容组合
- Embedding 改为按需导入，避免未启用本地 embedding 时也触发该错误
