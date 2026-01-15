# 阿拉伯语 OCR 基准测试工具

[English](./README.md)

对比测试 Gemini / Mistral OCR 模型的阿拉伯语识别准确率。

## 目录结构

```
benchmark/
├── benchmark_test.py      # 测试脚本
├── config.example.env     # 配置模板
├── .env                   # API密钥 (需自行创建)
├── requirements.txt       # 依赖
├── samples/               # 测试PDF文件
├── ground_truth/          # 人工校对文本
└── results/               # 测试报告输出
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**系统依赖** (pdf2image 需要):
- macOS: `brew install poppler`
- Ubuntu: `apt-get install poppler-utils`
- Windows: 下载 [poppler](https://github.com/oschwartz10612/poppler-windows/releases)

### 2. 配置 API 密钥

```bash
cp config.example.env .env
# 编辑 .env 填入你的 API 密钥
```

```env
GEMINI_API_KEY=your_gemini_key
MISTRAL_API_KEY=your_mistral_key
PRIMARY_LANGUAGE=Arabic
PDF_DPI=200
```

### 3. 准备测试样本

1. 将 PDF 放入 `samples/` 目录
2. 在 `ground_truth/` 创建对应的 `.txt` 文件（文件名需与 PDF 相同）

**重要**: ground_truth 文件需要用 `---` 分隔每一页的内容：

```text
第一页内容...
这是第一页的文字

---
第二页内容...
这是第二页的文字

---
第三页内容...
```

### 4. 运行测试

```bash
# 测试所有模型
python benchmark_test.py

# 只测试 Gemini 模型
python benchmark_test.py --gemini-only

# 只测试 Mistral 模型
python benchmark_test.py --mistral-only

# 指定特定模型
python benchmark_test.py --models gemini-3-pro-preview mistral-ocr-latest
```

## 可用模型

| 模型 | 类型 | 说明 |
|------|------|------|
| `gemini-3-pro-preview` | Gemini | 最高准确率，较慢 |
| `gemini-3-flash-preview` | Gemini | 快速，准确率较高 |
| `gemini-2.5-pro` | Gemini | 高准确率，成本适中 |
| `gemini-2.5-flash` | Gemini | 最快，性价比高 |
| `mistral-ocr-latest` | Mistral | 专用 OCR 模型 |

## 测试报告

报告保存在 `results/` 目录，包含：

- `benchmark_report_YYYYMMDD_HHMMSS.md` - Markdown 格式报告
- `benchmark_raw_YYYYMMDD_HHMMSS.json` - 原始数据 (JSON)

### 指标说明

| 指标 | 全称 | 说明 |
|------|------|------|
| **CER** | Character Error Rate | 字符级错误率，越低越好 |
| **WER** | Word Error Rate | 词级错误率，越低越好 |

## 实测结果参考

以下是 14 页中文+阿拉伯语混合 PDF 的测试结果 (2026-01-12)：

| 模型 | CER ↓ | WER ↓ | 速度 | 成本/页 |
|------|-------|-------|------|---------|
| gemini-3-pro-preview | **1.77%** | **4.27%** | 35s/页 | $0.0034 |
| gemini-2.5-pro | 2.12% | 4.93% | 25s/页 | $0.0025 |
| gemini-3-flash-preview | 3.19% | 4.68% | 13s/页 | $0.0009 |
| gemini-2.5-flash | 5.69% | 8.68% | 7s/页 | $0.0005 |
| mistral-ocr-latest | 12.01% | 15.47% | 3s/页 | $0.002 |

> **定价说明**：成本估算基于 2026-01 Vertex AI 官方定价（context ≤200K）。每页按约 800 input tokens（图片）+ 300 output tokens 计算。实际成本因图片分辨率和输出文本长度而异。

**结论**: `gemini-2.5-flash` 性价比最优（成本最低）；`gemini-3-pro-preview` 准确率最高但成本较高。

## 常见问题

### Q: ground_truth 没有按页分隔怎么办？

脚本会警告，但仍可运行。此时只输出整文档对比，不输出逐页对比。

### Q: 如何获取 ground_truth？

1. 手动人工校对 PDF 内容
2. 或使用最高准确率模型 (gemini-3-pro-preview) 识别后校对
3. 每页内容之间用单独一行 `---` 分隔

### Q: 测试大文件很慢？

- 降低 `PDF_DPI` 值 (如 150)
- 使用 `--models` 只测试需要的模型
- 先用少量页面测试，确认效果后再批量处理
