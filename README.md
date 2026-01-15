# Arabic OCR Benchmark Tool

[中文文档](./README.zh-CN.md)

Benchmark tool for comparing Gemini / Mistral OCR models on Arabic text recognition accuracy.

## Directory Structure

```
benchmark/
├── benchmark_test.py      # Main benchmark script
├── config.example.env     # Configuration template
├── .env                   # API keys (create your own)
├── requirements.txt       # Dependencies
├── samples/               # Test PDF files
├── ground_truth/          # Human-verified text
└── results/               # Benchmark reports
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**System dependencies** (required by pdf2image):
- macOS: `brew install poppler`
- Ubuntu: `apt-get install poppler-utils`
- Windows: Download [poppler](https://github.com/oschwartz10612/poppler-windows/releases)

### 2. Configure API Keys

```bash
cp config.example.env .env
# Edit .env and add your API keys
```

```env
GEMINI_API_KEY=your_gemini_key
MISTRAL_API_KEY=your_mistral_key
PRIMARY_LANGUAGE=Arabic
PDF_DPI=200
```

### 3. Prepare Test Samples

1. Place PDF files in `samples/` directory
2. Create corresponding `.txt` files in `ground_truth/` (filename must match PDF)

**Important**: Separate each page's content with `---` in ground_truth files:

```text
First page content...

---
Second page content...

---
Third page content...
```

### 4. Run Benchmark

```bash
# Test all models
python benchmark_test.py

# Test Gemini models only
python benchmark_test.py --gemini-only

# Test Mistral models only
python benchmark_test.py --mistral-only

# Test specific models
python benchmark_test.py --models gemini-3-pro-preview mistral-ocr-latest
```

## Available Models

| Model | Provider | Notes |
|-------|----------|-------|
| `gemini-3-pro-preview` | Gemini | Highest accuracy, slower |
| `gemini-3-flash-preview` | Gemini | Fast, good accuracy |
| `gemini-2.5-pro` | Gemini | High accuracy, moderate cost |
| `gemini-2.5-flash` | Gemini | Fastest, best value |
| `mistral-ocr-latest` | Mistral | Dedicated OCR model |

## Benchmark Reports

Reports are saved in `results/` directory:

- `benchmark_report_YYYYMMDD_HHMMSS.md` - Markdown report
- `benchmark_raw_YYYYMMDD_HHMMSS.json` - Raw data (JSON)

### Metrics

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **CER** | Character Error Rate | Character-level error rate (lower is better) |
| **WER** | Word Error Rate | Word-level error rate (lower is better) |

## Benchmark Results

Test results on a 14-page Chinese + Arabic mixed PDF (2026-01-12):

| Model | CER ↓ | WER ↓ | Speed | Cost/Page |
|-------|-------|-------|-------|-----------|
| gemini-3-pro-preview | **1.77%** | **4.27%** | 35s/page | $0.0034 |
| gemini-2.5-pro | 2.12% | 4.93% | 25s/page | $0.0025 |
| gemini-3-flash-preview | 3.19% | 4.68% | 13s/page | $0.0009 |
| gemini-2.5-flash | 5.69% | 8.68% | 7s/page | $0.0005 |
| mistral-ocr-latest | 12.01% | 15.47% | 3s/page | $0.002 |

> **Pricing Note**: Cost estimates based on Jan 2026 Vertex AI official rates (context ≤200K). Calculated at ~800 input tokens (image) + 300 output tokens per page. Actual costs vary with image resolution and output text length.

**Conclusion**: `gemini-2.5-flash` offers the best value (lowest cost); `gemini-3-pro-preview` has the highest accuracy but at higher cost.

## FAQ

### Q: What if ground_truth is not page-separated?

The script will warn but continue. Results will be compared at document level instead of page level.

### Q: How to create ground_truth?

1. Manually proofread the PDF content
2. Or use the highest accuracy model (gemini-3-pro-preview) and proofread its output
3. Separate each page with a single line containing `---`

### Q: Testing large files is slow?

- Lower `PDF_DPI` value (e.g., 150)
- Use `--models` to test only specific models
- Test with a few pages first, then process in batch
