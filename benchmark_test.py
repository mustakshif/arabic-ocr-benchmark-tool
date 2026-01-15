#!/usr/bin/env python3
"""
OCR Model Benchmark Script
===========================
对比测试 Gemini / Mistral OCR 模型的准确率。

使用方法：
1. 将测试PDF放入 samples/ 目录
2. 将对应的人工校对文本放入 ground_truth/ 目录 (文件名需对应)
   ⚠️ 重要：ground_truth文件需要用 "---" 分隔每一页的内容，例如：
   
   第一页内容...
   ---
   第二页内容...
   ---
   第三页内容...
   
3. 复制 config.example.env 为 .env 并填入API密钥
4. 运行: python benchmark_test.py
5. 查看 results/ 目录的测试报告
"""

import os
import sys
import json
import time
import base64
import difflib
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

try:
    from google import genai
    from google.genai import types
    from mistralai import Mistral
    from pdf2image import convert_from_path
    from PIL import Image
    from dotenv import load_dotenv
    from tqdm import tqdm
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    gemini_api_key: str = ""
    mistral_api_key: str = ""
    samples_dir: Path = field(default_factory=lambda: Path("samples"))
    ground_truth_dir: Path = field(default_factory=lambda: Path("ground_truth"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    pdf_dpi: int = 200
    primary_language: str = "Arabic"
    
    @classmethod
    def from_env(cls, base_dir: Path) -> "BenchmarkConfig":
        load_dotenv(base_dir / ".env")
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
            samples_dir=base_dir / "samples",
            ground_truth_dir=base_dir / "ground_truth",
            results_dir=base_dir / "results",
            pdf_dpi=int(os.getenv("PDF_DPI", "200")),
            primary_language=os.getenv("PRIMARY_LANGUAGE", "Arabic")
        )


@dataclass
class OCRResult:
    model_name: str
    sample_name: str
    page_num: int
    ocr_text: str
    ground_truth: str
    cer: float  # Character Error Rate
    wer: float  # Word Error Rate
    processing_time: float
    cost_estimate: float
    error: Optional[str] = None


def calculate_cer(ocr_text: str, ground_truth: str) -> float:
    """Character Error Rate = Levenshtein距离 / ground_truth长度"""
    if not ground_truth:
        return 1.0 if ocr_text else 0.0
    
    matcher = difflib.SequenceMatcher(None, ground_truth, ocr_text)
    distance = sum(
        max(i2 - i1, j2 - j1) 
        for tag, i1, i2, j1, j2 in matcher.get_opcodes() 
        if tag != 'equal'
    )
    return min(distance / len(ground_truth), 1.0)


def calculate_wer(ocr_text: str, ground_truth: str) -> float:
    """Word Error Rate"""
    gt_words = ground_truth.split()
    ocr_words = ocr_text.split()
    
    if not gt_words:
        return 1.0 if ocr_words else 0.0
    
    matcher = difflib.SequenceMatcher(None, gt_words, ocr_words)
    distance = sum(
        max(i2 - i1, j2 - j1) 
        for tag, i1, i2, j1, j2 in matcher.get_opcodes() 
        if tag != 'equal'
    )
    return min(distance / len(gt_words), 1.0)


def image_to_base64(image: Image.Image) -> str:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class GeminiTester:
    # Pricing as of Jan 2026 (Google AI Studio / Vertex AI standard rates, context ≤200K)
    # Pro and Gemini-3 models have thinking tokens billed separately
    # Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
    MODELS = {
        "gemini-3-pro-preview": {
            "name": "gemini-3-pro-preview",
            "input_cost_per_million": 2.00,
            "output_cost_per_million": 6.00,
            "thinking_cost_per_million": 1.00
        },
        "gemini-3-flash-preview": {
            "name": "gemini-3-flash-preview",
            "input_cost_per_million": 0.50,
            "output_cost_per_million": 1.50,
            "thinking_cost_per_million": 0.30
        },
        "gemini-2.5-pro": {
            "name": "gemini-2.5-pro",
            "input_cost_per_million": 1.25,
            "output_cost_per_million": 5.00,
            "thinking_cost_per_million": 1.00
        },
        "gemini-2.5-flash": {
            "name": "gemini-2.5-flash",
            "input_cost_per_million": 0.30,
            "output_cost_per_million": 1.00
        }
    }

    def __init__(self, api_key: str, language: str = "Arabic"):
        self.client = genai.Client(api_key=api_key)
        self.language = language
        self.prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        return f"""OCR this document page. Output the exact text in {self.language}, preserving structure.
Use Markdown formatting. Do not translate. Output text only."""

    def ocr_image(self, image: Image.Image, model_key: str) -> Tuple[str, float, float]:
        model_info = self.MODELS[model_key]
        
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        start_time = time.time()
        response = self.client.models.generate_content(
            model=model_info["name"],
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=self.prompt),
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                    ]
                )
            ]
        )
        elapsed = time.time() - start_time

        text = response.text.strip() if response.text else ""

        usage_meta = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_meta = response.usage_metadata
        
        input_tokens = getattr(usage_meta, 'prompt_token_count', None)
        output_tokens = getattr(usage_meta, 'candidates_token_count', None)
        thinking_tokens = getattr(usage_meta, 'thoughts_token_count', 0) or 0
        
        if input_tokens is None:
            image_tokens = self._estimate_image_tokens(image)
            input_tokens = image_tokens + 50
        if output_tokens is None:
            output_tokens = len(text) // 4
        
        cost = (input_tokens * model_info["input_cost_per_million"] +
                output_tokens * model_info["output_cost_per_million"]) / 1_000_000
        
        if thinking_tokens > 0 and "thinking_cost_per_million" in model_info:
            cost += (thinking_tokens * model_info["thinking_cost_per_million"]) / 1_000_000

        return text, elapsed, cost

    def _estimate_image_tokens(self, image: Image.Image) -> int:
        """Estimate token count for an image based on its dimensions.
        
        Google charges ~258 tokens for a 1024x1024 image.
        For larger images, tokens scale proportionally.
        """
        width, height = image.size
        pixels = width * height
        base_pixels = 1024 * 1024
        base_tokens = 258
        
        estimated_tokens = int((pixels / base_pixels) * base_tokens)
        return max(estimated_tokens, base_tokens)


class MistralTester:
    MODELS = {
        "mistral-ocr-latest": {
            "name": "mistral-ocr-latest",
            "cost_per_page": 0.002
        }
    }
    
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        
    def ocr_image(self, image: Image.Image, model_key: str = "mistral-ocr-latest") -> Tuple[str, float, float]:
        model_info = self.MODELS[model_key]
        img_b64 = image_to_base64(image)
        
        start_time = time.time()
        
        response = self.client.ocr.process(
            model=model_info["name"],
            document={
                "type": "image_url",
                "image_url": f"data:image/png;base64,{img_b64}"
            }
        )
        elapsed = time.time() - start_time
        
        text = ""
        if hasattr(response, 'pages') and response.pages:
            for page in response.pages:
                if hasattr(page, 'markdown'):
                    text += page.markdown + "\n"
        
        return text.strip(), elapsed, model_info["cost_per_page"]


def run_benchmark(config: BenchmarkConfig, models_to_test: List[str]) -> List[OCRResult]:
    results: List[OCRResult] = []
    
    gemini_models = [m for m in models_to_test if m.startswith("gemini")]
    mistral_models = [m for m in models_to_test if m.startswith("mistral")]
    
    gemini_tester = GeminiTester(config.gemini_api_key, config.primary_language) if gemini_models and config.gemini_api_key else None
    mistral_tester = MistralTester(config.mistral_api_key) if mistral_models and config.mistral_api_key else None
    
    sample_files = list(config.samples_dir.glob("*.pdf"))
    
    if not sample_files:
        print(f"未找到样本PDF，请将测试文件放入 {config.samples_dir}")
        return results
    
    print(f"\n找到 {len(sample_files)} 个测试样本")
    print("=" * 60)
    
    for pdf_path in sample_files:
        sample_name = pdf_path.stem
        
        gt_path = config.ground_truth_dir / f"{sample_name}.txt"
        if not gt_path.exists():
            print(f"⚠️  跳过 {sample_name}: 未找到 ground_truth/{sample_name}.txt")
            continue
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth_full = f.read().strip()
        
        print(f"\n📄 测试样本: {sample_name}")
        
        images = convert_from_path(pdf_path, dpi=config.pdf_dpi)
        num_pages = len(images)
        print(f"   共 {num_pages} 页")
        
        gt_pages = ground_truth_full.split("\n---\n") if "\n---\n" in ground_truth_full else None
        
        if gt_pages and len(gt_pages) != num_pages:
            print(f"   ⚠️ 警告: ground_truth 有 {len(gt_pages)} 页，PDF有 {num_pages} 页，页数不匹配")
        
        if not gt_pages:
            print(f"   ⚠️ 警告: ground_truth 未按页分隔 (用 --- 分隔)，将使用整文档对比")
        
        for page_num, image in enumerate(images, 1):
            page_gt = gt_pages[page_num - 1].strip() if gt_pages and page_num <= len(gt_pages) else ""
            
            if gemini_tester:
                for model_key in gemini_models:
                    try:
                        print(f"   [{model_key}] 页 {page_num}...", end=" ", flush=True)
                        text, elapsed, cost = gemini_tester.ocr_image(image, model_key)
                        
                        if page_gt:
                            cer = calculate_cer(text, page_gt)
                            wer = calculate_wer(text, page_gt)
                            print(f"CER={cer:.2%}, WER={wer:.2%}, {elapsed:.1f}s")
                        else:
                            cer, wer = 0.0, 0.0
                            print(f"完成 ({elapsed:.1f}s)")
                        
                        results.append(OCRResult(
                            model_name=model_key,
                            sample_name=sample_name,
                            page_num=page_num,
                            ocr_text=text,
                            ground_truth=page_gt,
                            cer=cer,
                            wer=wer,
                            processing_time=elapsed,
                            cost_estimate=cost
                        ))
                        time.sleep(1)
                    except Exception as e:
                        print(f"错误: {e}")
                        results.append(OCRResult(
                            model_name=model_key,
                            sample_name=sample_name,
                            page_num=page_num,
                            ocr_text="",
                            ground_truth=page_gt,
                            cer=1.0,
                            wer=1.0,
                            processing_time=0,
                            cost_estimate=0,
                            error=str(e)
                        ))
            
            if mistral_tester:
                for model_key in mistral_models:
                    try:
                        print(f"   [{model_key}] 页 {page_num}...", end=" ", flush=True)
                        text, elapsed, cost = mistral_tester.ocr_image(image, model_key)
                        
                        if page_gt:
                            cer = calculate_cer(text, page_gt)
                            wer = calculate_wer(text, page_gt)
                            print(f"CER={cer:.2%}, WER={wer:.2%}, {elapsed:.1f}s")
                        else:
                            cer, wer = 0.0, 0.0
                            print(f"完成 ({elapsed:.1f}s)")
                        
                        results.append(OCRResult(
                            model_name=model_key,
                            sample_name=sample_name,
                            page_num=page_num,
                            ocr_text=text,
                            ground_truth=page_gt,
                            cer=cer,
                            wer=wer,
                            processing_time=elapsed,
                            cost_estimate=cost
                        ))
                        time.sleep(1)
                    except Exception as e:
                        print(f"错误: {e}")
                        results.append(OCRResult(
                            model_name=model_key,
                            sample_name=sample_name,
                            page_num=page_num,
                            ocr_text="",
                            ground_truth=page_gt,
                            cer=1.0,
                            wer=1.0,
                            processing_time=0,
                            cost_estimate=0,
                            error=str(e)
                        ))
    
    return results


def generate_report(results: List[OCRResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_report_{timestamp}.md"
    
    stats: Dict[str, Dict] = defaultdict(lambda: {
        "total_cer": 0.0,
        "total_wer": 0.0,
        "total_time": 0.0,
        "total_cost": 0.0,
        "count": 0,
        "errors": 0
    })
    
    for r in results:
        s = stats[r.model_name]
        if r.error:
            s["errors"] += 1
        else:
            s["total_cer"] += r.cer
            s["total_wer"] += r.wer
            s["total_time"] += r.processing_time
            s["total_cost"] += r.cost_estimate
            s["count"] += 1
    
    report_lines = [
        "# OCR Benchmark Report",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "## Summary\n",
        "| Model | Avg CER ↓ | Avg WER ↓ | Avg Time | Est. Cost/Page | Pages | Errors |",
        "|-------|-----------|-----------|----------|----------------|-------|--------|"
    ]
    
    for model_name, s in sorted(stats.items()):
        if s["count"] > 0:
            avg_cer = s["total_cer"] / s["count"]
            avg_wer = s["total_wer"] / s["count"]
            avg_time = s["total_time"] / s["count"]
            avg_cost = s["total_cost"] / s["count"]
            report_lines.append(
                f"| {model_name} | {avg_cer:.2%} | {avg_wer:.2%} | "
                f"{avg_time:.1f}s | ${avg_cost:.4f} | {s['count']} | {s['errors']} |"
            )
    
    report_lines.extend([
        "\n## Metrics Explanation\n",
        "- **CER (Character Error Rate)**: 字符级错误率，越低越好",
        "- **WER (Word Error Rate)**: 词级错误率，越低越好",
        "- **Est. Cost**: 估算成本，基于官方定价\n",
        "## Detailed Results\n"
    ])
    
    for r in results:
        status = "❌ ERROR" if r.error else f"CER={r.cer:.2%}"
        report_lines.append(f"- **{r.model_name}** | {r.sample_name} p{r.page_num} | {status}")
    
    report_lines.extend([
        "\n## Recommendation\n"
    ])
    
    valid_stats = {k: v for k, v in stats.items() if v["count"] > 0}
    if valid_stats:
        best_model = min(valid_stats.items(), key=lambda x: x[1]["total_cer"] / x[1]["count"])
        report_lines.append(f"基于测试结果，**{best_model[0]}** 在准确率上表现最佳。")
    
    report_content = "\n".join(report_lines)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    json_path = output_dir / f"benchmark_raw_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    
    return report_path


ALL_GEMINI_MODELS = ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash"]
ALL_MISTRAL_MODELS = ["mistral-ocr-latest"]

def main():
    parser = argparse.ArgumentParser(
        description="OCR Model Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python benchmark_test.py                           # 测试所有模型
  python benchmark_test.py --models gemini-3-pro-preview gemini-3-flash-preview
  python benchmark_test.py --models mistral-ocr-latest
  python benchmark_test.py --gemini-only             # 测试所有Gemini模型
  python benchmark_test.py --mistral-only            # 测试所有Mistral模型

可用模型:
  Gemini: gemini-3-pro-preview, gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash
  Mistral: mistral-ocr-latest
        """
    )
    parser.add_argument("--models", nargs="+", help="指定要测试的模型列表")
    parser.add_argument("--gemini-only", action="store_true", help="只测试Gemini模型")
    parser.add_argument("--mistral-only", action="store_true", help="只测试Mistral模型")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    config = BenchmarkConfig.from_env(base_dir)
    
    config.samples_dir.mkdir(parents=True, exist_ok=True)
    config.ground_truth_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("OCR Model Benchmark")
    print("=" * 60)
    
    if args.models:
        models_to_test = args.models
    elif args.mistral_only:
        models_to_test = ALL_MISTRAL_MODELS
    elif args.gemini_only:
        models_to_test = ALL_GEMINI_MODELS
    else:
        models_to_test = ALL_GEMINI_MODELS + ALL_MISTRAL_MODELS
    
    gemini_models = [m for m in models_to_test if m.startswith("gemini")]
    mistral_models = [m for m in models_to_test if m.startswith("mistral")]
    
    if gemini_models and not config.gemini_api_key:
        print(f"⚠️  跳过 Gemini 模型: GEMINI_API_KEY 未配置")
        models_to_test = [m for m in models_to_test if not m.startswith("gemini")]
    
    if mistral_models and not config.mistral_api_key:
        print(f"⚠️  跳过 Mistral 模型: MISTRAL_API_KEY 未配置")
        models_to_test = [m for m in models_to_test if not m.startswith("mistral")]
    
    if not models_to_test:
        print("❌ 没有可测试的模型，请检查 .env 文件中的API密钥")
        sys.exit(1)
    
    print(f"测试模型: {', '.join(models_to_test)}")
    print(f"测试语言: {config.primary_language}")
    
    results = run_benchmark(config, models_to_test)
    
    if results:
        report_path = generate_report(results, config.results_dir)
        print("\n" + "=" * 60)
        print("测试完成!")
        print(f"报告已保存: {report_path}")
        print("=" * 60)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n" + f.read())
    else:
        print("\n未产生任何结果，请检查样本文件和配置")


if __name__ == "__main__":
    main()
