#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified evaluation script: eval.py
=====================

Features:
- Generate responses from the target model based on image-text payloads (text + image)
- Score (original_query, response) with the StrongREJECT evaluator
- Automatically read original_query from full_pipeline_result.json in each query_* directory
- Save evaluation results as JSON (with aggregate metrics and detailed scores for each instance)

Usage examples:
    python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\custom" \
        --target_model openai/gpt-4o-mini \
        --judge_model openai/gpt-4o-mini

python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\custom" --target_model openai/gpt-4o-mini --judge_model openai/gpt-4o-mini

python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\safebench_v6" --target_model gemini-2.5-flash-preview-09-2025 --judge_model MiniMax-M2


python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\safebench_v7" --target_model openai/gpt-4.1-mini --judge_model Qwen3-8B

python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\safebench_v7" --target_model openai/gpt-4.1 --judge_model Qwen3-8B

python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\safebench_v7" --target_model anthropic/claude-haiku-4.5 --judge_model Qwen3-8B

python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\test4" --target_model siliconcloud/Qwen/Qwen3-VL-235B-A22B-Instruct --judge_model siliconcloud/MiniMaxAI/MiniMax-M2


python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\test4" --target_model openai/gpt-4o --judge_model deepseek-chat


python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\test4" --target_model anthropic/claude-haiku-4.5 --judge_model anthropic/claude-haiku-4.5   --retry_attempts  1

python eval.py --dataset "e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\test4" --target_model openai/gpt-5 --judge_model deepseek-chat   --retry_attempts  1

python eval.py --dataset "e:\projects\large-models\reinforcement-learning\COMET\results\safebench_v7" --target_model openai/gpt-4.1-mini --judge_model Qwen3-8B   --retry_attempts  1  --prefer_ablation_text


python eval.py --dataset "e:\projects\large-models\reinforcement-learning\COMET\results\safebench_v7" --target_model anthropic/claude-haiku-4.5 --judge_model Qwen3-8B   --retry_attempts  1

Optional arguments:
- --output     Specify the Output JSON path (by default generated automatically in the dataset directory)
- --limit      Limit the number of instances to evaluate (e.g. run only 1)

Prerequisites:
- Configure API_KEYS / API_URLS in RIVAL/config.py or environment variables (openai/anthropic/siliconcloud/openrouter)
- The target model must support image-text input (e.g. openai/gpt-4o or openai/gpt-4o-mini)
"""

import os
import sys
import argparse

# Ensure the project root is on sys.path so the process_utils package can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from process_utils.strongreject_payload_eval import run as payload_run
except Exception as e:
    raise ImportError(f"Failed to import evaluation module: {e}. Please make sure process_utils/strongreject_payload_eval.py exists and has no syntax errors.")


def main():
    parser = argparse.ArgumentParser(description="StrongREJECT evaluation entry script for image-text payloads")
    parser.add_argument("--dataset", required=True, help="Dataset directory, e.g. e:\\projects\\large-models\\reinforcement-learning\\COMET\\results\\custom")
    parser.add_argument("--target_model", default="openai/gpt-4o-mini", help="Target model (image-text), e.g. openai/gpt-4o or openai/gpt-4o-mini")
    parser.add_argument("--judge_model", default="openai/gpt-4o-mini", help="Judge model (StrongREJECT scoring)")
    parser.add_argument("--output", default=None, help="Output JSON path; if omitted, generated automatically in the dataset directory")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of instances to evaluate (optional)")
    parser.add_argument("--retry", action="store_true", default=True, help="Enable refusal retry: if the evaluation result is a refusal, retry generation and judging (enabled by default)")
    parser.add_argument("--retry_attempts", type=int, default=3, help="Number of extra retries on refusal (default 3)")
    parser.add_argument("--retry_backoff_sec", type=float, default=0.0, help="Seconds to wait before each retry (default 0)")
    parser.add_argument("--prefer_ablation_text", action="store_true", default=False, help="Prefer ablation_attack_text.txt as the evaluation text (disabled by default)")

    args = parser.parse_args()

    # Basic checks
    if not os.path.isdir(args.dataset):
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset}")

    out_path = payload_run(
        dataset_dir=args.dataset,
        target_model=args.target_model,
        judge_model=args.judge_model,
        output_path=args.output,
        limit=args.limit,
        retry_on_refusal=args.retry,
        retry_attempts=args.retry_attempts,
        retry_backoff_sec=args.retry_backoff_sec,
        prefer_ablation_text=args.prefer_ablation_text,
    )

    print(f"StrongREJECT evaluation completed, results saved to: {out_path}")


if __name__ == "__main__":
    main()