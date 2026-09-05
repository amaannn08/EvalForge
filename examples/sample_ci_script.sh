#!/usr/bin/env bash
# Sample CI execution script using EvalForge CLI for automated evaluation and PR gating
set -eo pipefail

echo "=========================================================="
echo " EvalForge Automated CI Evaluation & Quality Gating "
echo "=========================================================="

# 1. Run baseline evaluation
echo "[CI] Running baseline evaluation on qa_benchmark.json..."
evalforge run \
  --dataset examples/benchmarks/qa_benchmark.json \
  --name "baseline_v1" \
  --model "model-stable" \
  --output "baseline_report.json"

# 2. Run candidate evaluation
echo "[CI] Running candidate evaluation on qa_benchmark.json..."
evalforge run \
  --dataset examples/benchmarks/qa_benchmark.json \
  --name "candidate_v2" \
  --model "model-candidate" \
  --output "candidate_report.json"

# 3. Evaluate regression gate
echo "[CI] Evaluating statistical regression gate..."
evalforge gate \
  --baseline baseline_report.json \
  --candidate candidate_report.json \
  --alpha 0.05 \
  --threshold 0.02 \
  --markdown "pr_comment.md"

echo "[CI] Gating successfully passed! Ready for deployment."
