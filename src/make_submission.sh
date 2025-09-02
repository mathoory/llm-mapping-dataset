#!/usr/bin/env bash
# Usage: bash make_submission.sh

IDS="123456789_123456789"
ZIP_NAME="nlp_final_project_${IDS}.zip"

# Remove previous archive if exists
rm -f "${ZIP_NAME}"

# Create new zip with selected files
zip -r "${ZIP_NAME}" \
  data/examples.jsonl \
  data/prompt.txt \
  evaluation.py \
  generate_data.py \
  run_eval.py \
  run_subtask.py \
  utils/*.py \
  README.md \
  requirements.txt \
  environment.yaml \
  "nlp_final_project_${IDS}.pdf"