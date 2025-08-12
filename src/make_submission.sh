#!/usr/bin/env bash
# Usage: bash make_submission.sh

IDS="319003711_207031311"
ZIP_NAME="nlp_final_project_${IDS}.zip"

# Remove previous archive if exists
rm -f "${ZIP_NAME}"

# Create new zip with selected files
zip -r "${ZIP_NAME}" \
  data/examples.jsonl \
  evaluation.py \
  generate_data.py \
  run_eval.py \
  utils/*.py \
  README.md \
  "nlp_final_project_${IDS}.pdf"