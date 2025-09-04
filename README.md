# LLM Mapping Dataset

## Project Report

[Download the project report (PDF)](report.pdf)

## 1. Project Overview
This project provides a framework for generating and evaluating datasets for natural language processing (NLP) mapping tasks using large language models (LLMs). It supports tasks such as string case conversion and RNA-to-protein translation, and is designed for extensibility and reproducibility.

## Supported Mapping Tasks
The following tasks are available for dataset generation and evaluation:

- `lowercase` — lowercase string to uppercase string
- `lowercase_words` — lowercase words to uppercase words
- `lowercase_text` — lowercase natural text to uppercase
- `uppercase` — uppercase string to lowercase string
- `uppercase_words` — uppercase words to lowercase words
- `uppercase_text` — uppercase natural text to lowercase
- `rna` — RNA sequence to protein sequence
- `country_code` — country codes to country names
- `digits` — integer digits (1-9) to English words
- `numbers` — English number words to digits

Specify any combination of these tasks using the `--tasks` argument.

## 2. Project Architecture 🗂️
- **src/**: Main source code directory
  - `generate_data.py`: Generates datasets for various mapping tasks and saves them in JSONL format.
  - `run_eval.py`: Evaluates LLMs on the generated datasets and prints results.
  - `utils/llm.py`: Wrapper for interacting with Google GenAI models.
  - `utils/`: Contains utility modules for mapping and string comparison.
  - `evaluation.py`: Defines mapping classes and evaluation logic.
  - `data/`: Contains prompt templates and generated datasets.
- **requirements.txt**: Minimal required Python packages for running the project.
- **environment.yaml**: Conda environment specification (optional, for full-featured development).
- **run_subtask.py**: Unified CLI for generating data and running evaluation.
- **data/**: Main dataset directory


## 3. Running Instructions 🏃‍♂️

### Environment Setup
Install dependencies with pip:

```bash
pip install -r requirements.txt
```

Or create the full environment with conda:

```bash
conda env create -f environment.yaml
conda activate nlp-final
```

### Data Generation
Generate datasets for all tasks (default):

```bash
python src/generate_data.py
```

Or specify tasks and size:

```bash
python src/generate_data.py --tasks lowercase rna --size 50 --output ./src/data/examples.jsonl
```

### Evaluation

Run evaluation on a generated dataset:

```bash
python src/run_eval.py
```

Or use the unified CLI:

```bash
python run_subtask.py --model pro --size 100 --tasks lowercase rna --eval
```

#### Saving Outputs

To save evaluation results and logs to the `runs` directory, use the `--save_outputs` flag:

```bash
python run_eval.py --model flash --data ./data/examples.jsonl --save_outputs
```

Or from `run_subtask.py`:

```bash
python run_subtask.py --model flash --size 100 --tasks task1 task2 --save_outputs
```

By default, outputs are not saved unless `--save_outputs` is specified.

## Sample Dataset Entry
Each dataset entry is a JSON object with the following fields:

```json
{
  "prompt": "Please convert the following English number words to their corresponding digits (1-9) separated by spaces: {nine two six two three nine five eight seven seven}\nPlease provide your output and a confidence score between 0% to 100% in the following JSON format:\n{\n\"answer\": \"Your answer here\",\n\"confidence_score\": number\n}",
  "metadata": {"difficulty": "easy", "topic": "numbers"},
  "input": "nine two six two three nine five eight seven seven",
  "output": "9 2 6 2 3 9 5 8 7 7"
}
```


### Notes
- 🔑 You must provide a valid API key for Google GenAI in `src/key.secret`.
- 📄 Generated datasets are saved in JSONL format in the specified output path.
- 📝 Evaluation prints results with timestamps and confidence scores.
- ⚡ **Model-specific behavior:**
  - For `flash` (Gemini 2.5 Flash), thinking is disabled (`thinking_budget=0`).
  - For `pro` (Gemini 2.5 Pro), minimum thinking is enabled (`thinking_budget=128`).

#### API Key Format
The file `src/key.secret` should contain only your Google GenAI API key string (no quotes, no extra lines).

## 4. Extending the project
1. **Create a new mapping class** in `src/evaluation.py` that inherits from `Mapping` and implements the required logic.
2. **Add your class to `topic_to_mapping`** in `evaluation.py` with a unique topic string.
3. **Add a dataset generator function** in `src/generate_data.py` for your new task.
4. **Update `ALL_TASK_CHOICES`** in `generate_data.py` to include your new task name.
5. (Optional) Add a new model alias in `src/utils/llm.py` if needed.

---

## runs/ Directory
Evaluation results and logs are saved in the `src/runs/` directory:
- `results_YYYYMMDD_HHMMSS.json`: Contains the evaluation results for each example.
- `run_log_YYYYMMDD_HHMMSS.txt`: Contains a log of the evaluation process, including errors and confidence scores.


## 5. Analyzing Results

To analyze evaluation results and generate error-rate tables, use the analysis script:

```bash
python src/utils/analyze.py src/runs --out_dir src/reports
```

This will:
- Merge all JSON result files in the specified results directory (e.g., `src/runs`).
- Compute error rates (100 - accuracy) for each model, topic, and difficulty.
- Print a styled summary table to the terminal (bold = max error per row, underline = min average per topic).
- Save CSV and Excel files with the results in the output directory (default: `src/reports`).

### Options
- `--out_dir`: Output directory for reports (default: `src/reports`).
- `--decimals`: Decimal places for error rates (default: 4).
- `--zero_tol`: Treat errors ≤ tol as zero (default: 1e-12).

Example with custom options:
```bash
python src/utils/analyze.py src/runs --out_dir src/reports --decimals 2 --zero_tol 1e-6
```

**Output files:**
- `maptic_table_error_rates.csv`: Machine-friendly error rates table.
- `maptic_table_error_rates.xlsx`: Excel file with styled error rates table.

For further details, see comments in the source files.
