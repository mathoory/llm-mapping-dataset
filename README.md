
# LLM Mapping Dataset 🚀

## 1. Project Overview
This project provides a framework for generating and evaluating datasets for natural language processing (NLP) mapping tasks using large language models (LLMs). It supports tasks such as string case conversion and RNA-to-protein translation, and is designed for extensibility and reproducibility.

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


### Notes
- 🔑 You must provide a valid API key for Google GenAI in `src/key.secret`.
- 📄 Generated datasets are saved in JSONL format in the specified output path.
- 📝 Evaluation prints results with timestamps and confidence scores.
- ⚡ **Model-specific behavior:**
  - For `flash` (Gemini 2.5 Flash), thinking is disabled (`thinking_budget=0`).
  - For `pro` (Gemini 2.5 Pro), minimum thinking is enabled (`thinking_budget=128`).

## 4. Extending the Project
- ➕ Add new mapping tasks by implementing new classes in `evaluation.py` and updating `generate_data.py`.
- 🛠️ Add new model aliases in `utils/llm.py` as needed.

---
For further details, see comments in the source files.
