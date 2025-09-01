# Run evaluation on the model using the dataset

import json
import argparse
import os
from evaluation import topic_to_mapping
from utils.llm import LLM
from datetime import datetime


def load_examples(data_path, size=None):
    """Load and subsample examples from a jsonl file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    total_lines = len(lines)
    if size is not None and size > 0 and size < total_lines:
        skip = max(total_lines // size, 1)
        selected_lines = [lines[i] for i in range(0, total_lines, skip)][:size]
    else:
        selected_lines = lines
    examples = [json.loads(line) for line in selected_lines]
    return examples


def save_outputs_and_logs(data_path, results_json, log_lines, model_name):
    """Save results and logs to files with timestamped filenames."""
    base = os.path.basename(data_path)
    # Expecting format: examples_YYYYMMDD_HHMMSS.jsonl
    import re
    m = re.search(r'_(\d{8}_\d{6})', base)
    if m:
        timestamp = m.group(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filename
    safe_model = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
    out_base = f"runs/results_{safe_model}_{timestamp}.json"
    log_base = f"runs/run_log_{safe_model}_{timestamp}.txt"
    out_filename = out_base
    log_filename = log_base
    suffix = 1
    while os.path.exists(out_filename) or os.path.exists(log_filename):
        out_filename = f"runs/results_{safe_model}_{timestamp}_{suffix}.json"
        log_filename = f"runs/run_log_{safe_model}_{timestamp}_{suffix}.txt"
        suffix += 1
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    with open(log_filename, "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
    print(f"Results written to {out_filename}")
    print(f"Log written to {log_filename}")


def run_eval(model_name, data_path, save_outputs=False, verbose=False, size=None, tasks=None):
    from utils.llm import LLM
    if verbose:
        LLM.set_log_level("DEBUG")
    llm = LLM(model=model_name)
    results_json = []
    log_lines = []
    # Load and subsample examples
    try:
        # load jsonl
        examples = load_examples(data_path, size)
        # Filter by tasks/topics if provided
        if tasks:
            task_set = set(tasks)
            examples = [ex for ex in examples if ex['metadata']['topic'] in task_set]
        prompts = [ex["prompt"] for ex in examples]

        # query LLM
        outputs_iter = llm.query_batch(prompts, parse=True)
        
        # evaluate responses
        for example, output_dict in zip(examples, outputs_iter):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # unpack llm response
            extracted_output = output_dict.get("output", "")
            confidence_score = output_dict.get("confidence", None)
            error = output_dict.get("error", None)

            # unpack prompt metadata
            topic = example["metadata"].get("topic", "")
            difficulty = example["metadata"].get("difficulty", "")

            # evaluate response
            res = topic_to_mapping[topic].evaluate(example["input"], extracted_output)

            # logging
            log_line = f"[{timestamp}]{' [error: {error}] ' if error else ' '}[{topic}] [{difficulty}] [{confidence_score}] {res}"
            if verbose:
                print(log_line)
            log_lines.append(log_line)

            # prepare result entry
            result_dict = {
                "INPUT": example.get("input", ""),
                "OUTPUT": extracted_output,
                "EXP": example.get("output", None),
                "ACC": 100-res.pct_mistakes,
                "CONF": confidence_score,
                "ERRORS": {
                    "SUBSTITUTIONS": res.substitutions,
                    "INSERTIONS": res.insertions,
                    "DELETIONS": res.deletions
                },
                "MODEL": model_name,
                "TOPIC": topic,
                "DIFFICULTY": difficulty,
                "ERR": error
            }
            results_json.append(result_dict)
    finally:
        if save_outputs:
            save_outputs_and_logs(data_path, results_json, log_lines, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Mapping Dataset Runner")
    parser.add_argument('--model', type=str, default='flash', help='Model name for evaluation')
    parser.add_argument('--dataset', type=str, default='./data/examples.jsonl', help='Path for generated data')
    parser.add_argument('--save_outputs', action='store_true', default=True, help='Save results and log files in runs directory (default: True)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--size', type=int, default=None, help='Subsample size: number of examples to use from the dataset (evenly spaced)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None, help='Space separated list of topics to filter for evaluation')

    args = parser.parse_args()

    tasks = args.tasks if args.tasks else None

    if args.verbose:
        LLM.set_log_level("DEBUG")
    run_eval(args.model, args.dataset, save_outputs=args.save_outputs, verbose=args.verbose, size=args.size, tasks=tasks)
