# Run evaluation on the model using the dataset
import json
import argparse
from evaluation import topic_to_mapping
from utils.llm import LLM
from datetime import datetime



def run_eval(model_name, data_path, save_outputs=False):
    llm = LLM(model=model_name)
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))

    prompts = [ex["prompt"] for ex in examples]
    outputs = llm.query_batch(prompts, parse=True)

    results_json = []
    log_lines = []
    for example, output_dict in zip(examples, outputs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # unpack output_dict
        extracted_output = output_dict.get("output", "")
        confidence_score = output_dict.get("confidence", None)
        error = output_dict.get("error", None)

        # unpack example metadata
        topic = example["metadata"].get("topic", "")
        difficulty = example["metadata"].get("difficulty", "")

        # evaluate response
        res = topic_to_mapping[topic].evaluate(example["input"], extracted_output)

        log_line = f"[{timestamp}]{' [error: {error}] ' if error else ' '}[{topic}] [{difficulty}] [{confidence_score}] {res}"
        print(log_line)
        log_lines.append(log_line)

        result_dict = {
            "INPUT": example.get("input", ""),
            "OUTPUT": extracted_output,
            "EXP": example.get("output", None),
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

    if save_outputs:
        # Use the timestamp from the dataset filename
        import os
        base = os.path.basename(data_path)
        # Expecting format: examples_YYYYMMDD_HHMMSS.jsonl
        import re
        m = re.search(r'_(\d{8}_\d{6})', base)
        if m:
            timestamp = m.group(1)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"runs/results_{timestamp}.json"
        log_filename = f"runs/run_log_{timestamp}.txt"
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)
        with open(log_filename, "w", encoding="utf-8") as f:
            for line in log_lines:
                f.write(line + "\n")
        print(f"Results written to {out_filename}")
        print(f"Log written to {log_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Mapping Dataset Runner")
    parser.add_argument('--model', type=str, default='flash', help='Model name for evaluation')
    parser.add_argument('--dataset', type=str, default='./data/examples.jsonl', help='Path for generated data')
    parser.add_argument('--save_outputs', action='store_true', default=False, help='Save results and log files in runs directory (default: False)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.verbose:
        LLM.set_log_level("DEBUG")
    run_eval(args.model, args.dataset, save_outputs=args.save_outputs)
