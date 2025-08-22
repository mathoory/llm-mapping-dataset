import logging
def setup_logging(level):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format='[%(levelname)s] %(message)s')
# Run evaluation on the model using the dataset
import json
import argparse
from evaluation import topic_to_mapping
from utils.llm import LLM
from datetime import datetime



def run_eval(model_name, data_path):
    llm = LLM(model=model_name)
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))

    prompts = [ex["prompt"] for ex in examples]
    outputs = llm.query_batch(prompts, parse=True)

    for example, (extracted_output, confidence_score) in zip(examples, outputs):
        topic = example["metadata"]["topic"]
        mapping = topic_to_mapping[topic]
        res = mapping.evaluate(example["input"], extracted_output)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{confidence_score}] {res}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Mapping Dataset Runner")
    parser.add_argument('--model', type=str, default='flash', help='Model name for evaluation')
    parser.add_argument('--data', type=str, default='./data/examples.jsonl', help='Output path for generated data')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (e.g., INFO, DEBUG, WARNING)')

    args = parser.parse_args()
    setup_logging(args.log_level)

    run_eval(args.model, args.data)
