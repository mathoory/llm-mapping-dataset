# Run evaluation on the model using the dataset
import json
import argparse
from evaluation import topic_to_mapping
from utils.llm import LLM
from datetime import datetime


def run_eval(model_name, data_path):
    llm = LLM(model=model_name)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            extracted_output, confidence_score = llm.query(example["prompt"], parse=True)
            topic = example["metadata"]["topic"]
            mapping = topic_to_mapping[topic]
            res = mapping.evaluate(example["input"], extracted_output)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{confidence_score}] {res}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="LLM Mapping Dataset Runner")
	parser.add_argument('--model', type=str, default='flash', help='Model name for evaluation')
	parser.add_argument('--data', type=str, default='./data/examples.jsonl', help='Output path for generated data')

	args = parser.parse_args()

	run_eval(args.model, args.data)
