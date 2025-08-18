# Run evaluation on the model using the dataset
import json

from evaluation import topic_to_mapping
from utils.llm import LLM
import re
from datetime import datetime

def run_eval(llm):
    with open("./data/examples.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)

            extracted_output, confidence_score = llm.query(example["prompt"], parse=True)
            
            topic = example["metadata"]["topic"]
            mapping = topic_to_mapping[topic]

            # Evaluate (expected vs actual)
            res = mapping.evaluate(example["input"], extracted_output)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{confidence_score}] {res}")


if __name__ == "__main__":
    llm = LLM()
    run_eval(llm)