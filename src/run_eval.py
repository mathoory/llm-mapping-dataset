# Run evaluation on the model using the dataset
import json

from evaluation import topic_to_mapping
from utils.llm import LLM
import re
from datetime import datetime

def run_eval():
    llm = LLM()

    with open("./data/examples.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)

            prompt = example["prompt"]
            response = llm.query(prompt)
            
            # Extract output from single curly brackets in the response
            match = re.search(r"\{([^{}]+)\}", response)
            extracted_output = match.group(1).strip() if match else ""

            # Get mapping
            topic = example["metadata"]["topic"]
            mapping = topic_to_mapping[topic]

            # Evaluate (expected vs actual)
            res = mapping.evaluate(example["input"], extracted_output)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {res}")

run_eval()