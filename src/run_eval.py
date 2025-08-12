# Run evaluation on the model using the dataset
import json

from evaluation import Mapping, UppercaseMap
from utils.llm import LLM
import re

def run_eval(mapping: Mapping):
    llm = LLM()

    with open("./data/examples.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)

            prompt = example["prompt"]
            response = llm.query(prompt)
            
            # Extract output from single curly brackets in the response
            match = re.search(r"\{([^{}]+)\}", response)
            output = match.group(1).strip() if match else ""
            res = mapping.evaluate(example["input"], output)
            print(res)

run_eval(UppercaseMap())