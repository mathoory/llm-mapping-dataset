# Generate the dataset

import json
import random
import string
from pathlib import Path
import random
from wordfreq import top_n_list
from evaluation import UppercaseMap

def generate_random_string(length):
    """Generate a random lowercase string of given length."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_dataset_lower_random(n, lengths=(5, 20, 50)):
    """Generate dataset of lowercase:uppercase mappings."""
    t = UppercaseMap()
    dataset = []
    difficulties = {length: diff for length, diff in zip(lengths, ["easy", "medium", "hard"])}

    for length in lengths:
        for _ in range(n):
            lowercase_str = generate_random_string(length)
            uppercase_str = t.translate(lowercase_str)
            dataset.append({
                "prompt": f"Convert the following string to uppercase, give the answer between curly brackets:\n{lowercase_str}",
                "difficulty": difficulties[length],
                "input": lowercase_str,
                "output": uppercase_str
            })

    return dataset

def generate_dataset_lower_words(n, lengths=(10, 200, 1000)):
    """Generate dataset of lowercase:uppercase mappings using real English words with spaces."""
    WORDS = [w.lower() for w in top_n_list("en", n=50000) if w.isalpha()]
    difficulties = {length: diff for length, diff in zip(lengths, ["easy", "medium", "hard"])}

    dataset = []
    for length in lengths:
        for _ in range(n):
            # Pick `length` words and join with spaces
            words = random.choices(WORDS, k=length)
            lowercase_str = " ".join(words)
            uppercase_str = lowercase_str.upper()
            dataset.append({
                "prompt": f"Convert the following string to uppercase, give the answer between curly brackets:\n{lowercase_str}",
                "difficulty": difficulties[length],
                "input": lowercase_str,
                "output": uppercase_str
            })

    return dataset

def save_to_jsonl(data, filename):
    """Save dataset to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Generate and save dataset
dataset = generate_dataset_lower_words(3)
output_path = Path("./data/examples.jsonl")
save_to_jsonl(dataset, output_path)