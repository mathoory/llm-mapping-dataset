import json
import random
import string
import os
import random
import re

import nltk
from nltk.corpus import gutenberg, brown, reuters, webtext
from wordfreq import top_n_list

from evaluation import Mapping, UppercaseMap, LowercaseMap, RNAMap

with open("./data/prompt.txt", 'r', encoding='utf-8') as f:
    PROMPT_TEMPLATE = f.read()

def gen_prompt(input, mapping: Mapping):
    return PROMPT_TEMPLATE.format(conversion=str(mapping), input=input)

def get_direction(task):
    if task.startswith("lowercase"):
        return "lower_to_upper"
    elif task.startswith("uppercase"):
        return "upper_to_lower"
    else:
        return None

def generate_random_string(length):
    """Generate a random lowercase string of given length."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_dataset_case_random(n, lengths=(5, 20, 50), direction="lower_to_upper"):
    """Generate dataset of case conversion mappings (lower<->upper)."""
    t = UppercaseMap() if direction == "lower_to_upper" else LowercaseMap()
    dataset = []
    difficulties = {length: diff for length, diff in zip(lengths, ["easy", "medium", "hard"])}

    for length in lengths:
        for _ in range(n):
            if direction == "lower_to_upper":
                input_str = generate_random_string(length)
                output_str = t.translate(input_str)
            else:
                input_str = generate_random_string(length).upper()
                output_str = t.translate(input_str)

            dataset.append({
                "prompt": gen_prompt(input_str, t),
                "metadata": {
                    "difficulty": difficulties[length],
                    "topic": f"{direction.replace('_', ' ')} string"
                },
                "input": input_str,
                "output": output_str
            })

    return dataset

def generate_dataset_case_words(n, lengths=(10, 200, 1000), direction="lower_to_upper"):
    """Generate dataset of case conversion mappings using real English words with spaces."""
    t = UppercaseMap() if direction == "lower_to_upper" else LowercaseMap()
    dataset = []
    difficulties = {length: diff for length, diff in zip(lengths, ["easy", "medium", "hard"])}
    WORDS = [w.lower() for w in top_n_list("en", n=50000) if w.isalpha()]

    for length in lengths:
        for _ in range(n):
            words = random.choices(WORDS, k=length)
            if direction == "lower_to_upper":
                input_str = " ".join(words)
                output_str = t.translate(input_str)
            else:
                input_str = " ".join(words).upper()
                output_str = t.translate(input_str)
            dataset.append({
                "prompt": gen_prompt(input_str, t),
                "metadata": {
                    "difficulty": difficulties[length],
                    "topic": f"{direction.replace('_', ' ')} words"
                },
                "input": input_str,
                "output": output_str
            })

    return dataset

def _ensure_corpora():
    # No try/except: just ensure these are present; no-op if already downloaded.
    for c in ["gutenberg", "brown", "reuters", "webtext"]:
        nltk.download(c, quiet=True)

def _clean_text(text: str) -> str:
    # Flatten whitespace for nicer spans.
    return re.sub(r"\s+", " ", text).strip()

def _random_span(text: str, target_chars: int, rng: random.Random) -> str:
    """
    Grab ~target_chars from text, clipping to word boundaries.
    Super simple: choose start, pick target_chars, then expand to nearest spaces.
    """
    if len(text) <= target_chars:
        return text

    # Pick a start that leaves room for target length
    start = rng.randint(0, max(0, len(text) - target_chars))
    end = start + target_chars

    # Snap to word boundaries
    left_space = text.rfind(" ", 0, start)
    right_space = text.find(" ", end)

    left = 0 if left_space == -1 else left_space + 1
    right = len(text) if right_space == -1 else right_space

    snippet = text[left:right].strip()

    # If we somehow cut too short (e.g., long token), fall back to exact slice
    if len(snippet) < target_chars // 2:
        snippet = text[start:end].strip()

    return snippet

def generate_dataset_case_text(
    n: int,
    lengths=(200, 800, 1600),
    *,
    seed=42,
    direction="lower_to_upper"
):
    """
    Generate dataset of case conversion mappings using real text spans.
    """
    _ensure_corpora()
    rng = random.Random(seed)

    sources = [
        ("webtext",  webtext.fileids(),  webtext.raw),
        ("gutenberg",gutenberg.fileids(),gutenberg.raw),
        ("reuters",  reuters.fileids(),  reuters.raw),
    ]
    sources = [(name, fids, raw) for name, fids, raw in sources if fids]
    if not sources:
        raise RuntimeError("No NLTK corpora found after download; check your NLTK setup.")

    t = UppercaseMap() if direction == "lower_to_upper" else LowercaseMap()
    dataset = []
    difficulties = {length: diff for length, diff in zip(lengths, ["easy", "medium", "hard"])}

    for length in lengths:
        for _ in range(n):
            # Try up to 10 times to get a valid snippet
            for attempt in range(10):
                corpus_name, fids, raw = rng.choice(sources)
                fid = rng.choice(fids)
                text = _clean_text(raw(fid))
                if not text:
                    continue
                snippet = _random_span(text, target_chars=length, rng=rng)
                # If any non-ASCII alphabetic character is present, retry
                if any((not c.isascii() and c.isalpha()) for c in snippet):
                    continue
                break
            else:
                # If no valid snippet found after 10 tries, skip
                continue

            if direction == "lower_to_upper":
                input_str = snippet.lower()
                output_str = t.translate(input_str)
            else:
                input_str = snippet.upper()
                output_str = t.translate(input_str)
            dataset.append({
                "prompt": gen_prompt(input_str, t),
                "metadata": {
                    "difficulty": difficulties[length],
                    "topic": f"{direction.replace('_', ' ')} natural text",
                    "source_corpus": corpus_name,
                    "source_fileid": fid,
                    "target_chars": length,
                    "span_len": len(snippet),
                },
                "input": input_str,
                "output": output_str,
            })

    return dataset


def generate_dataset_rna_random(n, lengths=(5, 20, 50)):
    """Generate dataset of random RNA sequences to protein sequences."""
    t = RNAMap()
    dataset = []
    difficulties = {length: diff for length, diff in zip(lengths, ["easy", "medium", "hard"])}

    for length in lengths:
        for _ in range(n):
            RNA_sequence = ''.join(random.choices('ACGU', k=length*3))
            protein_sequence = t.translate(RNA_sequence)
            dataset.append({
                "prompt": gen_prompt(RNA_sequence, t),
                "metadata": {
                    "difficulty": difficulties[length],
                    "topic": "RNA"
                },
                "input": RNA_sequence,
                "output": protein_sequence
            })

    return dataset

def save_to_jsonl(data, filename):
    """Save dataset to JSONL file."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def generate_data(tasks, size, output_path):
    if isinstance(tasks, str):
        tasks = [tasks]
    all_data = []
    for task in tasks:
        direction = get_direction(task)
        if task in ["lowercase", "uppercase"]:
            all_data.extend(generate_dataset_case_random(size, direction=direction or "lower_to_upper"))
        elif task in ["lowercase_words", "uppercase_words"]:
            all_data.extend(generate_dataset_case_words(size, direction=direction or "lower_to_upper"))
        elif task in ["lowercase_text", "uppercase_text"]:
            all_data.extend(generate_dataset_case_text(size, direction=direction or "lower_to_upper"))
        elif task == "rna":
            all_data.extend(generate_dataset_rna_random(size))
        else:
            raise ValueError(f"Unknown task: {task}")
    return save_to_jsonl(all_data, output_path)

ALL_TASK_CHOICES = [
    'lowercase', 'lowercase_words', 'lowercase_text',
    'uppercase', 'uppercase_words', 'uppercase_text',
    'rna'
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LLM Mapping Dataset")
    parser.add_argument('--tasks', type=str, nargs='+', choices=ALL_TASK_CHOICES, default=ALL_TASK_CHOICES, help='Task types (space separated)')
    parser.add_argument('--size', type=int, default=40, help='Number of examples to generate per task, will be multiplied by number of difficulties (3) per subtask')
    parser.add_argument('--output', type=str, default='./data/examples.jsonl', help='Output path for generated data')

    args = parser.parse_args()

    generate_data(args.tasks, args.size, args.output)