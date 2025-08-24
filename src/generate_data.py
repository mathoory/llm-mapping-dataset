import json
import random
import string
import os
import random
import re

import nltk
from nltk.corpus import gutenberg, brown, reuters, webtext
from wordfreq import top_n_list

from evaluation import Mapping, UppercaseMap, RNAMap

with open("./data/prompt.txt", 'r', encoding='utf-8') as f:
    PROMPT_TEMPLATE = f.read()

def gen_prompt(input, mapping: Mapping):
    return PROMPT_TEMPLATE.format(conversion=str(mapping), input=input)

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
                "prompt": gen_prompt(lowercase_str, t),
                "metadata": {
                    "difficulty": difficulties[length],
                    "topic": "uppercase string"
                },
                "input": lowercase_str,
                "output": uppercase_str
            })

    return dataset

def generate_dataset_lower_words(n, lengths=(10, 200, 1000)):
    """Generate dataset of lowercase:uppercase mappings using real English words with spaces."""
    t = UppercaseMap()
    dataset = []
    difficulties = {length: diff for length, diff in zip(lengths, ["easy", "medium", "hard"])}
    WORDS = [w.lower() for w in top_n_list("en", n=50000) if w.isalpha()]

    for length in lengths:
        for _ in range(n):
            # Pick `length` words and join with spaces
            words = random.choices(WORDS, k=length)
            lowercase_str = " ".join(words)
            uppercase_str = t.translate(lowercase_str)
            dataset.append({
                "prompt": gen_prompt(lowercase_str, t),
                "metadata": {
                    "difficulty": difficulties[length],
                    "topic": "uppercase words"
                },
                "input": lowercase_str,
                "output": uppercase_str
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

def generate_dataset_lower_text(
    n: int,
    lengths=(200, 800, 1600),
    *,
    seed=42
):
    """
    Generate dataset of lowercase:uppercase mappings using real text spans.

    - Samples random contiguous character spans from NLTK corpora
      (webtext, brown, gutenberg, reuters).
    - Lowercases the input snippet; output is UppercaseMap().translate(lower).
    - Scales to any n.

    Args:
        n: number of samples per length bucket
        lengths: approximate character targets for difficulty buckets
        seed: optional RNG seed

    Returns:
        list[dict]: each with prompt, metadata, input, output
    """
    _ensure_corpora()
    rng = random.Random(seed)

    # Build a pool of (corpus_name, fileids, corpus_obj.raw getter)
    sources = [
        ("webtext",  webtext.fileids(),  webtext.raw),
        ("gutenberg",gutenberg.fileids(),gutenberg.raw),
        ("reuters",  reuters.fileids(),  reuters.raw),
    ]

    # Clean empty pools (just in case some corpora are unavailable locally)
    sources = [(name, fids, raw) for name, fids, raw in sources if fids]

    if not sources:
        raise RuntimeError("No NLTK corpora found after download; check your NLTK setup.")

    t = UppercaseMap()
    dataset = []
    difficulties = {length: diff for length, diff in zip(lengths, ["easy", "medium", "hard"])}

    for length in lengths:
        for _ in range(n):
            # Pick a random doc from a random corpus
            corpus_name, fids, raw = rng.choice(sources)
            fid = rng.choice(fids)

            text = _clean_text(raw(fid))
            if not text:
                continue  # try again next loop

            snippet = _random_span(text, target_chars=length, rng=rng)
            lowercase_str = snippet.lower()
            uppercase_str = t.translate(lowercase_str)

            dataset.append({
                "prompt": gen_prompt(lowercase_str, t),
                "metadata": {
                    "difficulty": difficulties[length],
                    "topic": "uppercase natural text",
                    "source_corpus": corpus_name,
                    "source_fileid": fid,
                    "target_chars": length,
                    "span_len": len(snippet),
                },
                "input": lowercase_str,
                "output": uppercase_str,
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
    from datetime import datetime
    base, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_ts = f"{base}_{timestamp}{ext}"
    with open(filename_ts, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')



def generate_data(tasks, size, output_path):
    if isinstance(tasks, str):
        tasks = [tasks]
    all_data = []
    for task in tasks:
        if task == "lowercase":
            all_data.extend(generate_dataset_lower_random(size))
        elif task == "lowercase_words":
            all_data.extend(generate_dataset_lower_words(size))
        elif task == "lowercase_text":
            all_data.extend(generate_dataset_lower_text(size))
        elif task == "rna":
            all_data.extend(generate_dataset_rna_random(size))
        else:
            raise ValueError(f"Unknown task: {task}")
    save_to_jsonl(all_data, output_path)

ALL_TASK_CHOICES = ['lowercase', 'lowercase_words', 'lowercase_text', 'rna']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LLM Mapping Dataset")
    parser.add_argument('--tasks', type=str, nargs='+', choices=ALL_TASK_CHOICES, default=ALL_TASK_CHOICES, help='Task types (space separated)')
    parser.add_argument('--size', type=int, default=40, help='Number of examples to generate per task, will be multiplied by number of difficulties (3) per subtask')
    parser.add_argument('--output', type=str, default='./data/examples.jsonl', help='Output path for generated data')

    args = parser.parse_args()

    generate_data(args.tasks, args.size, args.output)