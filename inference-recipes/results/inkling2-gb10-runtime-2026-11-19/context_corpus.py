"""Generate the six text-only Inkling retrieval prompts using the pinned tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer


def inventory_prompt(records, count, depth):
    index = int((count - 1) * depth)
    key, expected = records[index].split(" = ")
    prompt = "Use this inventory.\n" + "\n".join(records[:count])
    return prompt + f"\nWhat number belongs to {key}? Reply with that number only.", expected


def cases(tokenizer):
    rng = random.Random(20261119)
    records = [f"bin-{i:04d} = {rng.randrange(100000, 999999)}" for i in range(640)]
    for target in (512, 2048, 3584):
        for depth in (0.12, 0.84):
            low, high = 1, len(records)
            while low < high:
                middle = (low + high + 1) // 2
                prompt, _ = inventory_prompt(records, middle, depth)
                if len(tokenizer.encode(prompt, add_special_tokens=False).ids) <= target:
                    low = middle
                else:
                    high = middle - 1
            prompt, expected = inventory_prompt(records, low, depth)
            yield dict(target=target, depth=depth, records=low, prompt=prompt, expected=expected,
                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("tokenizer", type=Path, help="tokenizer.json from the pinned Inkling checkpoint")
    args = cli.parse_args()
    for item in cases(Tokenizer.from_file(str(args.tokenizer))):
        print(json.dumps(item))
