# /// script
# requires-python = ">=3.9"
# dependencies = ["jinja2==3.1.6", "tokenizers==0.22.1", "transformers==4.57.3"]
# ///
"""Locate single-letter answers in each checkpoint's rendered conversation."""

import json
from pathlib import Path
from urllib.request import urlopen

from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import render_jinja_template

from audit import REVISIONS, model_url


def trace(model, record):
    messages = record["body"]["messages"]
    kwargs = record["body"]["chat_template_kwargs"]
    with urlopen(model_url(model, "chat_template.jinja"), timeout=45) as response:
        template = response.read().decode()
    with urlopen(model_url(model, "tokenizer.json"), timeout=45) as response:
        tokenizer = Tokenizer.from_str(response.read().decode())
    prompts, _ = render_jinja_template([messages], chat_template=template,
                                       add_generation_prompt=True, **kwargs)
    if model == "Inkling" and prompts[0] != record["rendered"]["prompt"]:
        raise ValueError("Rendered prompt differs from the archived Spark fixture")
    prompt_ids = tokenizer.encode(prompts[0], add_special_tokens=False).ids
    letters = "ABCDEFGHIJ"
    conversations = [messages + [{"role": "assistant", "content": letter}] for letter in letters]
    rendered, _ = render_jinja_template(conversations, chat_template=template,
                                        add_generation_prompt=False, **kwargs)
    suffixes = {}
    for letter, conversation in zip(letters, rendered):
        ids = tokenizer.encode(conversation, add_special_tokens=False).ids
        if ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Prompt prefix changed for {model}/{letter}")
        suffixes[letter] = ids[len(prompt_ids):]
    return {"generation_prompt_last_token_id": prompt_ids[-1],
            "serialized_answer_suffix_ids": suffixes}


def main():
    fixture_path = (Path(__file__).resolve().parents[2] / "inference-recipes/experiments"
                    / "inkling-original/template-check.json")
    fixture = json.loads(fixture_path.read_text())
    record = next(r for r in fixture["tests"]
                  if r["server"] == "reasoning-off-effort-zero" and r["request"] == "numeric-zero")
    print(json.dumps({model: trace(model, record) for model in REVISIONS}, indent=2))


if __name__ == "__main__":
    main()
