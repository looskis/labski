"""Exercise K2 content boundaries in the pinned vLLM image with the real tokenizer."""

import argparse
from transformers import AutoTokenizer
from k2_effort_parser import K2EffortParser


def verify(tokenizer):
    tests = 0
    for effort, (opening, _) in K2EffortParser._EFFORT_TOKENS.items():
        def parser():
            return K2EffortParser(tokenizer, chat_template_kwargs={"reasoning_effort": effort})

        markers = [end for _, end in K2EffortParser._EFFORT_TOKENS.values()]
        cases = [(opening + "work" + end + "B", "work", "B") for end in markers]
        cases += [("work", "work", ""), ("work<ifm|tool_calls>x", "work", "<ifm|tool_calls>x"),
                  ("work" + markers[0] + "literal" + markers[1], "work", "literal" + markers[1])]
        for full, expected_reasoning, expected_answer in cases:
            assert parser()._split_model_output(full) == (expected_reasoning, expected_answer or None)
            for cut in range(len(full) + 1):
                current = parser()
                pieces = [current.extract_reasoning_streaming("", "", part, [], [], [])
                          for part in (full[:cut], full[cut:])]
                assert "".join(m.reasoning or "" for m in pieces if m) == expected_reasoning
                assert "".join(m.content or "" for m in pieces if m) == expected_answer
                tests += 1
        instance = parser()
        for end in markers:
            closing = tokenizer.convert_tokens_to_ids(end)
            token = tokenizer.encode("B", add_special_tokens=False)
            seq = [instance.start_token_id, closing, *token]
            assert instance.is_reasoning_end(seq) and instance.is_reasoning_end_streaming([], [closing])
            assert instance.extract_content_ids(seq) == token
    print(f"Passed {tests} streaming split points, complete-text and token-ID checks")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("tokenizer")
    verify(AutoTokenizer.from_pretrained(cli.parse_args().tokenizer, local_files_only=True, trust_remote_code=False))
