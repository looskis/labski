"""vLLM 0.33 K2 parser for responses that close with a different effort marker."""

import re

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.k2_horizon_reasoning_parser import K2HorizonReasoningParser
from vllm.tool_parsers.utils import partial_tag_overlap


@ReasoningParserManager.register_module("labski_k2_effort")
class K2EffortParser(K2HorizonReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        ends = [end for _, end in self._EFFORT_TOKENS.values() if end in self.vocab]
        self._end_ids = {self.vocab[end] for end in ends}
        self._markers = (*ends, self._TOOL_CALLS_START)
        self._boundary = re.compile("|".join(map(re.escape, self._markers)))

    def _split_at_boundary(self, text):
        match = self._boundary.search(text)
        if match is None:
            return text, None, False
        offset = match.start() if match.group() == self._TOOL_CALLS_START else match.end()
        return text[:match.start()], text[offset:] or None, True

    def _split_model_output(self, model_output):
        reasoning, content, _ = self._split_at_boundary(self._without_generated_start(model_output))
        return reasoning, content

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text,
                                    previous_token_ids, current_token_ids, delta_token_ids):
        if self._stream_finished:
            return DeltaMessage(content=delta_text) if delta_text else None
        self._stream_buffer += delta_text
        if not self._strip_stream_start():
            return None
        reasoning, content, finished = self._split_at_boundary(self._stream_buffer)
        if finished:
            self._stream_finished = True
            self._stream_buffer = ""
            return DeltaMessage(reasoning=reasoning or None, content=content)
        # A single marker may span several streamed text fragments.
        overlap = max(partial_tag_overlap(reasoning, marker) for marker in self._markers)
        cut = len(reasoning) - overlap
        self._stream_buffer = reasoning[cut:]
        return DeltaMessage(reasoning=reasoning[:cut]) if cut else None

    def is_reasoning_end(self, input_ids):
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return False
            if token_id in self._end_ids or token_id == self._tool_calls_start_id:
                return True
        return False

    def is_reasoning_end_streaming(self, input_ids, delta_ids):
        return self._stream_finished or self.is_reasoning_end(tuple(delta_ids))

    def extract_content_ids(self, input_ids):
        for i, token_id in enumerate(input_ids):
            if token_id in self._end_ids:
                return input_ids[i + 1:]
            if token_id == self._tool_calls_start_id:
                return input_ids[i:]
        return input_ids if self._stream_finished else []
