"""LLM-jp Harmony dialect (llm-jp/llm-jp-4.1-*-thinking).

The models emit OpenAI's Harmony format like gpt-oss, with two differences:

* The SentencePiece-style tokenizer decodes a space after every special
  token, so generated text reads ``<|channel|> analysis<|message|> ...`` and
  every message body carries one leading space that is not content.
* Parallel tool calls are consecutive assistant messages; every call but the
  last is closed by ``<|end|>``, the last one by ``<|call|>``. The model
  config declares ``<|call|>`` as an end-of-sequence token, so it ends the
  generation and never reaches the text.

The chat template declares the dialect with ``chat_format=llm-jp-harmony-v1``;
``tokenizer_utils`` selects this module and infers the reasoning and
structural markers of the dialect from the same declaration.

A tool call segment starts after ``to=functions.`` and ends at ``<|end|>`` or
at the end of the generation, so ``parse_tool_call`` receives text such as::

    get_weather<|channel|> commentary <|constrain|>  json<|message|> {"city": "Tokyo"}
    get_weather <|constrain|>  json<|message|> {"city": "Tokyo"}
"""

import json
import re
from typing import Any, Optional

_NAME = re.compile(r"^\s*([^\s<]+)")
_BODY = "<|message|>"


def parse_tool_call(text: str, _: Optional[Any] = None) -> dict:
    match = _NAME.match(text)
    if not match:
        raise ValueError("No function name found.")
    body_at = text.rfind(_BODY)
    if body_at < 0:
        raise ValueError("No tool call arguments found.")
    body = text[body_at + len(_BODY) :]
    if body.startswith(" "):
        body = body[1:]
    arguments = json.loads(body) if body.strip() else {}
    return dict(name=match.group(1), arguments=arguments)


tool_call_start = " to=functions."
tool_call_end = "<|end|>"
