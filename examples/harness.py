#!/usr/bin/env python3
"""Turn a request into the chat prompt of the Instruct model, and decode what the Vx program wrote.

    python3 harness.py encode TOKENIZER_DIR PROMPT_FILE [SYSTEM_FILE]
    python3 harness.py encode_fix TOKENIZER_DIR PROMPT_FILE REPLY_FILE ERROR_FILE
    python3 harness.py decode TOKENIZER_DIR VX_OUTPUT OUT_DIR

`encode` prints the token ids of the prompt, separated by commas, for `main.vx --tokens`.
`encode_fix` adds a second turn that gives the model the compiler's error for its reply.
`decode` writes the reply up to <|im_end|> as reply.md, and its longest code block as code.<ext>.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import re
import sys
from pathlib import Path

from tokenizers import Tokenizer


def tokenizer(model_dir):
    return Tokenizer.from_file(str(Path(model_dir).expanduser() / "tokenizer.json"))


def encode(model_dir, prompt_file, system_file=None):
    """The chat prompt of one request. A system file, such as RUST-SKILLS.md, goes first."""
    text = Path(prompt_file).read_text().strip()
    chat = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    if system_file:
        chat = f"<|im_start|>system\n{Path(system_file).read_text().strip()}<|im_end|>\n" + chat
    ids = tokenizer(model_dir).encode(chat, add_special_tokens=False).ids
    print(",".join(map(str, ids)))


def encode_fix(model_dir, prompt_file, reply_file, error_file):
    """A second turn: the first request, the model's reply, and the compiler's error."""
    first = Path(prompt_file).read_text().strip()
    reply = Path(reply_file).read_text()
    error = Path(error_file).read_text().strip()
    second = f"The compiler rejects this code:\n\n```\n{error}\n```\n\nFix it. Reply with only the whole corrected code, in one ```rust block."
    chat = (f"<|im_start|>user\n{first}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n"
            f"<|im_start|>user\n{second}<|im_end|>\n<|im_start|>assistant\n")
    ids = tokenizer(model_dir).encode(chat, add_special_tokens=False).ids
    print(",".join(map(str, ids)))


def decode(model_dir, vx_output, out_dir):
    tok = tokenizer(model_dir)
    out = Path(vx_output).read_text()
    ids = [int(t) for t in re.findall(r"^step=\d+ token=(\d+)", out, re.M)]
    end = tok.token_to_id("<|im_end|>")
    stopped = end in ids
    if stopped:
        ids = ids[: ids.index(end)]
    reply = tok.decode(ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reply.md").write_text(reply)
    blocks = re.findall(r"```(\w*)\n(.*?)```", reply, re.S)
    if blocks:
        lang, code = max(blocks, key=lambda b: len(b[1]))
        ext = {"rust": "rs", "cpp": "cpp", "c++": "cpp", "python": "py"}.get(lang.lower(), lang or "txt")
        (out_dir / f"code.{ext}").write_text(code)
    ms = re.search(r"ms_per_token=([0-9.]+)", out)
    print(f"tokens={len(ids)} stopped_at_im_end={stopped} code_blocks={len(blocks)} "
          f"ms_per_token={float(ms.group(1)):.1f}" if ms else "")


if __name__ == "__main__":
    {"encode": encode, "encode_fix": encode_fix, "decode": decode}[sys.argv[1]](*sys.argv[2:])
