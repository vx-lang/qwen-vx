#!/usr/bin/env python3
"""Chat with a model through a session of the Vx program, which keeps the model and its KV cache.

    python3 scripts/chat.py WEIGHTS TOKENIZER [--system TEXT] [--cache FILE] [--context N]
    python3 scripts/chat.py WEIGHTS TOKENIZER --check

The session runs `src/main.vx --session`. vxc does not give its standard input to the program, and
keeps its standard output until it ends, so the requests and the replies go through two named
pipes. Each turn sends the whole conversation in the chat format of
the Instruct models, and the session runs only the tokens after the longest prefix that its KV
cache holds. With --cache, the session loads the KV cache of the system prompt from FILE, or makes
FILE first when it is missing. So a long system prompt runs once, ahead of time.

--check runs a short conversation and prints whether each turn gives the text of a fresh run of
the Vx program, whether the later turns reuse the cache, and whether a saved cache gives the same
text in a new session. `tests/session.vx` runs it.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VX_HOME = Path(os.environ.get("VX_HOME", Path.home() / ".vx"))
VX_FLEET = Path(os.environ.get("VX_FLEET", VX_HOME / "current" / "fleet"))
DONE = re.compile(rb"^done prompt=(\d+) reused=(\d+) ran=(\d+) generated=(\d+) prompt_us=(\d+) us_per_token=(\d+)")


def vxc_command(weights, tokenizer, *program_args):
    vxc = os.environ.get("VXC", "vxc")
    return [vxc, str(ROOT / "src" / "main.vx"), "--package", f"qwen={ROOT / 'src'}", "--host", "default",
            "--machine", str(VX_FLEET / "m4-uma.vx"), "--", str(weights), "--tokenizer", str(tokenizer), *program_args]


def chat_text(system, turns):
    """The prompt of a conversation: an optional system message, and then (user, reply) pairs. The
    last reply is None, for the turn that the model is to write."""
    text = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    for user, reply in turns:
        text += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        if reply is not None:
            text += f"{reply}<|im_end|>\n"
    return text


class Session:
    """A running session of the Vx program."""

    def __init__(self, weights, tokenizer, context):
        self.dir = tempfile.TemporaryDirectory()
        requests = os.path.join(self.dir.name, "requests")
        replies = os.path.join(self.dir.name, "replies")
        os.mkfifo(requests)
        os.mkfifo(replies)
        self.log = open(os.path.join(self.dir.name, "log"), "wb")
        command = vxc_command(weights, tokenizer, "--session", "--context", str(context),
                              "--requests", requests, "--replies", replies)
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        # Each open waits for the session to open the other end of the pipe.
        self.replies = open(replies, "rb", buffering=0)
        line = self._line()
        if not line.startswith(b"ready "):
            raise RuntimeError(f"the session did not start: {line!r}")
        self.requests = open(requests, "wb", buffering=0)

    def _line(self):
        line = bytearray()
        while True:
            b = self.replies.read(1)
            if not b or b == b"\n":
                return bytes(line)
            line += b

    def _reply(self, stream=None):
        """The bytes up to the NUL byte, and the line after it."""
        text = bytearray()
        while True:
            b = self.replies.read(1)
            if not b:
                raise RuntimeError("the session ended in the middle of a reply")
            if b == b"\0":
                break
            text += b
            if stream:
                stream.write(b.decode("utf-8", "replace"))
                stream.flush()
        return bytes(text), self._line()

    def generate(self, prompt, steps, stream=None):
        data = prompt.encode("utf-8")
        self.requests.write(f"generate {steps} {len(data)}\n".encode() + data)
        text, line = self._reply(stream)
        m = DONE.match(line)
        if not m:
            raise RuntimeError(line.decode("utf-8", "replace"))
        keys = ["prompt", "reused", "ran", "generated"]
        stats = dict(zip(keys, map(int, m.groups()[:4])))
        stats["prompt_ms"], stats["ms_per_token"] = int(m.group(5)) / 1000, int(m.group(6)) / 1000
        return text.decode("utf-8", "replace"), stats

    def _simple(self, request):
        self.requests.write(request.encode() + b"\n")
        _, line = self._reply()
        if not line.startswith(b"ok "):
            raise RuntimeError(line.decode("utf-8", "replace"))
        return line.decode()

    def save(self, path):
        return self._simple(f"save {path}")

    def load(self, path):
        return self._simple(f"load {path}")

    def close(self):
        self.requests.close()
        self.process.wait()
        self.replies.close()
        self.log.close()
        self.dir.cleanup()


def fresh_reply(weights, tokenizer, prompt, steps, folder):
    """The text that a run of the Vx program without a session writes after `prompt`."""
    path = os.path.join(folder, "prompt.txt")
    Path(path).write_bytes(prompt.encode("utf-8"))
    out = subprocess.run(vxc_command(weights, tokenizer, "--prompt-file", path, "--steps", str(steps)),
                         capture_output=True, check=True).stdout
    out = b"\n".join(l for l in out.split(b"\n") if not l.startswith(b"[JIT]"))
    data = prompt.encode("utf-8")
    assert out.startswith(data), "a fresh run did not print the prompt first"
    return out[len(data):].split(b"\ntime ms_per_token")[0].decode("utf-8", "replace")


def check(args):
    users = ["def fibonacci(n):", "and the total price?", "fn main() {"]
    system = "You write code."
    steps = 8
    with tempfile.TemporaryDirectory() as folder:
        cache = os.path.join(folder, "cache.kv")
        session = Session(args.weights, args.tokenizer, args.context)
        turns, prompts, replies = [], [], []
        for i, user in enumerate(users):
            turns.append((user, None))
            prompt = chat_text(system, turns)
            reply, stats = session.generate(prompt, steps)
            turns[-1] = (user, reply)
            prompts.append(prompt)
            replies.append(reply)
            same = reply == fresh_reply(args.weights, args.tokenizer, prompt, steps, folder)
            print(f"turn {i + 1}: prompt {stats['prompt']} tokens, reused {stats['reused'] > 0}, same text as a fresh run: {same}")
            if i == 1:
                print(f"saved: {session.save(cache).startswith('ok tokens=')}")
        session.close()
        session = Session(args.weights, args.tokenizer, args.context)
        print(f"loaded: {session.load(cache).startswith('ok tokens=')}")
        reply, stats = session.generate(prompts[2], steps)
        print(f"turn 3 after the load: reused {stats['reused'] > 0}, same text as the first session: {reply == replies[2]}")
        session.close()


def interactive(args):
    system = args.system
    if args.system_file:
        system = Path(args.system_file).read_text()
    session = Session(args.weights, args.tokenizer, args.context)
    if args.cache and system:
        if os.path.exists(args.cache):
            print(f"[{session.load(os.path.abspath(args.cache))} from {args.cache}]")
        else:
            _, stats = session.generate(chat_text(system, []), 0)
            print(f"[the system prompt ran: {stats['prompt']} tokens in {stats['prompt_ms']:.0f} ms. {session.save(os.path.abspath(args.cache))}]")
    turns = []
    try:
        while True:
            user = input("> ")
            turns.append((user, None))
            reply, stats = session.generate(chat_text(system, turns), args.steps, stream=sys.stdout)
            turns[-1] = (user, reply)
            print(f"\n[prompt {stats['prompt']} tokens, {stats['reused']} from the cache, {stats['ran']} ran in "
                  f"{stats['prompt_ms']:.0f} ms; {stats['generated']} new tokens at {stats['ms_per_token']:.1f} ms each]")
    except (EOFError, KeyboardInterrupt):
        print()
    session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights", help="a weight file of convert.py")
    parser.add_argument("tokenizer", help="a tokenizer file of convert_tokenizer.py")
    parser.add_argument("--system", help="the system prompt")
    parser.add_argument("--system-file", help="a file with the system prompt")
    parser.add_argument("--cache", help="a file for the KV cache of the system prompt")
    parser.add_argument("--context", type=int, default=8192, help="the size of the KV cache, in tokens")
    parser.add_argument("--steps", type=int, default=1024, help="the most tokens of a reply")
    parser.add_argument("--check", action="store_true", help="check the session against fresh runs")
    args = parser.parse_args()
    if args.check:
        check(args)
    else:
        interactive(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
