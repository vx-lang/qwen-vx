#!/usr/bin/env python3
"""Read the logs of run.sh, and write runs.csv and summary.md.

    python3 parse.py OUT
    python3 parse.py OUT --choose

OUT is the folder that run.sh wrote. The token counts of the prompts come from prompts/tokens.csv.
A number is missing from a row when its log does not have it, for example after a failed run.
With --choose, the script prints the fastest draft length of each program in table T, as
DRAFT=… MTP_N=… LOOKUP_N=…, for run.sh.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAMS = {
    "vx": "qwen-vx",
    "vxdraft": "qwen-vx --draft",
    "completion": "llama-completion",
    "lookup": "llama-lookup",
    "ngram": "llama.cpp ngram-mod",
    "mtp": "llama.cpp draft-mtp",
}
# The variable of run.sh that takes the chosen draft length of each program.
CHOICES = {"vxdraft": "DRAFT", "mtp": "MTP_N", "lookup": "LOOKUP_N"}
KINDS = ["edit", "complete", "write", "explain"]
SETS = {"T": "twelve", "C": "heldout"}


def num(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m[1]) if m else None


def parse_log(program, text):
    """The numbers of one log. Times are in ms."""
    r = {}
    if program.startswith("vx"):
        r["ms_per_token"] = num(r"time ms_per_token=([0-9.]+)", text)
        r["prompt_ms_per_token"] = num(r"time prompt_ms_per_token=([0-9.]+)", text)
        r["prompt_runs_ms"] = [float(x) for x in re.findall(r"time prompt_ms=([0-9.]+)", text)]
        m = re.search(r"rounds=(\d+) drafted=(\d+) accepted=(\d+)", text)
        if m:
            r["checks"], r["drafted"], r["accepted"] = int(m[1]), int(m[2]), int(m[3])
    elif program == "completion":
        perf = r"common_perf_print: *"
        r["ms_per_token"] = num(perf + r"eval time = *[0-9.]+ ms / *\d+ runs *\( *([0-9.]+) ms per token", text)
        r["timed_tokens"] = num(perf + r"eval time = *[0-9.]+ ms / *(\d+) runs", text, int)
        r["prompt_ms"] = num(perf + r"prompt eval time = *([0-9.]+) ms", text)
        r["llama_prompt_tokens"] = num(perf + r"prompt eval time = *[0-9.]+ ms / *(\d+) tokens", text, int)
    else:  # llama-lookup and llama-speculative-simple print the same lines
        m = re.search(r"decoded +(\d+) tokens in +([0-9.]+) seconds", text)
        if m:
            r["timed_tokens"] = int(m[1])
            r["ms_per_token"] = float(m[2]) * 1000 / int(m[1])
        m = re.search(r"encoded +(\d+) tokens in +([0-9.]+) seconds", text)
        if m:
            r["llama_prompt_tokens"], r["prompt_ms"] = int(m[1]), float(m[2]) * 1000
        r["drafted"] = num(r"n_drafted *= *(\d+)", text, int)
        r["accepted"] = num(r"n_accept *= *(\d+)", text, int)
    if not program.startswith("vx"):
        r["threads"] = num(r"n_threads = (\d+)", text, int)
    return r


def generated_text(path):
    """The output of the app after the line of the compiler, without its lines of times and drafts."""
    lines = path.read_text(errors="replace").split("\n")
    start = next((i + 1 for i, line in enumerate(lines) if line.startswith("[JIT] Compiling to native object")), None)
    assert start is not None, f"{path} has no line of the compiler"
    lines = [re.sub(r"rounds=\d+ drafted=\d+ accepted=\d+;", "", line) for line in lines[start:]
             if not line.startswith("time ")]
    return "\n".join(lines).rstrip("\n")


def read_rows(out):
    tokens = {row["prompt"]: int(row["tokens"]) for row in csv.DictReader(open(HERE / "prompts" / "tokens.csv"))}
    meta = {}
    for line in open(out / "log.txt"):
        m = re.match(r"(\S+) start=(\S+) end=(\S+) rc=(\d+) busy=\[(.*)\] gpu=(\d*)%", line)
        if m:
            meta[m[1]] = dict(start=m[2], end=m[3], rc=int(m[4]), busy=m[5], gpu=m[6])
    fa = re.search(r"flash attention:.*chosen: (\w+)", (out / "env.txt").read_text())
    rows = []
    for name, m in sorted(meta.items()):
        if name.startswith("llama-bench"):
            continue
        parts = name.split(".")
        setting = ""
        if name.startswith("fa-"):  # fa-on.ctx2048.r1
            table, prompt, rnd, program = "fa", f"context/{parts[1]}", parts[2], "completion"
        elif parts[0] in ("A", "P"):  # A.vx.ctx512.r1
            table, prompt, rnd = parts[0], f"context/{parts[2]}", parts[3]
            program = "vx" if parts[1] == "vx" else "completion"
        else:  # T.edit1.r1.vxdraft7
            table, prompt, rnd = parts[0], f"{SETS[parts[0]]}/{parts[1]}", parts[2]
            program, setting = re.fullmatch(r"([a-z]+)(\d*)", parts[3]).groups()
        r = parse_log(program, (out / "logs" / f"{name}.txt").read_text(errors="replace"))
        n = tokens[prompt]
        if r.get("prompt_ms_per_token") is not None:
            r["prompt_ms"] = r["prompt_ms_per_token"] * n
        runs = r.get("prompt_runs_ms") or []
        flash = parts[0][3:] if table == "fa" else (fa[1] if fa and not program.startswith("vx") else "")
        rows.append(dict(
            table=table, prompt=prompt, kind=re.sub(r"\d", "", prompt.split("/")[1]), round=rnd[1:],
            program=PROGRAMS[program], setting=setting, prompt_tokens=n,
            llama_prompt_tokens=r.get("llama_prompt_tokens"), prompt_ms=r.get("prompt_ms"),
            first_prompt_ms=runs[0] if runs else None, later_prompt_ms=" ".join(f"{x:.1f}" for x in runs[1:]),
            ms_per_token=r.get("ms_per_token"), timed_tokens=r.get("timed_tokens"), checks=r.get("checks"),
            drafted=r.get("drafted"), accepted=r.get("accepted"), flash_attn=flash, threads=r.get("threads"),
            log=f"logs/{name}.txt", **m,
        ))
    return rows


def write_csv(out, rows):
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def spread(xs, fmt="{:.2f}"):
    """The median, and the fastest and the slowest run."""
    return f"{fmt.format(statistics.median(xs))} ({fmt.format(min(xs))}-{fmt.format(max(xs))})"


def timed(rows, table, program=None, setting=None):
    return [r for r in rows if r["table"] == table and r["rc"] == 0 and r["ms_per_token"] is not None
            and (program is None or r["program"] == program) and (setting is None or r["setting"] == setting)]


def tuning(rows):
    """For each program and draft length of table T: the geometric mean over the prompts, if it ran on all."""
    prompts = {r["prompt"] for r in rows if r["table"] == "T"}
    means = {}
    for key in {(r["program"], r["setting"]) for r in timed(rows, "T")}:
        rr = timed(rows, "T", *key)
        if {r["prompt"] for r in rr} == prompts:
            means[key] = geomean([r["ms_per_token"] for r in rr])
    return means


def choose(rows):
    means = tuning(rows)
    words = []
    for program, variable in CHOICES.items():
        options = {s: v for (p, s), v in means.items() if p == PROGRAMS[program]}
        assert options, f"table T has no complete set of runs of {PROGRAMS[program]}"
        words.append(f"{variable}={min(options, key=options.get)}")
    return " ".join(words)


def same_text(out, rows, table):
    """For each run with --draft, whether its text is that of the run without drafts."""
    same = []
    for r in timed(rows, table, PROGRAMS["vxdraft"]):
        plain = out / re.sub(r"\.vxdraft\d+\.txt$", ".vx.txt", r["log"])
        if plain.exists():
            same.append(generated_text(plain) == generated_text(out / r["log"]))
    return same


def summary(out, rows):
    lines = []
    say = lines.append
    bad = [r["log"] for r in rows if r["rc"] != 0]
    if bad:
        say("Failed runs: " + ", ".join(bad))
        say("")
    wrong = sorted({r["prompt"] for r in rows if r["llama_prompt_tokens"] not in (None, r["prompt_tokens"])})
    if wrong:
        say("llama.cpp read another number of tokens for: " + ", ".join(wrong))
        say("")
    lines += [line.strip() for line in open(out / "env.txt") if line.startswith(("flash attention:", "draft lengths"))]
    say("")

    say("## Table A: generated tokens after a prompt")
    say("")
    say("ms for each generated token: the median of the runs, and the fastest and slowest run. The prompt time "
        "is that of the first prompt of a process.")
    say("")
    say("| Prompt tokens | qwen-vx ms | llama.cpp ms | qwen-vx tok/s | llama.cpp tok/s | llama.cpp ms / qwen-vx ms "
        "| qwen-vx prompt ms | llama.cpp prompt ms |")
    say("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for prompt in sorted({r["prompt"] for r in rows if r["table"] == "A"}, key=lambda p: int(p.split("ctx")[1])):
        a = [r for r in timed(rows, "A", "qwen-vx") if r["prompt"] == prompt]
        b = [r for r in timed(rows, "A", "llama-completion") if r["prompt"] == prompt]
        if not a or not b:
            continue
        ma, mb = statistics.median(r["ms_per_token"] for r in a), statistics.median(r["ms_per_token"] for r in b)
        pa = [r["prompt_ms"] for r in a if r["prompt_ms"] is not None]
        pb = [r["prompt_ms"] for r in b if r["prompt_ms"] is not None]
        say(f"| {a[0]['prompt_tokens']:,} | {spread([r['ms_per_token'] for r in a])} "
            f"| {spread([r['ms_per_token'] for r in b])} | {1000 / ma:.1f} | {1000 / mb:.1f} | {mb / ma:.3f} "
            f"| {spread(pa, '{:.0f}') if pa else ''} | {spread(pb, '{:.0f}') if pb else ''} |")
    say("")

    say("## Table P: the prompt")
    say("")
    say("Tokens of the prompt each second: the median of the runs, and the slowest and fastest run. qwen-vx ran "
        "the prompt several times in one process. The first run includes the first launch of each kernel. "
        "llama-completion ran the prompt once, after its warm-up.")
    say("")
    say("| Prompt tokens | qwen-vx first run | qwen-vx later runs | llama-completion |")
    say("| ---: | ---: | ---: | ---: |")
    for prompt in sorted({r["prompt"] for r in rows if r["table"] == "P"}, key=lambda p: int(p.split("ctx")[1])):
        ok = [r for r in rows if r["table"] == "P" and r["rc"] == 0 and r["prompt"] == prompt]
        n = ok[0]["prompt_tokens"] if ok else 0
        first = [n * 1000 / r["first_prompt_ms"] for r in ok if r["program"] == "qwen-vx" and r["first_prompt_ms"]]
        later = [n * 1000 / float(x) for r in ok if r["program"] == "qwen-vx" for x in r["later_prompt_ms"].split()]
        llama = [n * 1000 / r["prompt_ms"] for r in ok if r["program"] == "llama-completion" and r["prompt_ms"]]
        say(f"| {n:,} | {spread(first, '{:.0f}') if first else ''} | {spread(later, '{:.0f}') if later else ''} "
            f"| {spread(llama, '{:.0f}') if llama else ''} |")
    say("")

    means = tuning(rows)
    if means:
        say("## Table T: draft lengths, on the 12 tuning prompts")
        say("")
        say("ms for each generated token: the geometric mean over the 12 prompts, one run each.")
        say("")
        say("| Program | Draft length | ms | accepted / drafted |")
        say("| --- | ---: | ---: | ---: |")
        for (program, setting), v in sorted(means.items(), key=lambda kv: (kv[0][0], int(kv[0][1] or 0))):
            rr = timed(rows, "T", program, setting)
            acc, dr = sum(r["accepted"] or 0 for r in rr), sum(r["drafted"] or 0 for r in rr)
            say(f"| {program} | {setting} | {v:.2f} | {f'{acc} / {dr} ({100 * acc / dr:.0f}%)' if dr else ''} |")
        same = same_text(out, rows, "T")
        say("")
        say(f"- qwen-vx text with and without drafts: the same in {sum(same)} of {len(same)} pairs")
        say("")

    programs = [p for p in PROGRAMS.values() if timed(rows, "C", p)]
    if programs:
        say("## Table C: the 12 held-out prompts")
        say("")
        say("ms for each generated token: the geometric mean over the prompts of each kind, and the fastest and "
            "slowest run.")
        say("")
        say("| Kind | Prompt tokens | " + " | ".join(programs) + " |")
        say("| --- | --- | " + " | ".join("---:" for _ in programs) + " |")
        for kind in KINDS + [None]:
            cells = []
            for p in programs:
                xs = [r["ms_per_token"] for r in timed(rows, "C", p) if kind is None or r["kind"] == kind]
                cells.append(f"{geomean(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})" if xs else "")
            counts = sorted({r["prompt_tokens"] for r in timed(rows, "C", "qwen-vx") if r["kind"] == kind})
            say(f"| {kind or 'all 12'} | {', '.join(map(str, counts))} | " + " | ".join(cells) + " |")
        say("")
        mean = {}
        for p in programs:
            by = defaultdict(list)
            for r in timed(rows, "C", p):
                by[r["prompt"]].append(r["ms_per_token"])
            mean[p] = {k: statistics.mean(v) for k, v in by.items()}

        def compare(a, b):
            ks = [k for k in mean.get(a, {}) if k in mean.get(b, {})]
            ratios = [mean[b][k] / mean[a][k] for k in ks]
            if ratios:
                say(f"- {a} against {b}: {geomean(ratios):.3f}, faster on {sum(x > 1 for x in ratios)} of "
                    f"{len(ratios)} prompts ({min(ratios):.2f} to {max(ratios):.2f})")

        compare("qwen-vx", "llama-completion")
        for p in programs:
            if p != "qwen-vx --draft":
                compare("qwen-vx --draft", p)
        say("")
        say("### Drafts, table C, round 1")
        say("")
        drafters = [p for p in programs if p not in ("qwen-vx", "llama-completion")]
        say("| Kind | " + " | ".join(f"{p} accepted / drafted" for p in drafters) + " |")
        say("| --- | " + " | ".join("---:" for _ in drafters) + " |")
        for kind in KINDS + [None]:
            cells = []
            for p in drafters:
                rr = [r for r in timed(rows, "C", p) if r["round"] == "1" and (kind is None or r["kind"] == kind)]
                acc, dr = sum(r["accepted"] or 0 for r in rr), sum(r["drafted"] or 0 for r in rr)
                cells.append(f"{acc} / {dr} ({100 * acc / dr:.0f}%)" if dr else "")
            say(f"| {kind or 'all 12'} | " + " | ".join(cells) + " |")
        same = same_text(out, rows, "C")
        say("")
        say(f"- qwen-vx text with and without drafts: the same in {sum(same)} of {len(same)} pairs")
        say("")

    for name in ("llama-bench-pp", "llama-bench-tg"):
        path = out / "logs" / f"{name}.txt"
        if path.exists():
            say(f"## {name}")
            say("")
            lines += [line for line in path.read_text().splitlines() if line.startswith("|")]
            say("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[2:] not in ([], ["--choose"]):
        sys.exit(__doc__.split("\n\n")[1])
    out = Path(sys.argv[1])
    rows = read_rows(out)
    if sys.argv[2:] == ["--choose"]:
        print(choose(rows))
        return
    write_csv(out, rows)
    summary(out, rows)


if __name__ == "__main__":
    main()
