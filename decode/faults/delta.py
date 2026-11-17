"""Prints the change of each value in a file of sampler.c between two times.

    python3 delta.py FILE.tsv START END [MS_PER_TOKEN]

START and END are seconds from the start of the process. A negative time counts back from the
last sample, the one that the process writes at its exit. With MS_PER_TOKEN, the script also
prints the change for each token, for the tokens that the window holds.
"""

import sys


def main():
    path, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    ms_per_token = float(sys.argv[4]) if len(sys.argv) > 4 else None
    with open(path) as f:
        names = f.readline().split()
        rows = [[float(v) for v in line.split()] for line in f if line.strip()]
    last = rows[-1][0]
    if start < 0:
        start += last
    if end < 0:
        end += last

    def at(t):
        return min(rows, key=lambda r: abs(r[0] - t))

    a, b = at(start), at(end)
    seconds = b[0] - a[0]
    print(f"from {a[0]:.3f} s to {b[0]:.3f} s: {seconds:.3f} s")
    tokens = seconds * 1000 / ms_per_token if ms_per_token else None
    if tokens:
        print(f"about {tokens:.1f} tokens")
    for i, name in enumerate(names[1:], 1):
        change = b[i] - a[i]
        line = f"{name:10} {change:16.3f}"
        if name in ("footprint", "resident"):
            line = f"{name:10} {a[i] / 1e9:8.2f} GB to {b[i] / 1e9:.2f} GB"
        elif tokens:
            per = change / tokens
            line += f"   {per * 1000:.3f} ms a token" if name in ("user", "sys") else f"   {per:.2f} a token"
        print(line)


if __name__ == "__main__":
    main()
