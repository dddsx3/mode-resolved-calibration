#!/usr/bin/env python3
"""make_tables · rebuild paper tables (Table I-VII interface frozen).

Tables are rebuilt only from the machine-readable summaries under results/; remain frozen.
"""

from __future__ import annotations

import argparse

TABLES = {f"Table {roman}": i for i, roman in
          enumerate(["I", "II", "III", "IV", "V", "VI", "VII"], start=1)}


def make_table(n, out_dir="paper/tables"):
    raise NotImplementedError(
        f"Table {n} not implemented yet (interface frozen: --table {n}, input under results/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, required=True, choices=range(1, 8))
    args = ap.parse_args()
    make_table(args.table)


if __name__ == "__main__":
    main()
