"""Check a saved Q-table without clicking: from which starting squares does the greedy bot deliver?

Run:  python check_coverage.py                 checks q_table.npy
      python check_coverage.py q_fixed.npy     checks another table
      python check_coverage.py q_table.npy 46 41 26 1     also test the squares (46, 41) and (26, 1)

Prints the share of squares that deliver (without / carrying the package), a map in 10 x 10
blocks, and the result for each square you list.
"""
import sys

import numpy as np

from env import WarehouseEnv as E
from test_bot import run


def delivered_map(q, has):
    """{square: delivered?} for every free square the bot can start on."""
    result = {}
    for cell in E.FREE_CELLS:
        if (cell == E.DELIVERY and has) or (cell == E.PACKAGE and not has):
            continue
        result[cell] = run(q, cell, has)[-1]["delivered"]
    return result


def print_blocks(result):
    print("            cols " + "".join(f"{c * 10:>2d}-{c * 10 + 9:<2d}  " for c in range(5)))
    for br in range(5):
        row = ""
        for bc in range(5):
            values = [ok for (r, c), ok in result.items() if r // 10 == br and c // 10 == bc]
            row += f"{100 * np.mean(values):5.0f}%  "
        print(f"  rows {br * 10:2d}-{br * 10 + 9:2d} {row}")


def main():
    args = sys.argv[1:]
    q_file = args[0] if args else "q_table.npy"
    numbers = [int(a) for a in args[1:]]
    squares = list(zip(numbers[0::2], numbers[1::2]))
    q = np.load(q_file)

    print(f"Table: {q_file}")
    for has, name in ((False, "without the package"), (True, "while carrying the package")):
        result = delivered_map(q, has)
        share = 100 * np.mean(list(result.values()))
        print(f"\nStarting {name}: the bot delivers from {share:.0f}% of squares")
        print_blocks(result)
        for cell in squares:
            if cell in result:
                print(f"  from {cell}: {'delivers' if result[cell] else 'does NOT deliver'}")
            else:
                print(f"  from {cell}: not a valid starting square for this case")


if __name__ == "__main__":
    main()
