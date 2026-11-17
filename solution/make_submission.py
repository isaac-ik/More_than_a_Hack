"""Data in, submission out.

    python solution/make_submission.py

writes output/Isaac.csv. Pipeline (see Isaac.md):

  1. parse the tactical ATD (cached)                              io_atd.py
  2. discrete observations on 0.2 s blocks                        features.py
  3. alive / dead segmentation, 2-state HMM, label-free tables    deadball.py
     (nothing is alive before a half's Smart Tagging kick-off tag)
  4. team in possession by the controlled-possession rule         possession.py
  5. expand blocks to frames, X outside the two halves, write the CSV

No ground truth is read anywhere in this file or in anything it imports, and there is no
randomness: the same data always gives the same file, byte for byte.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
import deadball  # noqa: E402
import features  # noqa: E402
import io_atd  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession  # noqa: E402

TEAM_NAME = "Isaac"
RESULTS_DIR = os.path.join(ROOT, "output")
LETTERS = np.array(["H", "A", "D", "X"])
X_CODE = 3


def kickoff_tag_starts(game=GAME):
    """First frame of each 'Kick Off (Start)' Smart Tagging instance, in time order.

    The tag is a ~5 s window that contains the kick-off, so nothing before its first frame can be
    live play. Read from the provided automatic tagging -- not typed in from the documentation.
    """
    starts = [e["start_frame"] for e in load_smart_tagging.load(game)
              if (e["code"] or "").strip().lower() == "set pieces"
              and (e["tags"].get("Type") or "").strip().lower() == "kick off (start)"]
    return sorted(starts)


def prepare(game=GAME):
    """Everything that does not depend on a decoder setting: observations per half."""
    d = io_atd.load(game)
    length, width = GAMES[game]["pitch"]
    f = features.frame_features(d, length, width)
    halves = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    sequences = [features.block_observations(f, a, b) for a, b in halves]
    # For each half, the block index before which the ball cannot be alive: the start of the
    # kick-off tag that falls inside (or just before) that half. 0 if there is none.
    dead_until = []
    for a, b in halves:
        inside = [k for k in kickoff_tag_starts(game) if a - 25 * 60 <= k < b]
        dead_until.append(max(0, (inside[0] - a) // features.BLOCK) if inside else 0)
    return {"n": len(d["bx"]), "halves": halves, "sequences": sequences, "frames": f,
            "dead_until": dead_until}


def decode(prep, dead_kwargs=None, team_kwargs=None, tables=None):
    """Label codes (0 H / 1 A / 2 D / 3 X) for every video frame."""
    dead_kwargs = dead_kwargs or {}
    team_kwargs = team_kwargs or {}
    if tables is None:
        tables = deadball.estimate_tables(prep["sequences"])
    codes = np.full(prep["n"], X_CODE, dtype=np.int8)
    for (a, b), (ball, tempo, control), until in zip(prep["halves"], prep["sequences"],
                                                     prep["dead_until"]):
        dead = deadball.segment(tables, ball, tempo, dead_until=until, **dead_kwargs)
        blocks = possession.label_half(control, dead, **team_kwargs)
        codes[a:b] = np.repeat(blocks, features.BLOCK)[:b - a]
    return codes


def write_submission(path, codes):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    labels = LETTERS[codes]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=TEAM_NAME, help="file name: the team name")
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()

    prep = prepare()
    tables = deadball.estimate_tables(prep["sequences"])
    for key in sorted(tables):
        print("{:<12} {}".format(key, np.round(tables[key], 3)))
    codes = decode(prep, tables=tables)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), codes)
    for k, lab in enumerate(LETTERS):
        print("{} {:>7}".format(lab, int(np.sum(codes == k))))
    print("wrote", os.path.relpath(path, ROOT))
