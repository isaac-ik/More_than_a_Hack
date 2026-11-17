"""Parse the tactical ATD once into dense numpy arrays and cache them.

Output (solution/cache/atd.npz, gitignored -- it is derived match data and must not be published):
    px, py   float32 [n_frames, n_slots]   player positions in METRES (NaN where the slot is empty)
    ps       float32 [n_frames, n_slots]   player speed in m/s as delivered
    bx, by   float32 [n_frames]            ball position in metres (NaN where the ball is missing)
    team     int8    [n_slots]             0 = home, 1 = away

Row i is video frame i (ATD frame_id - 1), which is also submission row i.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
CACHE = os.path.join(CACHE_DIR, "atd.npz")


def team_of_each_track(meta_path):
    """0 for slots of the first (home) team in the metadata, 1 for the other."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams, found {}".format(len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def parse(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)
    k = len(team)

    px = np.full((n, k), np.nan, dtype=np.float32)
    py = np.full((n, k), np.nan, dtype=np.float32)
    ps = np.full((n, k), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n:
                continue
            groups = parts[1].split(";")
            flat = np.array(",".join(groups).split(","), dtype=np.float32)
            if flat.size == 3 * k:
                flat = flat.reshape(k, 3)
                px[i], py[i], ps[i] = flat[:, 0], flat[:, 1], flat[:, 2]
            else:  # ragged line: fall back to per-slot parsing
                for slot, group in enumerate(groups[:k]):
                    vals = group.split(",")
                    if len(vals) >= 2 and vals[0][:1] != "N":
                        px[i, slot], py[i, slot] = float(vals[0]), float(vals[1])
                        if len(vals) > 2:
                            ps[i, slot] = float(vals[2])
            ball = parts[2].split(",")
            if ball[0][:1] != "N":
                bx[i], by[i] = float(ball[0]), float(ball[1])

    px *= length
    bx *= length
    py *= width
    by *= width
    return {"px": px, "py": py, "ps": ps, "bx": bx, "by": by, "team": team}


def load(game=GAME, data_dir=None, refresh=False):
    """Cached arrays as a dict. Parses the raw file on first use (about a minute)."""
    if os.path.isfile(CACHE) and not refresh:
        with np.load(CACHE) as z:
            return {key: z[key] for key in z.files}
    data = parse(game, data_dir)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(CACHE, **data)
    return data


if __name__ == "__main__":
    d = load(refresh=True)
    print("frames {}, slots {}, ball seen on {:.1f}% of frames".format(
        len(d["bx"]), len(d["team"]), 100.0 * np.isfinite(d["bx"]).mean()))
