"""Turn the cached tracking arrays into a small set of discrete per-block observations.

Everything is computed on blocks of BLOCK frames (0.2 s). Consecutive 25 fps frames are almost
perfectly correlated, and an HMM that treats each as independent evidence becomes absurdly
over-confident and flickers. One observation per 0.2 s is roughly one independent look at the game.

Three observation channels, each a small categorical variable:

  ball      0 missing | 1 out of bounds | 2 still | 3 slow | 4 fast
  tempo     quintile-like bin of the mean player speed (a dead ball is mostly players walking)
  control   0 nobody | 1 home player on the ball | 2 away player on the ball
            (only defined when the ball is seen and in bounds)

All thresholds are physical quantities chosen a priori (metres, m/s); none was fitted to the
ground truth.
"""

import warnings

import numpy as np

# Params -- physical, set a priori
BLOCK = 5                 # frames per block (0.2 s at 25 fps)
OOB_MARGIN = 0.5          # m beyond a boundary line before the ball counts as out
STILL_SPEED = 1.0         # m/s: a ball slower than this is "placed", e.g. for a set piece
FAST_SPEED = 6.0          # m/s: a ball faster than this is a pass or shot in flight
CONTROL_RADIUS = 1.5      # m: a player this close to the ball is "on the ball"
CONTROL_MARGIN = 0.5      # m: ...and the nearest opponent must be at least this much further
TEMPO_EDGES = (0.8, 1.2, 1.6, 2.0, 2.6)   # m/s bin edges for mean player speed
TEMPO_SMOOTH = 12         # +/- frames for the mean-speed smoother (about 1 s in total)
MAX_BALL_STEP = 3.0       # m per frame (75 m/s): a larger jump is a tracker glitch

N_BALL, N_TEMPO, N_CONTROL = 5, len(TEMPO_EDGES) + 1, 3


def _box_mean(x, half):
    """Centred moving average that ignores NaN."""
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def frame_features(d, length, width):
    """Per-frame continuous features from the cached arrays (see io_atd.load)."""
    bx, by = d["bx"].astype(np.float64), d["by"].astype(np.float64)
    px, py, ps, team = d["px"], d["py"], d["ps"], d["team"]
    n = len(bx)
    seen = np.isfinite(bx)

    # Ball speed from a +/-2 frame central difference; glitch jumps are treated as unseen speed.
    step = np.full(n, np.nan)
    step[1:] = np.hypot(np.diff(bx), np.diff(by))
    speed = np.full(n, np.nan)
    speed[2:-2] = np.hypot(bx[4:] - bx[:-4], by[4:] - by[:-4]) / 4.0 * 25.0
    one_sided = np.isnan(speed) & np.isfinite(step)
    speed[one_sided] = step[one_sided] * 25.0
    glitch = np.isfinite(step) & (step > MAX_BALL_STEP)

    out = seen & ((bx < -OOB_MARGIN) | (bx > length + OOB_MARGIN) |
                  (by < -OOB_MARGIN) | (by > width + OOB_MARGIN))

    # Nearest player of each team to the ball.
    dist = np.hypot(px - d["bx"][:, None], py - d["by"][:, None])
    dist[~np.isfinite(dist)] = np.inf
    d_home = dist[:, team == 0].min(axis=1)
    d_away = dist[:, team == 1].min(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tempo = _box_mean(np.nanmean(ps, axis=1), TEMPO_SMOOTH)
    n_players = np.isfinite(px).sum(axis=1)

    return {"seen": seen, "out": out, "speed": speed, "glitch": glitch,
            "d_home": d_home, "d_away": d_away, "tempo": tempo, "n_players": n_players,
            "bx": bx, "by": by}


def block_observations(f, start, stop):
    """Discrete observations for the blocks covering frames [start, stop).

    Returns (ball, tempo, control) int arrays of length ceil((stop-start)/BLOCK).
    control is -1 where it is undefined (ball missing or out of bounds).
    """
    n_blocks = -(-(stop - start) // BLOCK)
    ball = np.zeros(n_blocks, dtype=np.int8)
    tempo = np.zeros(n_blocks, dtype=np.int8)
    control = np.full(n_blocks, -1, dtype=np.int8)

    for b in range(n_blocks):
        lo = start + b * BLOCK
        hi = min(stop, lo + BLOCK)
        seen = f["seen"][lo:hi]

        t = f["tempo"][lo:hi]
        t = t[np.isfinite(t)]
        tempo[b] = int(np.searchsorted(TEMPO_EDGES, t.mean())) if len(t) else 0

        if seen.sum() * 2 < (hi - lo):          # ball missing on most of the block
            ball[b] = 0
            continue
        if f["out"][lo:hi][seen].mean() >= 0.5:
            ball[b] = 1
            continue
        s = f["speed"][lo:hi][seen]
        s = s[np.isfinite(s)]
        v = np.median(s) if len(s) else 0.0
        ball[b] = 2 if v < STILL_SPEED else (3 if v < FAST_SPEED else 4)

        dh = f["d_home"][lo:hi][seen]
        da = f["d_away"][lo:hi][seen]
        home = (dh < CONTROL_RADIUS) & (da > dh + CONTROL_MARGIN)
        away = (da < CONTROL_RADIUS) & (dh > da + CONTROL_MARGIN)
        if home.sum() > away.sum() and home.mean() >= 0.5:
            control[b] = 1
        elif away.sum() > home.sum() and away.mean() >= 0.5:
            control[b] = 2
        else:
            control[b] = 0
    return ball, tempo, control
