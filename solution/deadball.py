"""Stage A: is the ball alive or dead?

A two-state hidden Markov model (alive / dead) over the 0.2 s blocks of features.py, using two
observation channels: the ball symbol and the tempo (mean player speed) bin.

Where the likelihood tables come from -- cross-view estimation, no labels
-------------------------------------------------------------------------
Plain Baum-Welch on this model converges to a useless optimum ("dead" == "ball missing"), so the
tables are instead estimated the way co-training does it: each channel is measured on blocks the
OTHER channel is confident about.

    P(tempo | dead)   measured where the ball channel is sure: ball out of bounds, or deep inside
                      a long stretch with no ball at all
    P(tempo | alive)  measured where the ball is travelling fast (a pass or a shot)
    P(ball  | dead)   measured where the tempo channel is sure: everybody walking for seconds
    P(ball  | alive)  measured where everybody has been running for seconds

Nothing here reads a ground truth and no person decides any label: the anchors are rules.

The transition probabilities are a prior from general football knowledge, not from this match: a
professional game has roughly 55-60 minutes of effective play broken by about a hundred stoppages,
i.e. live stretches of ~30 s and stoppages of ~20 s.
"""

import numpy as np

import features

ALIVE, DEAD = 0, 1

# Params
BLOCKS_PER_S = 25 // features.BLOCK
MEAN_ALIVE_S = 33.0            # prior mean length of a live stretch
MEAN_DEAD_S = 22.0             # prior mean length of a stoppage
P_DIE = 1.0 / (MEAN_ALIVE_S * BLOCKS_PER_S)
P_RESTART = 1.0 / (MEAN_DEAD_S * BLOCKS_PER_S)

SURE_SLOW_BIN = 1              # tempo bin <= this (under 1.2 m/s) is "everybody walking"
SURE_FAST_BIN = 4              # tempo bin >= this (over 2.0 m/s) is "everybody running"
SURE_SLOW_HALF = 2 * BLOCKS_PER_S   # ... sustained over +/- 2 s
SURE_FAST_HALF = 1 * BLOCKS_PER_S   # ... sustained over +/- 1 s
LONG_GAP_HALF = 12             # blocks: "deep inside" a ball gap means +/- 2.4 s of no ball

# The tempo channel is a 25-frame moving average sampled every 5 frames, so neighbouring blocks
# repeat most of the same evidence. Weighting its log-likelihood down corrects the double count.
W_TEMPO = 1.0
W_BALL = 1.0

# A ball tracked beyond the boundary line is out of play by the Laws of the Game, whatever the
# players are doing, so that observation rules the "alive" state out instead of merely voting.
OOB_IS_DEAD = True
IMPOSSIBLE = -1e6

MIN_ALIVE_BLOCKS = 2 * BLOCKS_PER_S   # a "live" blip shorter than 2 s inside a stoppage is noise
MIN_DEAD_BLOCKS = 2 * BLOCKS_PER_S    # no real stoppage lasts under 2 s


def _sustained(mask, half):
    """True where `mask` holds on every block within +/- half."""
    c = np.concatenate(([0], np.cumsum(mask)))
    i = np.arange(len(mask))
    lo, hi = np.maximum(0, i - half), np.minimum(len(mask), i + half + 1)
    return (c[hi] - c[lo]) == (hi - lo)


def estimate_tables(sequences):
    """Cross-view likelihood tables. Returns dict of four probability vectors."""
    ball_dead = np.zeros(features.N_BALL)
    ball_alive = np.zeros(features.N_BALL)
    tempo_dead = np.zeros(features.N_TEMPO)
    tempo_alive = np.zeros(features.N_TEMPO)
    for ball, tempo, _ in sequences:
        tempo_says_dead = _sustained(tempo <= SURE_SLOW_BIN, SURE_SLOW_HALF)
        tempo_says_alive = _sustained(tempo >= SURE_FAST_BIN, SURE_FAST_HALF)
        ball_says_dead = (ball == 1) | ((ball == 0) & _sustained(ball == 0, LONG_GAP_HALF))
        ball_says_alive = ball == 4

        ball_dead += np.bincount(ball[tempo_says_dead], minlength=features.N_BALL)
        ball_alive += np.bincount(ball[tempo_says_alive], minlength=features.N_BALL)
        tempo_dead += np.bincount(tempo[ball_says_dead], minlength=features.N_TEMPO)
        tempo_alive += np.bincount(tempo[ball_says_alive], minlength=features.N_TEMPO)

    def normalise(counts):       # add-one smoothing: no symbol is ever impossible
        return (counts + 1.0) / (counts + 1.0).sum()

    return {"ball_dead": normalise(ball_dead), "ball_alive": normalise(ball_alive),
            "tempo_dead": normalise(tempo_dead), "tempo_alive": normalise(tempo_alive)}


def log_emissions(tables, ball, tempo, w_tempo=W_TEMPO, w_ball=W_BALL, oob_is_dead=OOB_IS_DEAD):
    e = np.empty((len(ball), 2))
    e[:, ALIVE] = (w_ball * np.log(tables["ball_alive"][ball]) +
                   w_tempo * np.log(tables["tempo_alive"][tempo]))
    e[:, DEAD] = (w_ball * np.log(tables["ball_dead"][ball]) +
                  w_tempo * np.log(tables["tempo_dead"][tempo]))
    if oob_is_dead:
        e[ball == 1, ALIVE] = IMPOSSIBLE
    return e


def viterbi(log_e, log_t, log_pi):
    n, k = log_e.shape
    score = log_pi + log_e[0]
    back = np.zeros((n, k), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] + log_t
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + log_e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs_of(mask):
    """[(start, stop)] maximal True runs of a boolean array."""
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def remove_blips(dead, min_alive=MIN_ALIVE_BLOCKS, min_dead=MIN_DEAD_BLOCKS):
    """Drop live stretches and then stoppages that are too short to be real (interior only)."""
    dead = dead.copy()
    for a, b in runs_of(dead == ALIVE):
        if b - a < min_alive and a > 0 and b < len(dead):
            dead[a:b] = DEAD
    for a, b in runs_of(dead == DEAD):
        if b - a < min_dead and a > 0 and b < len(dead):
            dead[a:b] = ALIVE
    return dead


def segment(tables, ball, tempo, w_tempo=W_TEMPO, w_ball=W_BALL,
            p_die=P_DIE, p_restart=P_RESTART, oob_is_dead=OOB_IS_DEAD, dead_until=0):
    """Per-block ALIVE / DEAD for one half. Every half starts dead (players lined up).

    dead_until: blocks before this index cannot be alive -- the half has not kicked off yet.
    """
    log_t = np.log(np.array([[1 - p_die, p_die], [p_restart, 1 - p_restart]]))
    log_pi = np.log(np.array([0.1, 0.9]))
    log_e = log_emissions(tables, ball, tempo, w_tempo, w_ball, oob_is_dead)
    log_e[:max(0, dead_until), ALIVE] = IMPOSSIBLE
    return remove_blips(viterbi(log_e, log_t, log_pi))
