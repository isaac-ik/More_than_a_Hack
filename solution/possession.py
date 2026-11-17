"""Stage B: inside a live stretch, which team has the ball?

An explicit implementation of the possession rules in CHALLENGE.md:

  * after a restart there is a clear possessor: the first team seen on the ball;
  * while the ball is loose, contested or not tracked, possession stays with the last team;
  * possession changes only on CONTROLLED possession by the other team. "Controlled" is either
      - one uninterrupted spell on the ball of at least HOLD_S seconds, or
      - two consecutive spells on the ball by that team with no touch by the old owner in
        between (a completed pass, or a second touch after a duel);
    a single short touch -- a deflection, a block, a header won and immediately lost -- is not.
  * the change is dated to the start of the first of those spells.

A "spell on the ball" (control episode) is a run of blocks in which a player of that team is within
CONTROL_RADIUS of the ball and clearly closer than any opponent (features.py), tolerating gaps of
up to GAP_S.
"""

import numpy as np

import features
from deadball import ALIVE, runs_of

HOME, AWAY, DEAD_CODE = 0, 1, 2

# Params -- seconds, chosen from the wording of the rules, not fitted
BLOCKS_PER_S = 25 // features.BLOCK
HOLD_S = 1.0          # one continuous spell this long is control on its own
GAP_S = 0.4           # a break this short inside a spell is still the same spell
SECOND_TOUCH_S = 5.0  # the confirming second spell must start within this long of the first


def control_episodes(control, lo, hi, gap):
    """[[start, stop, team]] inside blocks [lo, hi); team is 0 home / 1 away."""
    out = []
    for t in range(lo, hi):
        c = control[t]
        if c <= 0:
            continue
        if out and out[-1][2] == c - 1 and t - out[-1][1] <= gap:
            out[-1][1] = t + 1
        else:
            out.append([t, t + 1, c - 1])
    return out


def label_half(control, dead, hold_s=HOLD_S, gap_s=GAP_S, second_touch_s=SECOND_TOUCH_S):
    """Per-block labels 0 home / 1 away / 2 dead for one half."""
    hold = int(round(hold_s * BLOCKS_PER_S))
    gap = int(round(gap_s * BLOCKS_PER_S))
    second = int(round(second_touch_s * BLOCKS_PER_S))

    labels = np.full(len(control), DEAD_CODE, dtype=np.int8)
    stretches = runs_of(dead == ALIVE)
    pending = []              # live stretches with no touch at all: decided once a neighbour is
    last_owner = None
    for lo, hi in stretches:
        episodes = control_episodes(control, lo, hi, gap)
        if not episodes:
            if last_owner is None:
                pending.append((lo, hi))
            else:
                labels[lo:hi] = last_owner      # nobody seen on the ball: possession is kept
            continue

        owner = episodes[0][2]
        for a, b in pending:                    # head-of-half stretches inherit backwards
            labels[a:b] = owner
        pending = []

        cursor = lo
        for k in range(1, len(episodes)):
            start, stop, team = episodes[k]
            if team == owner:
                continue
            held = (stop - start) >= hold
            nxt = episodes[k + 1] if k + 1 < len(episodes) else None
            confirmed = nxt is not None and nxt[2] == team and nxt[0] - stop <= second
            if held or confirmed:
                labels[cursor:start] = owner
                owner, cursor = team, start
        labels[cursor:hi] = owner
        last_owner = owner
    for a, b in pending:                        # a half with no touch anywhere (never happens)
        labels[a:b] = HOME
    return labels
