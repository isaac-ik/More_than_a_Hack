# Technical document: who has the ball

More than a Hack 2027, Phase 1: Ball Possession.
Match: FC Barcelona (home, **H**) vs Athletic Club (away, **A**).
Team: **Isaac**. Submitted file: `Isaac.csv`.

## In short

For every frame of the tactical video we say **H** (home has the ball), **A** (away has the
ball), **D** (the ball is dead: play is stopped) or **X** (not graded).

We do it in two steps, using only the tactical tracking data:

1. **Is the ball alive or dead?** We look at the ball and at how fast the players are moving.
2. **If it is alive, which team has it?** We look at which player is on the ball, and change
   team only when the other team clearly takes control.

No labels are used to train anything. Every setting is a plain football number (metres,
seconds, metres per second), written at the top of the code file that uses it.

| | spellF1 | macroF1 | possessions found |
|---|---|---|---|
| **our submission**, public 5-minute check | **0.494** | **0.794** | 17 (true: 17) |
| `baseline_smooth50`, same check | 0.326 | 0.621 | 26 |

The public check only has 17 possessions, so it tells us the code works, not how well it will
score on the full match. **We do not know our full-match score.**

---

## 1. The approach

### What we found in the data first

Three things in the tracking data decided the design:

- **The ball goes missing a lot, and in long blocks.** On 41% of the graded frames there is
  no ball. These missing frames come in only 150 gaps, the middle one about 11 seconds long.
  You cannot just draw a line across a gap that long.
- **When the ball is missing, players are usually walking.** Average player speed is about
  1.0 m/s with no ball, against 2.0 m/s with a ball. So most gaps are stoppages, but not all of
  them. We need the players' speed to decide, not the ball alone.
- **"Nearest player to the ball" is only a good team guess when the player is really on the
  ball.** On the public 5 minutes it named the right team 89% of the time when the player was
  within 1.5 m, and much less often for loose or high balls. So we only trust it when a player is close, and in between we keep the last
  team.

The score (spellF1) counts whole possessions. So we aimed to get the **right number of
possessions with roughly the right start and end**, not to be perfect on every single frame.

### Step 1: alive or dead (`solution/deadball.py`)

We cut the match into small blocks of 5 frames (0.2 seconds). For each block we note:

- **the ball:** missing, out of play, still (under 1 m/s), slow (1 to 6 m/s) or fast
  (over 6 m/s);
- **the tempo:** the average speed of all players, put into 6 bands.

A small two-state model (alive / dead) reads these blocks in order and picks the most likely
alive/dead story for the whole half. This is a hidden Markov model decoded with Viterbi.

The model needs to know "how does the ball look when play is alive, and when it is dead?"
We have no labels, so each signal teaches the other:

- when the ball is clearly flying fast, play must be alive, so we learn what the tempo looks
  like when alive;
- when every player has been walking for a few seconds, play is very likely dead, so we learn
  what the ball looks like when dead;
- and the same the other way round.

We also tell the model that play usually lasts about 33 seconds and a stoppage about 22 seconds.
This comes from general football knowledge, not from this match.

Three fixed rules sit on top:

- **A ball over the line is dead.** No exceptions.
- **Nothing is alive before kick-off.** The kick-off moment is read from the Smart Tagging file
  (the `Kick Off (Start)` tag). This makes the start of the second half, before the kick-off,
  correctly **D**.
- **No flicker.** A stoppage or a burst of play shorter than 2 seconds is removed.

### Step 2: which team (`solution/possession.py`)

A player is **on the ball** when they are within **1.5 m** of it and the nearest opponent is at
least **0.5 m** further away. A run of blocks with the same team on the ball is one **spell on
the ball** (breaks under 0.4 s are ignored).

Inside each stretch of live play:

- the first team on the ball after a restart owns it;
- the other team takes over only with **controlled possession** (see section 4);
- otherwise the last team keeps the ball.

### Why this and not something else

- **We tried one big model for H, A and D together** (a three-state model that learns
  everything by itself). It went wrong: "dead" just became "ball missing", and it found 594
  possessions, worse than the simple baseline. Splitting the job into two smaller questions
  fixed that.
- **We did not train a model on labels**, because the rules forbid labels from this match and
  we did not have time to collect and clean labelled matches from elsewhere.
- **We did not build our own ball detector on the video.** It would help most (see section 5),
  but it was too big a job for the time we had.

---

## 2. Which assets we used, and how

| asset | used? | how, or why not |
|---|---|---|
| **tactical tracking (ATD)**: players and ball | **yes, the main input** | everything in steps 1 and 2 comes from it |
| **Smart Tagging** | **yes, a little** | only the start frame of the two `Kick Off (Start)` tags. We also used the tags to compare versions of our method without labels (section 6) |
| tactical homography | not for the labels | the tracking is already in pitch metres. We used it only to draw our results on the video |
| tactical video | not for the labels | only to make a video showing our predictions |
| **broadcast tracking (ATD)** | **tried, dropped** | see below |
| broadcast video and homography | no | |
| public 5-minute ground truth | only to check the code | never used to train or to pick settings |

### What we tried and dropped

**Broadcast tracking, to find the hidden ball.** We lined up the broadcast feed with the tactical
feed in time (one fixed shift per half: about 4,014 frames in the first half, 26,920 in the
second). We fitted a unifying homography between the two (it came out almost the same as doing
nothing: the two feeds already agree to about 0.3 m). We threw away replays and close-ups by
checking that the broadcast players sit on top of the tactical players.

The result was small. Both feeds lose the ball at mostly the same moments, so the broadcast feed
only gave back the ball on about 9% of the missing frames. Ball coverage went from 59.3% to
63.0%, and the labels changed on 1.78% of graded frames. Not worth the extra moving parts.

**Active control from deliberate direction change.** Instead of "a player is close", we tried
"a player did something to the ball":

- a **touch** is a sudden change in the ball's speed or direction (at least 4 m/s) with a player
  within 2 m;
- a **carry** is when the ball slows down near a player's feet and moves at the same pace as
  that player for at least half a second;
- the other team takes over with a carry, or with a touch followed by a **follow-up action**
  (another touch or carry by them) within 5 seconds.

It found 323 possessions instead of 291. It changed 5.94% of the graded frames (all H/A swaps,
no change to D). On the public check spellF1 went up a little (0.511) and macroF1 went down a
little (0.781). On our label-free check against the Smart Tagging it agreed slightly less
(0.790 vs 0.807). Because it was not clearly better, we kept the simpler version.

---

## 3. Training data

**None.**

- No labelled dataset of any kind: not from this game, not from any other game.
- No pre-trained model.
- No hand labels.

The only numbers learned from data are four small probability tables in step 1, and they are
learned from the **unlabelled** tracking of this match, using the "each signal teaches the other"
trick above. No person decided any label.

---

## 4. The possession rules we implemented

| situation | what our code does |
|---|---|
| **restart** (kick-off, throw-in, free kick, corner, goal kick) | the ball is **D** until play restarts. The first team seen on the ball after the restart owns it from that moment |
| **before kick-off** | always **D**, using the Smart Tagging kick-off tag |
| **ball over the line** | always **D** |
| **loose ball** (rolling, bouncing, nobody close) | the last team **keeps** it |
| **ball in the air** (long ball, cross, clearance) | the last team **keeps** it until someone is on the ball. The tracked ball is unreliable in the air, so we do not guess |
| **ball not tracked** while play is alive | the last team **keeps** it |
| **challenge or duel** (both teams close, opponent within 0.5 m of the player) | nobody is "on the ball", so the last team **keeps** it |
| **one quick touch** by the other team (block, deflection, header won and lost) | **not** a change. Possession stays |
| **controlled possession** by the other team | **a change**: either one spell on the ball of at least **1.0 second**, or **two spells in a row** (the second starting within 5 seconds) with no touch by the old team in between. That second spell is a completed pass or a second touch |
| **when the change happens** | at the start of the first of those spells, not when we became sure |
| **a stretch of play where nobody is ever seen on the ball** | takes the team from before |

We always give H, A or D inside the two halves, and X only outside them.

---

## 5. Limitations

Where it goes wrong:

- **Fouls are spotted late.** A whistle is not in the tracking. We only notice play has stopped
  when players slow down, usually 3 to 8 seconds after the whistle. Ball-out stoppages are
  caught on time.
- **Restarts can start early.** When a player carries the ball to the spot while others jog,
  it looks like live play.
- **Some live play reads as dead.** For example a goalkeeper holding the ball while the tracker
  has lost it and everybody is walking. This splits one possession into two.
- **Very short possessions are missed.** Possessions under about a second cannot pass the
  1-second / two-spell rule. Making the rule looser creates far more false possessions than it
  finds real ones.
- **Changes after long high balls can be 5+ seconds late**, because nobody is "on the ball"
  until it comes down.
- **Only one match.** The settings are football numbers and should carry over, but we could not
  test on another game.

What we would fix with more time:

1. **Find the ball ourselves in the tactical video** with a ball detector, to shrink the 41% of
   frames with no ball. This would help almost every error above.
2. **Learn when play stops from other matches** that have open labels, to catch fouls sooner.
3. **Use team shape when the ball is missing.** The team with the ball spreads out and the team
   without it squeezes together, so player positions alone can hint at who has it.

---

## 6. How we chose the settings

Every setting is a plain number with a football meaning (1.5 m, 1.0 second, 33 seconds of play).
There was no search over settings.

To compare versions without using labels, we used the Smart Tagging over the **whole** match:

- Is play **D** at the start of each set-piece tag? (0.83 for us)
- Is play alive inside open-play tags? (0.913)
- Does our team match the tagged team? (0.807)

The tags are noisy, so none of these can reach 1.0, but a better labelling agrees more often.

To be honest: we could see the public 5-minute score while building this, so we cannot say it had
zero effect. But it did not pick our settings. Two of the variants we tested scored higher on the
public check than the one we submitted, and we did not switch to them.

---

## 7. Reproduction

Needs Python 3.14.3 and numpy 2.3.4. No GPU, no network, no
randomness. It takes about a minute the first time, then about ten seconds.

**Step 1.** Unpack the data bundle into `./data` (or set `MTAH_DATA_DIR`).

**Step 2.** Install:

```bash
pip install -r requirements.txt
```

**Step 3.** Make the file. This is the command that regenerates the submission:

```bash
python solution/make_submission.py
```

It writes `output/Isaac.csv`.

**Step 4.** Check it is the same file:

```bash
python -c "import hashlib; print(hashlib.sha256(open('output/Isaac.csv','rb').read()).hexdigest())"
```

It must print `3b4e03355d4d31792921a1620c0c112717ac312c391b1676c16d1e8cf8e8e87a`. The file comes
out the same byte for byte every time, also after deleting `solution/cache/`. LF line endings.

Counts: H 51,835 · A 30,805 · D 76,217 · X 40,462 (291 possessions, 131 stoppages).

| file | what it does |
|---|---|
| `solution/make_submission.py` | runs everything and writes the CSV |
| `solution/io_atd.py` | reads the tactical tracking into arrays in metres |
| `solution/features.py` | turns frames into 0.2 s blocks: ball state, tempo, who is on the ball |
| `solution/deadball.py` | step 1: alive or dead |
| `solution/possession.py` | step 2: which team |
| `solution/load_smart_tagging.py` | reads the Smart Tagging file (competition starter kit) |
| `games.py` | match constants and data paths (competition repository) |
| `requirements.txt` | pinned dependency: numpy 2.3.4 |

The experiments in sections 2 and 6 (broadcast fusion, active control, the label-free checks) were
development work. They are kept out of the submission folder because they are not needed to make
the file.
