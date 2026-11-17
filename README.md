# Team Isaac

Submission for **More than a Hack 2027, Phase 1: Ball Possession** (FC Barcelona vs Athletic Club).

This folder holds everything needed to go from the provided data bundle to the submitted file,
`Isaac.csv`. Run it and you get **the same file, byte for byte**.

How the method works, and why: [Isaac.md](Isaac.md) (also as `Isaac.pdf`).

## What you need

- Python **3.14.3** (the version the file was made with)
- numpy **2.3.4** (pinned in `requirements.txt`; nothing else is used)
- the competition data bundle. Only these three files of it are read:
  - `data/tactical/BAR-ATH_tactical_atd.xml`
  - `data/tactical/BAR-ATH_tactical_atd.txt`
  - `data/tactical/BAR-ATH_tactical_smart_tagging.xml`

No GPU, no network, no randomness.

## Install

```bash
pip install -r requirements.txt
```

Unpack the data bundle into `./data`, so that `./data/tactical/` holds the files above. To keep
it somewhere else, point `MTAH_DATA_DIR` at the folder that contains `tactical/`:

```bash
export MTAH_DATA_DIR=/path/to/data
```

```powershell
$env:MTAH_DATA_DIR = "D:\path\to\data"
```

## Make the file

This is the command that regenerates the submission:

```bash
python solution/make_submission.py
```

It writes `output/Isaac.csv` (199,319 rows plus a header, `frame,label`, LF line endings). The
first run reads the raw tracking and caches it in `solution/cache/`, which takes about a minute;
after that a run takes about ten seconds.

## Check it is the same file

```bash
python -c "import hashlib; print(hashlib.sha256(open('output/Isaac.csv','rb').read()).hexdigest())"
```

It must print:

```
3b4e03355d4d31792921a1620c0c112717ac312c391b1676c16d1e8cf8e8e87a
```

The script also prints the label counts, which must be H 51835, A 30805, D 76217, X 40462.

To prove the cache plays no part, delete `solution/cache/` and run again: the hash is the same.

## What is in the folder

| path | what it is |
|---|---|
| `solution/make_submission.py` | runs everything and writes the CSV |
| `solution/io_atd.py` | reads the tactical tracking into arrays, in metres |
| `solution/features.py` | turns frames into 0.2 s blocks: ball state, player tempo, who is on the ball |
| `solution/deadball.py` | step 1: is the ball alive or dead (two-state hidden Markov model) |
| `solution/possession.py` | step 2: which team has it (the controlled-possession rule) |
| `solution/load_smart_tagging.py` | reads the Smart Tagging file (from the competition starter kit); only the kick-off tags are used |
| `games.py` | match constants: frame offsets, halves, pitch size, data paths (from the competition repository) |
| `requirements.txt` | pinned dependency |
| `Isaac.md`, `Isaac.pdf` | technical document |
| `LICENSE` | licence of the competition code reused here (`games.py`, `load_smart_tagging.py`) |

Not committed (see `.gitignore`): `data/` (the gated match data), `output/` and `solution/cache/`
(derived from it), and `trials/` (experiments and development work, not needed to make the file).
