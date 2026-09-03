# Project: hashing_101

> **Source of truth:** [`.llm/PRD_hashing_101.md`](.llm/PRD_hashing_101.md). This file is the working
> contract derived from it. If they disagree, the PRD wins — update this file to match.

## 1. Identity

**Hashing 101: How to Remember a Billion Things in a Megabyte and Be Wrong on Purpose.**

An educational "101" portfolio project for a data science audience: hashing and probabilistic data
structures, every one built from scratch on CPU. 6 notebooks + `src/` library + Streamlit app.

**Build-up (each chapter depends on the previous):**
what a hash really is → hash tables → Bloom filters → HyperLogLog + Count-Min →
MinHash/LSH → feature hashing + decision guide.

**Thesis to defend with measurements, not prose:** a Bloom filter tracks a million items in ~1.2 MB
(9.6 bits each at 1%) and never returns a false negative; HyperLogLog counts hundreds of thousands
of uniques to ~2% using 1.5 KB. Both come from the same primitive the reader already trusts every
time they write `my_dict[key]`. Trading a *controllable, quantifiable* error for orders-of-magnitude
less memory is a tool, not a compromise.

**One correction to the PRD's framing, load-bearing everywhere:** "a billion things in a megabyte"
is the title, not the arithmetic. At 9.6 bits per item a billion items need ~1.2 GB of bits — still
133x less than the ~160 GB an exact `set` would need, and the difference between one machine and a
cluster, but it is *gigabytes*. A megabyte holds a million items. Notebooks and prose must use the
honest number; `bloom_memory_projection` returns it with `measured=False` because a billion items
are never allocated, only computed.

**Audience:** data scientists who use `dict`/`set`/`groupby` daily and have never looked inside a
hash table. Frame everything as a *problem* (dedup a training set, count unique users, encode
high-cardinality categoricals, find near-duplicate documents), never as an internals tour.

## 2. Current state

Complete and green: library, tests, config, app, `.github/workflows/ci.yml`, and all six notebooks
with their committed outputs. `make ci` passes (205 tests, ~10 s); `make notebooks` executes all six
in ~60 s and rewrites `outputs/`. **Not created:** `.devcontainer/` (the user asked to leave it
alone).

`src/` is installed editable (hatchling, `packages = ["src"]`), so `from src.core... import` works
from any directory, including notebooks run out of `notebooks/`.

Three deviations from PRD §4.1, all deliberate: `src/data.py` holds the generators PRD §4.3
requires (the layout names no home for them), `src/results.py` writes every notebook number to
`outputs/results/*.json`, and `tests/` has one file per module rather than the five the PRD lists.

## 3. Stack and environment

- Python **3.11+**. Package manager **`uv`** — never `pip install`, never `python -m venv`.
- Runtime deps: `numpy`, `scipy`, `mmh3`, `pandas`, `matplotlib`, `plotly`, `streamlit`, `pydantic`,
  `pydantic-settings`, `pyyaml`.
- Dev deps: `pytest`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.
- Target box: **GitHub Codespace, 2 CPU / 8 GB RAM, no GPU.** Everything must run there.
  No multiprocessing pools, no cluster, no CUDA, no dataset downloads.
- Adding a dependency needs a reason the PRD list does not already cover. Prefer stdlib + numpy.

## 4. Commands

```
make setup      # first-time: install uv, uv sync --all-extras, register ipykernel
make sync       # uv sync --all-extras
make lint       # format + check + typecheck
make format     # uv run ruff format src/ tests/ app/
make check      # uv run ruff check --fix src/ tests/ app/
make typecheck  # uv run ty check src/
make test       # uv run pytest
make test-cov   # pytest --cov=src --cov-report=term-missing
make notebooks  # execute all notebooks IN PLACE via nbconvert, 240s timeout each (~60 s total)
make run        # streamlit run app/streamlit_app.py --server.port 8501
make lab        # jupyter lab --no-browser --port 8888
make clean      # caches;  make reset  also drops .venv
make ci         # sync lint test        make dev  # lint test (fast loop)
```

Run anything Python as `uv run ...`. `make dev` before declaring work done; `make ci` before a push.
`LINT_PATHS := $(wildcard src tests app)` in the Makefile: ruff errors on a path that does not exist,
so the lint targets only pass directories that are actually present.

## 5. Layout

```
src/
  config.py               pydantic-settings loader over config/settings.yaml
  core/
    hashes.py             hash functions + uniformity/avalanche diagnostics
    hashtable.py          from-scratch dict: chaining AND open addressing
  probabilistic/
    bloom.py              Bloom filter, tunable false-positive rate
    counting_bloom.py     counting variant, supports deletion
    hyperloglog.py        cardinality estimator with bias corrections
    count_min.py          Count-Min Sketch, frequency estimation
  similarity/
    minhash.py            MinHash signatures for Jaccard
    lsh.py                banded LSH for near-duplicate lookup
  features/
    feature_hashing.py    the hashing trick for categoricals
  evaluation/
    error_analysis.py     measured error vs memory, per structure
    comparison.py         exact vs approximate: memory, speed, accuracy
  visualisation.py        all plotting; notebooks import, never define, figures
  data.py                 seeded generators with known ground truth (streams, corpora, tables)
  results.py              save_results(): every notebook number to outputs/results/*.json
notebooks/  01_what_a_hash_really_is · 02_hash_tables_from_scratch · 03_bloom_filters
            04_counting_unique_with_hyperloglog · 05_near_duplicates_minhash_lsh
            06_feature_hashing_and_guide
tests/      one file per module, plus test_notebooks.py (artefacts) and test_app.py (AppTest)
app/streamlit_app.py   config/settings.yaml   .github/workflows/ci.yml
outputs/figures/*.png (15)   outputs/results/*.json (35)   -- both committed
```

Rules: `src/` holds all logic; notebooks and the app only orchestrate and narrate. No cross-imports
between sibling packages except `evaluation/` and `visualisation.py`, which may import anything.
Every structure exposes `memory_bytes()` so `comparison.py` can chart it honestly.

## 6. Module contracts

Keep constructors *parameterised by the thing the reader wants to choose*, and derive the rest.

- **`hashes.py`** — `hash64(value, seed, backend=...)` over `mmh3`, with a **pure-Python FNV-1a +
  SplitMix64 fallback** so notebooks run without the wheel (`HAS_MMH3` reports which; PRD §10).
  `hash_many(values, seed)` is the batch path: one Python loop over the C hash, then numpy does the
  arithmetic on the whole array — that is what keeps million-item streams to seconds.
  `HashFamily(k, seed)` derives k indices by double hashing (`h_i = h1 + i*h2`, mod 2^64 then mod m)
  with `.indices(value, m)` and `.indices_many(values, m)`. `leading_zeros(values, width)` is the
  vectorised primitive HyperLogLog counts with. Diagnostics: `chi_square_uniformity(...)`,
  `avalanche_matrix(...)`, and `clumping_hash` as the deliberately bad counter-example.
- **`hashtable.py`** — `ChainingHashTable`, `OpenAddressingHashTable`, same public API
  (`__setitem__`, `__getitem__`, `__contains__`, `__len__`, `load_factor`, `probe_stats`).
  Must expose probe counts; notebook 02 plots lookup cost vs load factor.
- **`bloom.py`** — construct from `(expected_items n, target_fp_rate p)`; derive `m`, `k`.
  Expose `estimated_fp_rate()` and `memory_bytes()`. Back it with a numpy bit array, not a `set`.
- **`counting_bloom.py`** — same, with `remove()`; document counter width and saturation behaviour.
- **`hyperloglog.py`** — construct from precision `p` (registers `m = 2**p`). Implements the
  **small-range (linear counting)** correction and asserts against the theoretical band, never a
  point value. It deliberately has **no large-range correction**: that correction only undoes
  collisions in a 32-bit hash space, and this implementation hashes to 64 bits, so it would take
  ~2^64 distinct items to matter. Do not add one. `memory_bytes()` is what numpy holds (1 byte per
  register); `packed_memory_bytes()` is the honest 6-bits-per-register figure the 1.5 KB claim uses.
- **`count_min.py`** — construct from `(epsilon, delta)`; derive `(w, d)`. `estimate()` never
  underestimates for non-negative counts — that one-sidedness is the testable property.
- **`minhash.py`** — `signature(set) -> np.ndarray[k]`; `estimated_jaccard(sig_a, sig_b)`.
  Permutations are `splitmix64(h(x) XOR salt_i)`, **not** the textbook `(a x + b) mod p` family:
  with multipliers small enough to avoid 64-bit overflow that family degenerates into near-monotone
  maps that all pick the same minimum, which measured a −0.21 bias at J=0.5. Do not "restore" it.
- **`lsh.py`** — banding over signatures: `(b bands, r rows)`, `k = b*r`. Expose the S-curve
  `probability(s)` so notebook 05 can plot threshold tuning, and report candidate-pair counts vs the
  all-pairs baseline.
- **`feature_hashing.py`** — `n_buckets` + `alternate_sign` (the signed-hash trick that makes
  colliding contributions cancel in expectation); `collision_rate(tokens)` and
  `expected_collision_rate(n_tokens, n_buckets)` so measurement and balls-in-bins theory can be
  plotted together, and `colliding_groups(...)` to name the categories that share a column.
- **`data.py`** — `generate_stream` (exact unique count, Zipf or uniform), `generate_unique_items`,
  `generate_text_corpus` (returns the planted `duplicate_pairs` as ground truth),
  `generate_categorical_table`. Every one takes a seed; nothing is downloaded.
- **`results.py`** — `save_results(payload, name)` writes `outputs/results/<name>.json`, cleaning
  DataFrames, numpy scalars, dataclasses and sets, and turning non-finite floats into `null` so the
  output is strict JSON. `load_results(name)` reads it back.
- **`visualisation.py`** — every figure, returning bare `Figure` objects (no pyplot state);
  `save_figure(figure, name)` writes `outputs/figures/<name>.png` at the configured dpi.

**Invariant across every structure with a batch path** (`add_many`, `contains_many`,
`indices_many`, `estimate_many`): the batched numpy route must produce *byte-identical* state and
answers to the scalar loop. Each is asserted in tests; an optimisation that quietly diverges is a
bug, not a speedup.

## 7. The math these implementations must obey

Assert these in tests and re-derive them (briefly) in the notebooks — the reader should see where
every constant comes from.

| Structure | Sizing | Error |
|---|---|---|
| Bloom | `m = -n·ln(p) / (ln 2)²`, `k = (m/n)·ln 2` | `fp ≈ (1 - e^(-kn/m))^k`; **false negatives: exactly 0, always** |
| HyperLogLog | `m = 2^p` registers, 6 bits each; `p=11 → m=2048 → 1.5 KB` | standard error `≈ 1.04/√m` (`p=11 → ~2.3%`, `p=12 → ~1.6%`); `α_m = 0.7213/(1 + 1.079/m)` for `m ≥ 128`, else 0.673/0.697/0.709 for m=16/32/64 |
| Count-Min | `w = ⌈e/ε⌉`, `d = ⌈ln(1/δ)⌉` | estimate ≥ truth always; overshoot `≤ ε·N` with probability `1-δ` |
| MinHash | signature length `k` | `P(min_h(A)=min_h(B)) = J(A,B)`; std error `≈ √(J(1-J)/k)` |
| LSH | `k = b·r` | `P(candidate | sim s) = 1 - (1 - s^r)^b`; threshold `≈ (1/b)^(1/r)` |
| Feature hashing | `n_buckets` | collision probability rises as cardinality/`n_buckets`; signed hashing zeroes the collision bias in expectation |

Defaults that these formulas fix, and that other code depends on: `p = 11` for HyperLogLog
(1,536 bytes, 2.3%), and `b = 32, r = 4` for LSH — a knee at J ≈ 0.42, chosen because the planted
near-duplicates sit at J ≈ 0.73 and the 16x8 split (knee 0.71) recovers only 60% of them.

**The one guarantee that is never allowed to soften:** a Bloom filter (and the counting variant,
absent deletions) returns `False` only for items definitely not added. Any refactor that could
introduce a false negative is a bug, not a tradeoff.

## 8. Data

Everything is **generated on the fly** — no downloads. That is the point: known ground truth at
controllable scale. A stream generator yields item streams with a **known unique count and known
frequency distribution** (Zipf for heavy hitters), so every approximate answer is scored against
exact truth: Bloom's measured FP rate vs its target, HLL's estimate vs real cardinality, Count-Min's
estimates vs real counts, MinHash's similarity vs true Jaccard. Plus a small synthetic text corpus
(with planted near-duplicates) for notebook 05 and a high-cardinality categorical table for 06.

Every generator takes an explicit `seed` from `config/settings.yaml`. No unseeded randomness anywhere
in `src/`, tests, or notebooks.

**Coupled defaults, calibrated together:** `corpus.edit_fraction = 0.04` puts planted duplicates at
J ≈ 0.73 with 5-word shingles, which is what makes the `lsh` banding (32x4, knee 0.42) recover all
of them. Raising the edit fraction or narrowing the banding breaks LSH recall tests — change the two
together, and re-measure.

## 9. Configuration

All tunables (bucket counts, hash counts, error targets, precisions, stream sizes, seeds) live in
`config/settings.yaml`, loaded through `src/config.py` with `pydantic-settings`. No magic numbers in
`src/`, notebooks, or the app — read them from config so a reader can change scale in one place, and
so notebook runtime can be dialled down without editing code. `extra = "forbid"`, so a key in the
YAML with no model field fails loudly. Environment variables override the file with the
`HASHING101_` prefix (`HASHING101_SEED=1`). `figures.directory` and `results.directory` decide where
notebook artefacts land.

## 10. Tests

`pytest`, `testpaths = ["tests"]`, `addopts = "-v --tb=short"`. 205 tests, ~10 s. Tests are the
proof that the headline claims are true, so they are written as claims. Current inventory:

| File | The claim it defends |
|---|---|
| `test_hashes.py` | chi-square uniformity passes and avalanche ≈ 50%, for both backends; the bad hash fails both; scalar and batch index paths agree exactly |
| `test_hashtable.py` | behaves like a real `dict` under random operations; tombstones keep deleted probe chains intact; probe cost explodes near a full table |
| `test_bloom.py` | **zero false negatives across the whole stream** (the flagship), also when overfilled; measured FP within sampling error of theory; `memory_bytes()` matches derived `m` |
| `test_counting_bloom.py` | removal is exact, saturated counters never decrement, removing an unknown item raises |
| `test_hyperloglog.py` | estimate inside the `1.04/√m` band at several precisions; small-range path; duplicates change nothing; merge equals the union |
| `test_count_min.py` | never underestimates; overshoot inside `ε·N` for all but a `δ` fraction |
| `test_minhash.py` | estimated Jaccard within `4·sqrt(J(1-J)/k)` of truth across similarities and seeds |
| `test_lsh.py` | recall on planted duplicates ≥ 0.85 at a work ratio < 2%; empirical candidate rate matches the S-curve |
| `test_feature_hashing.py` | shape and determinism; measured collision rate tracks balls-in-bins |
| `test_data.py` | generated ground truth really is exact: cardinality, heavy tails, planted pairs genuinely similar |
| `test_evaluation.py` | every harness function returns the columns the notebooks plot, with its invariants intact |
| `test_visualisation.py`, `test_results.py`, `test_config.py` | figures build and save; results are strict JSON with no `NaN`; settings load and stay coherent |
| `test_app.py` | the Streamlit app runs end to end under `AppTest` with no exceptions |
| `test_notebooks.py` | six notebooks exist, executed, error-free; every figure and result they claim to save exists on disk; no stale artefacts |

Statistical tests must be **seeded and given a tolerance derived from the theory** (4σ of the
estimator, or the sampling error of the measurement itself), never a magic threshold tuned until
green. Keep the suite fast — seconds, not minutes.

## 11. Notebooks

- Hard budget: **each notebook executes in under 4 minutes** on 2 CPU (`make notebooks` enforces a
  240 s timeout). Currently 7–17 s each, ~60 s for all six. If a demo is too slow, shrink the stream
  via config — don't special-case the code.
- Start every notebook with `%matplotlib inline`. `visualisation.py` builds bare `Figure` objects
  rather than going through pyplot, and without the inline backend active they do not render.
- Narrative first: intuition → build → measure → figure. The coin-flip story comes *before* the HLL
  formula; the memory-vs-error curve is the figure that sells notebook 03.
- Import from `src/`; do not define structures or plotting functions inline.
- Commit executed notebooks with outputs (figures are the deliverable). **Every figure goes to
  `outputs/figures/*.png` via `visualisation.save_figure`, and every number, table and takeaway to
  `outputs/results/*.json` via `results.save_results`** - so README quote files
  rather than restating prose. `tests/test_notebooks.py` fails if a notebook claims an artefact it
  did not write, or if a stale artefact has no notebook claiming it.
- Result JSON carries no timestamps, so a rerun of the same seed rewrites the same statistics;
  only files holding wall-clock timings move. A diff in anything else means a *result* changed.
- A billion items is **not** run literally: measure at millions, validate the theory at that scale,
  extrapolate analytically, and say plainly in the text that that is what is happening (PRD §10).

## 12. Streamlit app

Four panels, all interactive and instant on 2 CPU: (1) Bloom playground — add/query, watch false
positives appear as it fills, tune size to a target rate; (2) HyperLogLog counter — stream uniques,
watch the KB-sized estimate track truth with an error band; (3) near-duplicate finder — paste texts,
MinHash+LSH groups them; (4) collision visualiser — hash into a small table, watch collisions pile up
with load factor. Cache heavy state with `@st.cache_data` / `@st.cache_resource`; keep every default
small enough to respond in under a second. `tests/test_app.py` runs the whole script under
`AppTest`, which resolves relative paths against the *calling* file — pass it an absolute path built
from `PROJECT_ROOT`.

## 13. Conventions

- `ruff` with `target-version = "py311"`, `line-length = 99`, lint select
  `["E","F","W","I","UP","N","B","A","SIM","PTH"]`, ignore `["E501"]`. `pathlib` over `os.path` (PTH).
  `N803`/`N806` are off in `src/` and `tests/` so the papers' notation (`m`, `k`, `b`, `r`, `w`, `d`,
  `p`) survives; `notebooks/` and `outputs/` are excluded from ruff entirely.
- Type hints on every public function; `ty check src/` must be clean.
- Docstrings on public APIs: what it does, the formula it implements, and the paper it comes from.
- British spelling in `visualisation.py` (the filename is authoritative); otherwise match local style.
- Comments explain *why the math is that way*, not what the line does.

## 14. Definition of done

`make ci` clean (lint + 205 tests) and `make notebooks` green, with the regenerated
`outputs/figures/*.png` and `outputs/results/*.json` committed alongside the executed notebooks.
Every success criterion in PRD §8 measured and asserted; memory savings vs the exact baseline
reported as numbers, and any number quoted in prose readable from a JSON file in `outputs/results/`.

## 15. Non-goals

No GPU. No distributed anything. No downloaded datasets. No third-party implementation of a structure
this project teaches (`pybloom`, `datasketch`, `sklearn.FeatureHasher` are for *comparison only*, if
used at all). No new chapters beyond the six — the future extensions in PRD §11 (consistent hashing,
cuckoo filters, SimHash, streaming quantiles, hashing for privacy) stay out of scope. `.devcontainer/`
is off limits — the user asked for it to be left alone.

## 16. References

Bloom 1970 (CACM) · Flajolet et al. 2007 (HyperLogLog, AofA) · Broder 1997 (MinHash) ·
Cormode & Muthukrishnan 2005 (Count-Min) · Weinberger et al. 2009 (Feature Hashing, ICML) ·
Indyk & Motwani 1998 (LSH, STOC).

Sibling projects to cross-link, not duplicate: `information_theory_101`, `causal_ml_101`,
`retrieval_arena` (MinHash/LSH as the near-duplicate layer under retrieval), `encoding_arena`
(feature hashing benchmarked as an encoder — this project builds it from scratch instead).

## 17. Token rules

Caveman mode: ON. Code-first, minimal prose in responses. `/compact` at notebook boundaries.
