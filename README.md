# hashing_101

**How to remember a billion things in a megabyte and be wrong on purpose.**

A hands-on introduction to the one trick that quietly powers deduplication, feature engineering,
database joins, near-duplicate detection and streaming analytics — every structure built from
scratch, on CPU, inside a 2-core Codespace.

Every dictionary lookup, `groupby`, `set` membership test and join you write is a hash table
underneath. Point that same idea at a slightly harder question and you get a family of structures
that answer things exact methods cannot afford to answer at all — for a controllable,
*quantifiable* error.

## Quickstart

```bash
make setup      # install uv, sync dependencies, register the Jupyter kernel
make test       # 205 tests, ~10 seconds
make notebooks  # execute all six notebooks in place, ~60 seconds
make run        # the Streamlit playground on :8501
make ci         # what CI runs: sync + lint (ruff + ty) + test
```

## The six notebooks

Each one poses a problem a data scientist actually has, builds the structure that solves it, and
measures the result against the exact answer.

| Notebook | The question it answers |
|---|---|
| [01 What a hash really is](notebooks/01_what_a_hash_really_is.ipynb) | What makes a hash usable? Uniformity and avalanche, measured on a good hash and on a deliberately bad one |
| [02 Hash tables from scratch](notebooks/02_hash_tables_from_scratch.ipynb) | The dictionary you use every day: chaining, open addressing, and why lookup cost hits a wall as the table fills |
| [03 Bloom filters](notebooks/03_bloom_filters.ipynb) | Have I seen this item? 9.6 bits each, zero false negatives, and an error rate you pick with a formula |
| [04 Counting unique with HyperLogLog](notebooks/04_counting_unique_with_hyperloglog.ipynb) | How many distinct items? 500,000 uniques counted in 1,536 bytes — plus Count-Min for frequencies |
| [05 Near-duplicates with MinHash and LSH](notebooks/05_near_duplicates_minhash_lsh.ipynb) | Which documents are almost the same? 40 comparisons instead of 79,800 |
| [06 Feature hashing and a decision guide](notebooks/06_feature_hashing_and_guide.ipynb) | How do I encode a million categories? And: which structure, for which question |

## What the numbers say

Every figure below is a real run on a 2-core Codespace, read straight out of
[outputs/results/](outputs/results/) — the notebooks write their numbers to JSON so this table and any later analysis all quote the same execution.

| Question | Exact answer | Approximate answer | Error |
|---|---|---|---|
| Membership over 100,000 items | `set`, 9.69 MB | Bloom filter, **117 KiB** — 81x less | 0.99% false positives, **0 false negatives** |
| Distinct count over 500,000 items | `set`, 44.3 MB | HyperLogLog p=11, **1,536 bytes** — 28,800x less | −0.69%, inside the 2.3% standard error |
| Frequencies over a 200,000-event stream | `Counter`, ~1.5 MB | Count-Min Sketch, **106 KiB** | never underestimates; overshoot bounded by 200 |
| Near-duplicates among 400 documents | 79,800 pair comparisons, 0.58 s | MinHash + LSH, **40 comparisons**, 0.06 s | 40 of 40 planted pairs found |
| Encoding 4,914 categories | one-hot: a vocabulary to build, store and ship | feature hashing, **256 columns**, no vocabulary | R² 0.581 at a 92% collision rate |
| Membership over 1 **billion** items | `set`, ~160 GB (projected) | Bloom filter, **1.2 GB** — 133x less | 1% false positives |

**About that last row.** It is computed from the closed-form sizing plus a measured per-item cost of
a real Python `set` — a projection, and the code labels it as one (`measured: false`). Read what it
says: 9.6 bits per item means a *million* items fit in 1.2 MB and a billion need 1.2 GB. The
megabyte in the title buys you a million things. What the filter actually changes is that 160 GB (a
cluster, a budget, a design doc) becomes 1.2 GB (a process on one laptop), and the price is a 1%
chance of skipping a record.

## Repository map

| Path | What it holds |
|---|---|
| [src/core/hashes.py](src/core/hashes.py) | Hash functions and the two diagnostics that judge one — chi-square uniformity, per-bit avalanche — plus a deliberately bad hash to compare against, and a pure-Python fallback when `mmh3` is absent |
| [src/core/hashtable.py](src/core/hashtable.py) | The dictionary from scratch: chaining and open addressing, counting probes so the cost of a full table is visible |
| [src/probabilistic/bloom.py](src/probabilistic/bloom.py) | Bloom filter sized from `(expected items, target error)`. Never a false negative |
| [src/probabilistic/counting_bloom.py](src/probabilistic/counting_bloom.py) | Counting variant: deletion, for 8x the memory |
| [src/probabilistic/hyperloglog.py](src/probabilistic/hyperloglog.py) | Cardinality estimation from runs of leading zeros, with the small-range correction and lossless merging |
| [src/probabilistic/count_min.py](src/probabilistic/count_min.py) | Count-Min Sketch: frequencies that overestimate, never under |
| [src/similarity/minhash.py](src/similarity/minhash.py) | MinHash signatures — the rate at which two signatures agree *is* their Jaccard similarity |
| [src/similarity/lsh.py](src/similarity/lsh.py) | Banded LSH: quadratic near-duplicate search becomes a dictionary lookup |
| [src/features/feature_hashing.py](src/features/feature_hashing.py) | The hashing trick for high-cardinality categoricals, with collisions made visible |
| [src/evaluation/](src/evaluation/) | Measured error against memory, and exact-versus-approximate comparisons |
| [src/data.py](src/data.py) | Seeded generators with known ground truth: streams with an exact unique count, corpora with planted near-duplicates, categorical tables with real signal |
| [src/visualisation.py](src/visualisation.py) | Every figure in the project, so notebooks stay narrative |
| [src/results.py](src/results.py) | `save_results()` — every notebook number, table and takeaway to strict JSON |
| [src/config.py](src/config.py) | Typed settings over [config/settings.yaml](config/settings.yaml); no magic numbers anywhere else |
| [app/streamlit_app.py](app/streamlit_app.py) | Bloom playground, HyperLogLog counter, near-duplicate finder, collision visualiser |

Nothing is downloaded. Everything is generated with a seed, so every approximation can be scored
against the exact truth at a scale you control.

**Artefacts.** `make notebooks` writes 15 figures to [outputs/figures/](outputs/figures/) as PNG and
35 result files to [outputs/results/](outputs/results/) as strict JSON. Both directories are
committed, so the numbers quoted here can be checked without running anything.

## How it is verified

`make ci` runs `ruff format`, `ruff check`, `ty` and 205 tests in about ten seconds. The tests are
written as claims rather than smoke checks:

- **zero false negatives** across a full stream, at three different error targets, plus the same
  assertion on an overfilled filter;
- measured false-positive, cardinality, frequency and Jaccard errors inside **tolerances derived
  from the theory** (4σ of the estimator, or the sampling error of the measurement) — never a
  threshold tuned until green;
- the batched numpy paths produce byte-identical state to the scalar ones, so the optimisation
  cannot silently change behaviour;
- the from-scratch hash tables are checked against a real `dict` under thousands of random
  operations;
- the Streamlit app is executed end to end via `AppTest`;
- every figure and result file a notebook claims to save actually exists, parses as strict JSON, and
  has no stale artefact left behind.

## Constraints

Python 3.11+, `uv`, no GPU, 2 CPU / 8 GB. Dependencies: numpy, scipy, mmh3, pandas, matplotlib,
plotly, streamlit, pydantic. `mmh3` is optional at runtime — a pure-Python backend takes over if it
is missing, and passes the same uniformity and avalanche tests. Every notebook executes in well
under the 4-minute budget (7–17 s each; ~60 s for all six).

## Notes on a few decisions

- **MinHash permutations are `splitmix64(h(x) XOR salt)`**, not the textbook `(a·x + b) mod p`
  family. With multipliers small enough to avoid 64-bit overflow, that family degenerates into
  near-monotone maps that all select the same minimum; measured across 60 seeds it was biased
  −0.21 at J=0.5. The salted mix is unbiased and its spread matches `sqrt(J(1-J)/k)`.
- **HyperLogLog has no large-range correction.** That correction exists to undo collisions in a
  32-bit hash space; with a 64-bit hash it would need ~2^64 distinct items to matter. The
  small-range (linear counting) path is implemented and tested.
- **Counting Bloom refuses to remove an item it never saw**, and never decrements a saturated
  counter. Both rules exist to protect the no-false-negative guarantee.

## References

Bloom 1970 · Flajolet et al. 2007 (HyperLogLog) · Broder 1997 (MinHash) · Cormode &
Muthukrishnan 2005 (Count-Min) · Weinberger et al. 2009 (feature hashing) · Indyk & Motwani 1998
(LSH). Licensed Apache-2.0.
