
### **Non-negotiable constraints**

1. **Do not ask me questions.** Make reasonable defaults and proceed.
2. **Use Python only** (no notebooks required for reproduction).
3. The repo must run on **CPU-only** as a baseline and support **CUDA if available**.
4. Provide clean separation: **src/bbpm** (library), **experiments/** (paper plots), **benchmarks/** (microbench), **tests/** (pytest), **scripts/** (one-command reproduction), **paper/** (LaTeX scaffold).
5. Everything must be reproducible with **fixed seeds**, saving outputs to **results/** and figures to **figures/**.
6. Must include **one LLM integration experiment** that requires **no training from scratch** (controller-based retrieval injection is acceptable).
7. Keep dependencies minimal: **torch**, **numpy**, **matplotlib**, **pyyaml**, **tqdm**, **pytest**. Optional: **transformers** only for LLM experiment.
8. Use **type hints**, docstrings, and consistent style. Add **basic linting** with **ruff** (optional but recommended).

### **Inputs you must reuse**

Refactor and incorporate logic from these existing prototype files (already present in the repo root or provided in workspace):

* **bbpm.py** (contains BBPM_Memory + experiments: capacity, ablations, etc.)
* **poc_for_bbpm.py** (contains differentiable hash map gradient flow, scalability, collision tolerance PoCs)

You must not just copy-paste; you must:

* **Extract a ****canonical BBPM definition** into src/bbpm/memory/float_superposition.py
* Extract hashing logic into **src/bbpm/hashing/***
* Convert each notebook-style experiment into **experiments/expXX_*/run.py** + **analyze.py**

---

# **High-Level Deliverables**

## **A) Library implementation (**

## **src/bbpm**

## **)**

Implement a clean BBPM library with:

* Hashing modules (global hash, block hash, multi-hash)
* Memory modules (float superposition with counts, binary bloom-like variant)
* Diagnostics (occupancy histograms, collision stats, skew proxies, self-collision estimates)
* Optional: simple eviction policies (decay, clipping, TTL)
* Integration helpers for LLM experiments (controller-based retrieval injection + minimal KV-augmentation hooks)

## **B) Experiments suite for paper (**

## **experiments/**

## **)**

Implement scripts that:

* Generate metrics
* Save JSON/CSV under **results/expXX_***
* Save figures under **figures/expXX_***
* Have consistent configs (**config.yaml**) and reproducible seeds

Minimum experiments:

* exp01_capacity_scaling
* exp02_ablation_K_H
* exp03_block_vs_global
* exp04_kv_memory_scaling (VRAM scaling plot; OK to use analytic estimate + optional GPU measurement)
* exp05_needle_haystack
* exp06_drift_stability
* exp07_llm_integration (no training from scratch)

## **C) Tests (**

## **tests/**

## **)**

Use **pytest** to validate:

* hashing determinism
* write-read identity in low load
* counts unbiasedness (counts normalization correctness)
* collision regimes sanity (fidelity decreases as load increases)
* gpu/cpu parity (if cuda exists; otherwise skip)

## **D) Scripts (**

## **scripts/**

## **)**

* **run_all_experiments.sh**: sequentially runs all experiments and generates all paper figures
* **reproduce_icml.sh**: fixed-seed run that produces the exact set of figures used in paper
* **download_models.sh**: downloads minimal HF model (only needed for LLM experiment)
* **format_and_lint.sh**: runs formatting/lint (optional)

## **E) Documentation**

* README.md: clear one-command reproduction instructions + artifact overview
* experiments/README.md: how to run each exp
* CITATION.cff and SECURITY.md: basic scaffolds
* Paper scaffold under **paper/** with placeholders (LaTeX files), without needing to be complete

---

# **Canonical BBPM Definition (must implement exactly)**

## **Memory model: Float Superposition with Counts (canonical)**

**Create class **BBPMMemoryFloat** in **src/bbpm/memory/float_superposition.py** with:**

Constructor parameters:

* **D: int** total slots
* **d: int** value dimension
* **K: int** active slots per item per hash
* **H: int** number of independent hashes (multi-hash)
* **hash_fn: HashFunction** dependency injection
* dtype**, **device
* write_scale: str** ∈ {**"unit"**, **"1/sqrt(KH)"**}, default **"1/sqrt(KH)"

Buffers:

* memory: [D, d]** float tensor**
* counts: [D, 1]** float tensor**

Methods:

* clear()
* write(keys: Tensor[B], values: Tensor[B, d])
* read(keys: Tensor[B]) -> Tensor[B, d]
* diagnostics(keys) -> dict** (optional helpers)**

Rules:

* Addressing returns indices of shape **[B, K*H]**.
* **write()** uses **index_add_** to add values into **memory**.
* **counts** increments by 1 per write per slot.
* **read()** gathers **memory[idx]** and **counts[idx]**, applies per-slot debias:
  * slot_val = memory / (counts + eps)** (as in your bbpm.py)**
* Then pools across **K*H** slots with mean.
* Ensure deterministic behavior given seed.

## **Hashing API**

**Define in **src/bbpm/hashing/base.py**:**

* class HashFunction(Protocol):
  * def indices(self, keys: torch.Tensor, K: int, H: int) -> torch.Tensor: ...
    Return shape: **[B, K*H]** (int64).

Implement:

1. **GlobalAffineHash** (like your salted formula in **bbpm.py**)
2. BlockPermutationHash**:**
   * memory divided into **B** blocks each of size **L** such that **D=B*L**
   * hash selects block id, then uses permutation for offsets
   * include self-collision minimization option
3. MultiHashWrapper**:**
   * wraps base hash and uses distinct salts to approximate independence
   * exposes diagnostics: collision rate, occupancy stats

## **Diagnostics**

**In **src/bbpm/hashing/diagnostics.py** implement functions:**

* occupancy_hist(indices, D) -> histogram
* max_load(indices, D)
* gini_load(indices, D)** (optional)**
* **collision_rate(indices)** (within-batch duplicates)
* estimate_q2(indices, D) = sum_i (c_i / total)^2** as a proxy for **∑ q_i^2
  These diagnostics must be used in experiments to connect theory ↔ behavior.

---

# **LLM Integration (must be implementable quickly; no training)**

**Implement ****controller-based retrieval injection** in **experiments/exp07_llm_integration/**:

## **Task**

Streaming associative recall beyond context window:

* **Stream N facts: **ID=`<integer>` VALUE=<token/string>
* Do NOT keep all facts in prompt. Keep only a small window W.
* Store mapping in BBPM: **ID -> value_id_code**
* Maintain a CPU store: **value_id -> fact_text**
* At query time:
  * Parse ID from query
  * BBPM read returns vector code
  * Decode nearest value_id (cosine similarity to codebook) among candidates
  * Inject retrieved fact text into the prompt and ask model to answer
* Baseline: same sliding window W without injection (should fail when fact is outside window).

## **Model**

Use Hugging Face **transformers** with a small causal LM (default: **sshleifer/tiny-gpt2** for portability). If no internet, allow skipping and still run synthetic scoring mode.

## **Outputs**

Save:

* accuracy vs N
* tokens/sec vs N (or vs streamed tokens)
* peak VRAM if CUDA available; otherwise report CPU RAM estimate
* store all metrics and configs in **results/exp07_llm_integration/**

---

# **Experiments Specification (must implement all)**

For each experiment:

* **run.py**: runs experiment, saves **metrics.json**, optionally **raw.csv**
* **analyze.py**: loads metrics, generates figure(s) under **figures/expXX_***
* **config.yaml**: contains parameters (D, d, K, H, seeds, trials, batch sizes)

## **exp01_capacity_scaling**

Replicate your “cos sim fidelity vs N” trend from **bbpm.py**.

* Sweep N over increasing values.
* Use normalized random unit vectors.
* Report mean cosine similarity over test subset.
* **Save figure: **capacity_vs_fidelity.png

## **exp02_ablation_K_H**

Reproduce K and multi-hash ablation.

* D small enough to force collisions.
* Sweep K in [4, 16, 64], H in [1, 3] (or configurable).
* Score: cosine sim threshold accuracy AND average cosine similarity.
* **Save figure: **ablation_K_H.png

## **exp03_block_vs_global**

Compare GlobalAffineHash vs BlockPermutationHash at same D, N, K, H.

* Report fidelity, collision stats, occupancy skew.
* **Save figure: **block_vs_global.png

## **exp04_kv_memory_scaling**

Produce plot showing memory scaling:

* KV cache memory ∝ layers * heads * T * head_dim (analytic estimate)
* BBPM memory = D*d (constant in T)
  If CUDA available, optionally measure actual **torch.cuda.max_memory_allocated**.
  **Save figure: **kv_vs_bbpm_memory.png

## **exp05_needle_haystack**

Simulate long context retrieval:

* Write N key/value pairs
* Query random subset
* Report success vs N, and compare to “window baseline” that only remembers last W items.
  **Save figure: **needle_accuracy.png

## **exp06_drift_stability**

Simulate key drift:

* Define keys as **hash(proj(x))** where x changes over time steps.
* Two modes:
  1. stable keys (fixed IDs) => should remain retrievable
  2. drifting keys (x updated with noise) => retrieval degrades
* Report retrieval accuracy over time.
  **Save figure: **drift_stability.png

## **exp07_llm_integration**

As described above.

---

# **Repository Files to Create/Fill**

Create all files from the tree below (already provided) with real content:

Top-level:

* README.md: include quickstart, install, run all experiments, reproduce ICML figures
* pyproject.toml: package metadata, dependencies, optional extras **[llm]**
* requirements.txt + environment.yml: pinned versions loosely
* **Makefile: targets **install**, **test**, **lint**, **exp**, **reproduce
* .gitignore: ignore **results/**, **figures/**, **.venv**, caches
* LICENSE: MIT
* CITATION.cff: minimal citation file
* SECURITY.md: basic disclosure template

Library:

* all modules under **src/bbpm** as specified
* Ensure **bbpm/__init__.py** exports main classes

Benchmarks:

* microbench_write_read.py: timing vs K, D, dtype
* occupancy_bench.py: occupancy distributions and skew
* kernel_bench.py: compare torch ops (and triton if installed)

Tests:

* add robust tests with **pytest** and deterministic seeding
* skip GPU parity tests if no CUDA

Scripts:

* **run_all_experiments.sh**: run exp01..exp07 and analyze
* **reproduce_icml.sh**: same but fixed configs + fixed seeds
* **download_models.sh**: downloads HF model cache for exp07
* **format_and_lint.sh**: optional ruff formatting

GitHub workflows:

* .github/workflows/tests.yml**: install, run pytest**
* .github/workflows/lint.yml**: ruff + import check**

Paper scaffold:

* **paper/icml2026/main.tex**, **references.bib**, section placeholders and appendix placeholders

bbpm/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ SECURITY.md
├─ .gitignore
├─ pyproject.toml
├─ requirements.txt
├─ environment.yml
├─ Makefile
│
├─ src/
│  └─ bbpm/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ utils/
│     │  ├─ seed.py
│     │  ├─ logging.py
│     │  ├─ timing.py
│     │  └─ device.py
│     │
│     ├─ hashing/
│     │  ├─ __init__.py
│     │  ├─ base.py                # interface: hash(keys)->indices
│     │  ├─ global_hash.py         # simple affine hash family
│     │  ├─ (block_hash.py removed - use addressing/bbpm_addressing.py instead)
│     │  ├─ multihash.py           # independent salts, collision diagnostics
│     │  └─ diagnostics.py         # occupancy histograms, skew metrics
│     │
│     ├─ memory/
│     │  ├─ __init__.py
│     │  ├─ base.py                # write/read API, scaling conventions
│     │  ├─ float_superposition.py # canonical BBPM (with counts)
│     │  ├─ binary_bloom.py        # 1-bit / uint8 variant
│     │  ├─ eviction.py            # optional: decay, clipping, TTL
│     │  └─ kernels/
│     │     ├─ __init__.py
│     │     ├─ torch_ops.py        # baseline index_add_/gather
│     │     └─ triton_ops.py       # optional fast kernels
│     │
│     ├─ integration/
│     │  ├─ __init__.py
│     │  ├─ interfaces.py          # how BBPM plugs into models
│     │  ├─ keying.py              # key strategy: ids/pos/stopgrad-proj
│     │  ├─ fusion.py              # gating/residual fusion modules
│     │  └─ kv_substitution.py      # limited-KV + BBPM augmentation
│     │
│     └─ theory/
│        ├─ __init__.py
│        ├─ collision_model.py      # collision prob, non-uniform q_i
│        ├─ snr_bounds.py           # expected + tail bounds
│        └─ sanity_checks.py        # verify assumptions empirically
│
├─ experiments/
│  ├─ README.md
│  ├─ exp01_capacity_scaling/
│  │  ├─ run.py
│  │  ├─ config.yaml
│  │  └─ analyze.py
│  ├─ exp02_ablation_K_H/
│  │  ├─ run.py
│  │  └─ analyze.py
│  ├─ exp03_block_vs_global/
│  │  ├─ run.py
│  │  └─ analyze.py
│  ├─ exp04_kv_memory_scaling/
│  │  ├─ run.py                  # reproduces your KV vs BBPM VRAM plot
│  │  └─ analyze.py
│  ├─ exp05_needle_haystack/
│  │  ├─ run.py
│  │  └─ analyze.py
│  ├─ exp06_drift_stability/
│  │  ├─ run.py                  # key drift / training-time stability test
│  │  └─ analyze.py
│  └─ exp07_llm_integration/
│     ├─ run_eval.py              # real model run: long context recall
│     ├─ run_speed.py             # tokens/sec vs context length
│     ├─ configs/
│     │  ├─ small_decoder.yaml
│     │  └─ bbpm_aug.yaml
│     └─ analyze.py
│
├─ benchmarks/
│  ├─ microbench_write_read.py     # latency per op vs K, D, dtype
│  ├─ occupancy_bench.py           # distribution of collisions
│  └─ kernel_bench.py              # torch vs triton (if present)
│
├─ tests/
│  ├─ test_hashing_determinism.py
│  ├─ test_write_read_identity.py
│  ├─ test_counts_unbiasedness.py
│  ├─ test_collision_regimes.py
│  └─ test_gpu_cpu_parity.py
│
├─ scripts/
│  ├─ download_models.sh
│  ├─ run_all_experiments.sh       # produces every figure for the paper
│  ├─ reproduce_icml.sh            # exact commands + fixed seeds
│  └─ format_and_lint.sh
│
├─ figures/                        # generated figures saved here (gitignored)
├─ results/                        # raw logs/metrics (gitignored)
│
├─ paper/
│  ├─ icml2026/
│  │  ├─ main.tex
│  │  ├─ references.bib
│  │  ├─ sections/
│  │  ├─ figures/                  # final paper figures
│  │  └─ tables/
│  └─ appendix/
│     ├─ extra_theory.tex
│     └─ extra_experiments.tex
│
└─ .github/
   ├─ workflows/
   │  ├─ tests.yml                 # unit tests + minimal smoke runs
   │  └─ lint.yml
   └─ ISSUE_TEMPLATE/

---

# **Implementation Details and Style Requirements**

1. **Seed control**: Implement set_global_seed(seed)** in **src/bbpm/utils/seed.py** (torch, numpy, random).**
2. **Logging**: implement a minimal structured logger writing to both console and **results/*/log.txt**.
3. **Timing**: implement a **Timer** context manager.
4. **Config loading**: use YAML; implement a small helper **load_config(path)** in **src/bbpm/config.py**.
5. **CLI behavior**: each run.py** should accept **--config path/to/config.yaml** and **--outdir results/...** and **--device cpu|cuda**. Use argparse (no heavy frameworks).**
6. **Numerical stability**: always use **eps=1e-8** for counts division.
7. **Performance**: use batched writes/reads to avoid OOM.
8. **Save artifacts**: metrics JSON must include:
   * all hyperparameters
   * collision diagnostics
   * runtime stats
   * fidelity stats

---

# **Mapping from Prototype Code**

Extract key logic directly:

* From **bbpm.py**: **BBPM_Memory** → become canonical float memory with counts + multi-hash behavior
* From **bbpm.py**: exp01 and exp02 logic → experiments folder
* From **poc_for_bbpm.py**: gradient flow PoC → convert into a short optional benchmark or docs section (not required for main claims), and speed vs scale plot → can inform microbench suite

Do not include unnecessary “universal neural memory” demos unless time allows; focus on paper-critical experiments.

---

# **Acceptance Criteria (you must satisfy)**

After implementation, these commands must work:

```
python -m pip install -e .
pytest -q
bash scripts/run_all_experiments.sh
```

And it must produce:

* figures/exp01_capacity_scaling/capacity_vs_fidelity.png
* figures/exp02_ablation_K_H/ablation_K_H.png
* figures/exp03_block_vs_global/block_vs_global.png
* figures/exp04_kv_memory_scaling/kv_vs_bbpm_memory.png
* figures/exp05_needle_haystack/needle_accuracy.png
* figures/exp06_drift_stability/drift_stability.png
* **figures/exp07_llm_integration/llm_accuracy_vs_N.png** (or skip with explicit message if transformers unavailable)

No silent failures. All scripts must exit non-zero on error.

---

# **Execution Plan (perform in order)**

1. Create all missing directories/files per tree.
2. Implement bbpm library modules first (hashing + memory + utils).
3. Implement experiments exp01–exp06 (synthetic).
4. Implement exp07 LLM integration last (with optional dependency).
5. Add tests and ensure they pass.
6. Write README with exact reproduction steps.
7. Add GitHub workflows.
8. Do a final run of **scripts/run_all_experiments.sh** in a clean environment mindset (simulate by deleting results/figures before running).

Proceed now and implement everything.
