# Dataset Documentation and Reproducibility Guide

**Assignment 4:** Distributed Matrix Multiplication
**Course:** BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC (2026–2027)

---

## 1. Repository Storage Policy

The generated dense benchmark CSV matrices located under `data/dense/` (~310 MB total across 14 files) are **deliberately excluded from version control** via `.gitignore` (`/data/dense/`) and reconstructed deterministically on demand using `scripts/generate_dataset.py` and `data/SHA256SUMS.txt`.

### Rationale
- **Repository Efficiency & Hygiene:** Tracking 310 MB of raw CSV matrices in Git would permanently bloat the Git packfile history by ~80 MB of compressed blobs.
- **GitHub Policy Compliance:** Two matrix files (`A_2000.csv` and `B_2000.csv`, ~76.29 MiB each) exceed GitHub's 50 MB warning threshold (`remote: warning: GH001: Large files detected`).
- **Complete Deterministic Reproducibility:** The generator script (`scripts/generate_dataset.py`) and the certified checksum manifest (`data/SHA256SUMS.txt`) are fully tracked in Git. Any fresh clone can reconstruct and verify the exact bit-for-bit dataset in approximately 15 seconds without third-party dependencies.

---

## 2. Dataset Inventory

The dataset is divided into two categories:

| Directory | Content Description | Version-Controlled in Git? | File Count & Size |
|---|---|---|---|
| `data/baselines/` | Frozen historical baseline summaries from Assignment 1 (`a1_summary.csv`) and Assignment 3 (`a3_parallel_results_summary.csv`) used by offline contextual analysis (`scripts/analyze_campaign.py`) | **Yes (Tracked)** | 2 files (15.02 KiB / ~15 KB) |
| `data/validation/` | Small analytical validation fixtures ($4 \times 4$) and independent numerical reference solution ($250 \times 250$ `C_250.csv`) | **Yes (Tracked)** | 6 files (1.13 MiB) |
| `data/dense/` | Large benchmark input matrices ($A_n$ and $B_n$ for $n \in \{100, 250, 500, 750, 1000, 1500, 2000\}$) | **No (Generated on-demand)** | 14 files (310.33 MiB) |
| `data/SHA256SUMS.txt` | Cryptographic SHA-256 checksum manifest indexing all 20 CSV files | **Yes (Tracked)** | 1 file (1,911 bytes) |

---

## 3. Fresh Clone Reconstruction Workflow

When cloning the repository freshly onto a new machine or environment, the `data/dense/` directory will initially be empty or absent.

### Step 1: Run the Deterministic Generator
Execute the generator from the repository root:

```bash
python scripts/generate_dataset.py
```

This script:
1. Re-creates `data/dense/` and writes all 14 benchmark matrices ($A_n$ and $B_n$).
2. Re-creates analytical fixtures in `data/validation/`.
3. Computes the pure Python reference product for $n=250$ in `data/validation/reference/C_250.csv`.
4. Outputs verification progress to stdout.

### Step 2: Certify Dataset Integrity
Immediately following generation, execute the independent verification script:

```bash
python scripts/verify_dataset.py data/SHA256SUMS.txt
```

### Expected Verification Output
A valid reconstruction produces the following output:

```text
=================================================================
Assignment 4: Dataset Integrity Verification
Manifest: .../Assignment04_Abel/data/SHA256SUMS.txt
=================================================================

SHA-256 Checksum Verification: 20 passed, 0 failed.
Matrix Structure & Dimension Verification: 20 passed, 0 failed.

[SUCCESS] All data files verified bit-for-bit with correct dimensions and finite values.
```

If the verification script exits with code 0 and reports `20 passed, 0 failed`, the environment is certified and ready for experimental execution.

> [!IMPORTANT]
> Because `scripts/verify_dataset.py` inspects every entry in `data/SHA256SUMS.txt`, running verification *before* generation on a fresh clone will correctly report missing files. The required execution sequence is **generation first, verification second**.

---

## 4. Frozen Data Immutability Rules

To maintain experimental validity across benchmark runs:

1. **Existing Experimental Workspace:** In an active workspace where `data/dense/` has already been materialized and verified, **do not run `scripts/generate_dataset.py` again**, and never modify, edit, or overwrite existing CSV matrices.
2. **Fresh Clone Workspace:** Regeneration is authorized and required **only** on a fresh clone or clean environment where `data/dense/` is missing.
3. **Frozen Generation Parameters:** The PRNG algorithm (`random.Random` MT19937), seed formula ($\mathrm{seed}(n) = 42 + n$), matrix dimensions ($n \in \{100, 250, 500, 750, 1000, 1500, 2000\}$), `%.17g` formatting, Unix LF line endings, and element values are frozen for reproducibility and certified against `data/SHA256SUMS.txt`.

---

## 5. Specification of the Generator Algorithm

The dataset generator (`scripts/generate_dataset.py`) implements a rigorous, portable, and standard-library-only generation process:

- **Random Number Generator:** Python standard library `random.Random(seed)` using the 32-bit Mersenne Twister algorithm (MT19937), yielding 53-bit resolution floats in $[0.0, 1.0)$.
- **Seed Derivation:** Fully deterministic per dimension:
  $$\mathrm{seed}(n) = 42 + n$$
  - $n=100 \implies \mathrm{seed} = 142$ (Auxiliary for Assignment 1 sequential baseline)
  - $n=250 \implies \mathrm{seed} = 292$ (Common 3-way benchmark)
  - $n=500 \implies \mathrm{seed} = 542$ (Common 3-way benchmark)
  - $n=750 \implies \mathrm{seed} = 792$ (Common 3-way benchmark)
  - $n=1000 \implies \mathrm{seed} = 1042$ (Common 3-way benchmark)
  - $n=1500 \implies \mathrm{seed} = 1542$ (Distributed scaling benchmark)
  - $n=2000 \implies \mathrm{seed} = 2042$ (Distributed scaling benchmark)
- **Generation Sequence:** For each dimension $n$, matrix $A$ is generated first in row-major order (`[[rng.random() for _ in range(n)] for _ in range(n)]`), followed immediately by matrix $B$ from the same seeded PRNG stream.
- **Serialization & Precision:** Values are serialized as comma-separated values using format string `%.17g`, which guarantees IEEE 754 double precision (binary64) without losing precision or storing superfluous characters.
- **Line Endings:** Explicit Unix LF (`\n`).
- **Dependencies & Compatibility:** Requires Python 3.9+ standard library (`random`, `csv`, `hashlib`, `os`, `sys`). Requires zero third-party packages (NumPy is not required for generation).

---

## 6. Official Academic Report Description

In the final Assignment 4 research report (Section 3: Experimental Methodology and Section 4: Empirical Results) and provenance documentation, distinguish clearly between the shared A4 dataset used for controlled common-input comparisons and the preserved historical baseline summaries:
- `data/dense/` provides deterministic IEEE 754 `binary64` matrices generated by `scripts/generate_dataset.py` ($\mathrm{seed}(n) = 42 + n$, values in $[0.0, 1.0)$) consumed identically by Assignment 4 Dask campaigns (`results/raw/raw_measurements.csv`) and the controlled common-input Assignment 1 and Assignment 3 baseline executions (`results/raw/a1_common_input_raw.csv`, `results/raw/a3_common_input_raw.csv`, orchestrated via `scripts/run_common_input_baselines.py` using `scripts/prepare_a1_runtime.py` and `scripts/prepare_a3_runtime.py`).
- `data/baselines/` preserves frozen historical summary CSVs (`a1_summary.csv` and `a3_parallel_results_summary.csv`) originally measured on each prior assignment's own random matrices (`float64`/`double`), retained strictly as historical contextual reference (`results/analysis/contextual_baseline_comparison.csv`) without mixing them into the controlled common-input comparison.

---

## 7. Certified Checksum Manifest Reference

The official hashes in `data/SHA256SUMS.txt` are:

```text
99c7730a58a915c01b2967b21d9bd3cd61f5e20278ea0f12bc30e0b679b28b36  dense/A_100.csv
1727aa2f3e620fe5e83e523d990f2141e6dc3d5524dc3332ec73965b7f2afa91  dense/A_1000.csv
f69b2ddd3625383740af8c056a28c84a000bf4f8f6ed279a0143647875ac15de  dense/A_1500.csv
a840b5d73026670a511cbc72145e81f71c317fdda37d913a646685e329c3edc2  dense/A_2000.csv
8fc7f538e6047d2fc11d5076154d106935ebd6a34a6ccc67e8c13536016c0d40  dense/A_250.csv
f27913d865d29276ef24f53f5a6862447b330e7f888182aff0246a0532acf756  dense/A_500.csv
3087c1228c76d369ed34bbb2f99b327ebb62894ef2103cb05e43f43351f9bc05  dense/A_750.csv
cc53ac9fef8c85356e547c0eb49841a680bd4ffc73b3ae3cc85d4bf3bf67bf51  dense/B_100.csv
b68ebbd44e43933c01afcdd6da96560083c76012aecfb871440efe900abf6062  dense/B_1000.csv
9694e3268dd0cab5076835a0a36edfe246ed0944eee58792515efa05f49f8c6c  dense/B_1500.csv
0e73154d2201c19196690485304e3c19c5d50f489192e02657df011d7caf6966  dense/B_2000.csv
0002373f420cd18209555e5d14557d2d0cbebc4da3a42f872c0886a79f8067c3  dense/B_250.csv
dee2a7abadff44e7bda78f877eff28914d9210c7a6525d540f37b66db2067c5b  dense/B_500.csv
d99bbcb0c0000343f921822ebbdc1ea10999dd4a4e4a31e67af4520cd8783c81  dense/B_750.csv
301e357c2477a2502677ff14790e970606f1d9251c2d7945a885a20b91995217  validation/expected_C_4.csv
1071e33f72efb352764e0ea040792695cd75360b1f1b292bf9c36059cd47f18f  validation/identity_4.csv
45acf3b6c71a4be592d02bd6e221efe32512e7e3a35a9742453865876840b974  validation/known_A_4.csv
280740971507426667e76a66c7bec90675edc61cfac696e1930bd9bbd4aab252  validation/known_B_4.csv
598e9a9fc8f0c8db1d5da277f4d93561ad83e2df77e3fcb5f968f67b8033909e  validation/reference/C_250.csv
506351148090d688af818822486473480a012ee2c4e142312b7b4a8b99a56595  validation/zero_4.csv
```
