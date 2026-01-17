# θ-Learning: Timing-Safe Neural Memory

[![Tests](https://github.com/hmshujaatzaheer/theta-learning-research/actions/workflows/tests.yml/badge.svg)](https://github.com/hmshujaatzaheer/theta-learning-research/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/hmshujaatzaheer/theta-learning-research/branch/main/graph/badge.svg)](https://codecov.io/gh/hmshujaatzaheer/theta-learning-research)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Status: Research Proposal](https://img.shields.io/badge/Status-Research%20Proposal-yellow)]()
[![Theory: Proven](https://img.shields.io/badge/Theory-Proven-green)]()
[![Experiments: Pending](https://img.shields.io/badge/Experiments-Pending-orange)]()

## Overview

This repository contains the reference implementation and experimental infrastructure for the **θ-Learning Principle**, a mathematical framework for timing-safe neural memory as described in the PhD research proposal.

> **Important Disclaimer**: This repository distinguishes between:
> - ✅ **Implemented & Verified**: Core algorithms, theoretical complexity analysis
> - 🔬 **Requires Lab Validation**: Hardware timing measurements, benchmark scores, security evaluation
> - 📐 **Mathematically Proven**: Theorems (proofs in proposal, code provides illustrations)

---

## Table of Contents

1. [What This Repository Contains](#what-this-repository-contains)
2. [What Requires Future Validation](#what-requires-future-validation)
3. [Installation](#installation)
4. [Repository Structure](#repository-structure)
5. [Mapping to Proposal](#mapping-to-proposal)
6. [Running the Code](#running-the-code)
7. [Research Phases](#research-phases)
8. [Citation](#citation)

---

## What This Repository Contains

### ✅ Implemented Now (This Repository)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| θMN Core Algorithm | `src/core/theta_mn.py` | ✅ Complete | Basic θ-Memory Network implementation |
| θMN(r) Low-Rank | `src/core/theta_mn_lr.py` | ✅ Complete | Low-rank factorized variant |
| EWC Integration | `src/core/ewc.py` | ✅ Complete | Elastic Weight Consolidation |
| FLOP Counter | `src/utils/complexity.py` | ✅ Complete | Operation counting for complexity verification |
| Theoretical Plots | `src/utils/theoretical_plots.py` | ✅ Complete | Complexity curves (theoretical, not empirical) |
| Timing Infrastructure | `src/utils/timing.py` | ✅ Complete | Measurement framework (NOT security validation) |
| Unit Tests | `tests/` | ✅ Complete | Correctness verification |

### 📐 Mathematically Proven (Proofs in Proposal)

| Theorem | Proposal Section | Code Illustration | What Code Shows |
|---------|------------------|-------------------|-----------------|
| Theorem 1: Storage Leaks | Section 3 | `experiments/phase1_theory/theorem1_illustration.py` | Demonstrates variable timing in storage-based memory |
| Theorem 2: θ-Learning Timing-Safety | Section 4 | `experiments/phase1_theory/theorem2_illustration.py` | Shows constant FLOP count regardless of input |
| Theorem 3: Complexity Bounds | Section 4 | `experiments/phase1_theory/theorem3_verification.py` | Verifies O(d²) and O(rd) operation counts |
| Theorem 4: Functional Equivalence | Section 5 | `experiments/phase1_theory/theorem4_illustration.py` | Demonstrates recall capability |
| Theorem 5: Universal Transformation | Section 7 | `experiments/phase1_theory/theorem5_illustration.py` | Shows transformation preserves functionality |

**Note**: Code *illustrates* theorems but does not *prove* them. Proofs are mathematical and in the proposal.

---

## What Requires Future Validation

### 🔬 Phase 2: Hardware Validation (Months 7-12)

| Experiment | File | Status | What's Needed |
|------------|------|--------|---------------|
| Constant-time on A100 | `experiments/phase2_implementation/gpu_timing_a100.py` | 🔬 Placeholder | Access to A100 GPU, statistical analysis |
| Constant-time on H100 | `experiments/phase2_implementation/gpu_timing_h100.py` | 🔬 Placeholder | Access to H100 GPU, statistical analysis |
| Memory bandwidth analysis | `experiments/phase2_implementation/memory_bandwidth.py` | 🔬 Placeholder | GPU profiling tools (NSight) |
| Wall-clock speedup | `experiments/phase2_implementation/wallclock_speedup.py` | 🔬 Placeholder | Optimized CUDA kernels |

**Required Resources**:
- NVIDIA A100/H100 GPUs
- CUDA profiling tools (NSight Compute, NSight Systems)
- Statistical analysis framework for timing measurements
- Minimum 1000 trials per configuration for statistical significance

### 🔬 Phase 3: Security Evaluation (Months 13-18)

| Experiment | File | Status | What's Needed |
|------------|------|--------|---------------|
| PROMPTPEEK reproduction | `experiments/phase3_security/promptpeek_baseline.py` | 🔬 Placeholder | Reproduce Wu et al. attack |
| Attack on θMN | `experiments/phase3_security/attack_theta_mn.py` | 🔬 Placeholder | Attempt timing attacks on θMN |
| Statistical timing analysis | `experiments/phase3_security/timing_statistics.py` | 🔬 Placeholder | Mutual information estimation |
| Covert channel capacity | `experiments/phase3_security/covert_channel.py` | 🔬 Placeholder | Information-theoretic analysis |

**Required Resources**:
- Isolated measurement environment (no other processes)
- High-precision timers (rdtsc or equivalent)
- Statistical tools for mutual information estimation
- Adversarial evaluation framework

### 🔬 Phase 4: Benchmark Evaluation (Months 19-24)

| Experiment | File | Status | What's Needed |
|------------|------|--------|---------------|
| MMLU evaluation | `experiments/phase4_benchmarks/mmlu_eval.py` | 🔬 Placeholder | Trained θMN model, MMLU dataset |
| GSM8K evaluation | `experiments/phase4_benchmarks/gsm8k_eval.py` | 🔬 Placeholder | Trained θMN model, GSM8K dataset |
| HumanEval evaluation | `experiments/phase4_benchmarks/humaneval_eval.py` | 🔬 Placeholder | Trained θMN model, HumanEval dataset |
| Comparison with TTT-E2E | `experiments/phase4_benchmarks/ttt_comparison.py` | 🔬 Placeholder | TTT-E2E reproduction |
| Comparison with Mamba | `experiments/phase4_benchmarks/mamba_comparison.py` | 🔬 Placeholder | Mamba reproduction |

**Required Resources**:
- Large-scale training infrastructure (8+ GPUs for weeks)
- Pre-training data (CommonCrawl, The Pile, etc.)
- Benchmark datasets and evaluation harnesses
- Baseline model reproductions

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/theta-learning-research.git
cd theta-learning-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## Repository Structure

```
theta-learning-research/
├── README.md                    # This file
├── VALIDATION_STATUS.md         # Detailed validation status
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── pyproject.toml              # Modern Python packaging
│
├── src/                        # Source code
│   ├── core/                   # Core implementations
│   │   ├── __init__.py
│   │   ├── theta_mn.py         # ✅ θMN implementation
│   │   ├── theta_mn_lr.py      # ✅ θMN(r) low-rank variant
│   │   ├── ewc.py              # ✅ Elastic Weight Consolidation
│   │   └── transformer.py      # ✅ Baseline Transformer attention
│   │
│   ├── models/                 # Full model implementations
│   │   ├── __init__.py
│   │   └── theta_lm.py         # Language model with θMN layers
│   │
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   ├── complexity.py       # ✅ FLOP counting
│   │   ├── timing.py           # ✅ Timing measurement infrastructure
│   │   └── theoretical_plots.py # ✅ Theoretical complexity plots
│   │
│   └── baselines/              # Baseline implementations
│       ├── __init__.py
│       ├── kv_cache.py         # Standard KV-cache (for comparison)
│       └── linear_attention.py # Linear attention baseline
│
├── tests/                      # Unit tests
│   ├── test_theta_mn.py        # ✅ Core algorithm tests
│   ├── test_theta_mn_lr.py     # ✅ Low-rank variant tests
│   ├── test_complexity.py      # ✅ Complexity verification tests
│   └── test_functional.py      # ✅ Functional equivalence tests
│
├── experiments/                # Experimental protocols
│   ├── phase1_theory/          # ✅ Theoretical illustrations
│   │   ├── theorem1_illustration.py
│   │   ├── theorem2_illustration.py
│   │   ├── theorem3_verification.py
│   │   ├── theorem4_illustration.py
│   │   └── theorem5_illustration.py
│   │
│   ├── phase2_implementation/  # 🔬 Hardware validation (placeholder)
│   │   ├── README.md           # Protocol documentation
│   │   ├── gpu_timing_a100.py
│   │   ├── gpu_timing_h100.py
│   │   └── wallclock_speedup.py
│   │
│   ├── phase3_security/        # 🔬 Security evaluation (placeholder)
│   │   ├── README.md           # Protocol documentation
│   │   ├── promptpeek_baseline.py
│   │   ├── attack_theta_mn.py
│   │   └── timing_statistics.py
│   │
│   └── phase4_benchmarks/      # 🔬 Benchmark evaluation (placeholder)
│       ├── README.md           # Protocol documentation
│       ├── mmlu_eval.py
│       ├── gsm8k_eval.py
│       └── humaneval_eval.py
│
├── docs/                       # Documentation
│   ├── THEOREMS.md             # Theorem statements and proof sketches
│   ├── ALGORITHMS.md           # Algorithm descriptions
│   └── EXPERIMENTAL_PROTOCOLS.md # Detailed experimental protocols
│
├── figures/                    # Generated figures
│   └── .gitkeep
│
├── data/                       # Data directory
│   ├── raw/                    # Raw experimental data
│   └── processed/              # Processed results
│
└── scripts/                    # Utility scripts
    ├── generate_theoretical_plots.py
    └── run_all_tests.py
```

---

## Mapping to Proposal

### Direct Correspondence

| Proposal Section | Repository Location | Status |
|-----------------|---------------------|--------|
| Section 4: θ-Learning Principle | `src/core/theta_mn.py` | ✅ Implemented |
| Section 4: Theorem 2 (Timing-Safety) | `experiments/phase1_theory/theorem2_illustration.py` | ✅ Illustrated |
| Section 4: Theorem 3 (Complexity) | `src/utils/complexity.py` | ✅ Verified |
| Section 5: Functional Recall | `tests/test_functional.py` | ✅ Tested |
| Section 6: θMN(r) Low-Rank | `src/core/theta_mn_lr.py` | ✅ Implemented |
| Section 6: EWC | `src/core/ewc.py` | ✅ Implemented |
| Section 9: Phase 1 | `experiments/phase1_theory/` | ✅ Complete |
| Section 9: Phase 2 | `experiments/phase2_implementation/` | 🔬 Placeholder |
| Section 9: Phase 3 | `experiments/phase3_security/` | 🔬 Placeholder |
| Section 9: Phase 4 | `experiments/phase4_benchmarks/` | 🔬 Placeholder |

### Figure Correspondence

| Proposal Figure | Repository File | Type |
|----------------|-----------------|------|
| Figure 3: Complexity Comparison | `figures/complexity_comparison.pdf` | 📐 Theoretical |
| Figure 4: θMN(r) Architecture | `docs/architecture.md` | 📐 Diagram |
| Figure 5: Speedup | `figures/speedup_theoretical.pdf` | 📐 Theoretical |

---

## Running the Code

### Quick Start

```bash
# Run all tests
python -m pytest tests/ -v

# Generate theoretical plots
python scripts/generate_theoretical_plots.py

# Run theorem illustrations
python experiments/phase1_theory/theorem2_illustration.py
python experiments/phase1_theory/theorem3_verification.py
```

### Example: Basic θMN Usage

```python
from src.core.theta_mn import ThetaMemoryNetwork

# Initialize θMN
model = ThetaMemoryNetwork(d=512, lr=0.01)

# Process a sequence
for token in sequence:
    output = model.forward(token)
    model.update(token, target)

# Query the memory
answer = model.query(question)
```

### Example: Verify Complexity

```python
from src.utils.complexity import count_flops

# Count FLOPs for θMN vs Transformer
theta_flops = count_flops('theta_mn', d=4096, r=512, n=1)  # Per token
transformer_flops = count_flops('transformer', d=4096, n=8192)  # Per token at n=8K

print(f"θMN(512): {theta_flops} FLOPs per token")
print(f"Transformer at 8K: {transformer_flops} FLOPs per token")
print(f"Ratio: {transformer_flops / theta_flops:.1f}x")
```

---

## Research Phases

### Phase 1: Theoretical Foundations (✅ This Repository)

**Timeline**: Months 1-6  
**Status**: ✅ Complete in this repository

**Deliverables**:
- [x] Core algorithm implementation
- [x] Complexity verification code
- [x] Theorem illustrations
- [x] Unit tests
- [x] Theoretical plots

**How to use**:
```bash
# Run all Phase 1 experiments
cd experiments/phase1_theory
python theorem1_illustration.py
python theorem2_illustration.py
python theorem3_verification.py
```

---

### Phase 2: Implementation & Hardware Validation (🔬 Placeholder)

**Timeline**: Months 7-12  
**Status**: 🔬 Protocols defined, requires lab execution

**What's Provided**:
- Experimental protocols in `experiments/phase2_implementation/README.md`
- Placeholder scripts with TODO markers
- Expected output formats

**What's Needed**:
1. **Hardware**: NVIDIA A100 or H100 GPUs
2. **Tools**: CUDA Toolkit, NSight Compute, NSight Systems
3. **Environment**: Isolated machine (no background processes)
4. **Time**: ~2-4 weeks of dedicated GPU time

**Key Experiments**:

| Experiment | Purpose | Success Criteria |
|------------|---------|------------------|
| `gpu_timing_a100.py` | Verify constant-time on A100 | CV < 1% across inputs |
| `gpu_timing_h100.py` | Verify constant-time on H100 | CV < 1% across inputs |
| `wallclock_speedup.py` | Measure actual speedup | Document real vs theoretical |

**Protocol** (from `experiments/phase2_implementation/README.md`):
```
1. Disable GPU boost clocks (fixed frequency)
2. Warm up GPU with 100 iterations
3. For each input configuration:
   a. Run 1000 trials
   b. Record high-precision timing
   c. Compute mean, std, CV
4. Statistical tests:
   a. ANOVA across input types
   b. Mutual information estimation
5. Report with confidence intervals
```

---

### Phase 3: Security Evaluation (🔬 Placeholder)

**Timeline**: Months 13-18  
**Status**: 🔬 Protocols defined, requires lab execution

**What's Provided**:
- Attack reproduction protocols
- Statistical analysis framework
- Expected result formats

**What's Needed**:
1. Reproduce PROMPTPEEK attack (Wu et al.)
2. Attempt attack on θMN
3. Measure mutual information between timing and input
4. Estimate covert channel capacity

**Key Experiments**:

| Experiment | Purpose | Success Criteria |
|------------|---------|------------------|
| `promptpeek_baseline.py` | Reproduce known attack | Match 95%+ accuracy from paper |
| `attack_theta_mn.py` | Attack θMN | Document any information leakage |
| `timing_statistics.py` | Estimate MI(T; X) | MI < ε (define threshold) |

---

### Phase 4: Benchmark Evaluation (🔬 Placeholder)

**Timeline**: Months 19-24  
**Status**: 🔬 Protocols defined, requires large-scale training

**What's Provided**:
- Evaluation harness structure
- Metric computation code
- Comparison framework

**What's Needed**:
1. **Training Infrastructure**: 8+ GPUs for 2-4 weeks
2. **Training Data**: Large corpus (100B+ tokens)
3. **Evaluation Datasets**: MMLU, GSM8K, HumanEval
4. **Baseline Models**: Trained Transformer, Mamba reproduction

**Key Experiments**:

| Experiment | Purpose | Success Criteria |
|------------|---------|------------------|
| `mmlu_eval.py` | Measure MMLU accuracy | Report with confidence intervals |
| `gsm8k_eval.py` | Measure math reasoning | Compare to Transformer baseline |
| `ttt_comparison.py` | Compare to TTT-E2E | Fair comparison (same params) |

---

## Validation Status Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION STATUS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ COMPLETE (This Repository)                                  │
│  ├── Core θMN implementation                                    │
│  ├── θMN(r) low-rank variant                                    │
│  ├── EWC integration                                            │
│  ├── FLOP counting & complexity verification                    │
│  ├── Theoretical complexity plots                               │
│  ├── Theorem illustrations                                      │
│  └── Unit tests                                                 │
│                                                                 │
│  📐 MATHEMATICALLY PROVEN (Proofs in Proposal)                  │
│  ├── Theorem 1: Storage-based memory leaks timing               │
│  ├── Theorem 2: θ-Learning achieves MI(T;X) = 0                 │
│  ├── Theorem 3: Complexity is O(d²) or O(rd)                    │
│  ├── Theorem 4: Functional equivalence                          │
│  └── Theorem 5: Universal transformation                        │
│                                                                 │
│  🔬 REQUIRES LAB VALIDATION                                     │
│  ├── Phase 2: Constant-time on real GPUs                        │
│  ├── Phase 2: Wall-clock speedup measurement                    │
│  ├── Phase 3: Security evaluation (attack attempts)             │
│  ├── Phase 3: Mutual information measurement                    │
│  ├── Phase 4: MMLU benchmark scores                             │
│  ├── Phase 4: GSM8K benchmark scores                            │
│  ├── Phase 4: HumanEval benchmark scores                        │
│  └── Phase 4: Comparison with baselines                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Theoretical Plots

The plots in `figures/` are **theoretical** based on mathematical complexity analysis:

- `complexity_comparison.pdf` - O(nd) vs O(d²) vs O(rd) curves
- `speedup_theoretical.pdf` - Theoretical speedup ratio n/r

**These are NOT empirical measurements**. Empirical validation requires Phase 2 experiments.

To regenerate:
```bash
python scripts/generate_theoretical_plots.py
```

---

## Citation

If you use this code, please cite the proposal:

```bibtex
@misc{theta-learning-2026,
  title={The θ-Learning Principle: A Universal Mathematical Framework for Timing-Safe Neural Memory},
  author={[Author]},
  year={2026},
  note={PhD Research Proposal}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work builds upon:
- Wu et al. (NDSS 2025) - PROMPTPEEK attack
- Gu & Dao (COLM 2024) - Mamba architecture
- Sun et al. (2025) - TTT-E2E
- Yao, Hu & Klimovic (EuroSys 2025) - DeltaZip

---

## Contact

For questions about this research, please open an issue or contact [email].
