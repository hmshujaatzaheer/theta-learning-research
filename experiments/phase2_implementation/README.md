# Phase 2: Implementation & Hardware Validation

## Status: 🔬 PLACEHOLDER - Requires Lab Execution

This directory contains experimental protocols for Phase 2 of the research plan (Months 7-12).

**These experiments CANNOT be completed without:**
1. Access to NVIDIA A100 or H100 GPUs
2. Isolated measurement environment
3. CUDA profiling tools (NSight)
4. Weeks of dedicated compute time

---

## Experiments Overview

| Experiment | Purpose | Status | Required Resources |
|------------|---------|--------|-------------------|
| `gpu_timing_a100.py` | Verify constant-time on A100 | 🔬 Placeholder | A100 GPU, NSight |
| `gpu_timing_h100.py` | Verify constant-time on H100 | 🔬 Placeholder | H100 GPU, NSight |
| `memory_bandwidth.py` | Analyze memory bottlenecks | 🔬 Placeholder | GPU profiler |
| `wallclock_speedup.py` | Measure actual speedup | 🔬 Placeholder | Multiple GPUs |

---

## Experiment 1: Constant-Time Verification on A100

### Purpose
Verify that θMN achieves constant execution time regardless of input values, which is essential for the timing-safety claim (Theorem 2).

### Protocol

```
1. ENVIRONMENT SETUP
   - Use dedicated A100 GPU (no other processes)
   - Disable GPU boost clocks: nvidia-smi -lgc <fixed_clock>
   - Set persistence mode: nvidia-smi -pm 1
   - Warm up GPU with 1000 dummy iterations

2. INPUT GENERATION
   Generate diverse input sets to test timing invariance:
   - all_zeros: torch.zeros(d)
   - all_ones: torch.ones(d)
   - random_normal: torch.randn(d)
   - random_uniform: torch.rand(d)
   - sparse_10pct: 90% zeros, 10% random
   - sparse_1pct: 99% zeros, 1% random
   - adversarial: Designed to trigger worst-case behavior

3. MEASUREMENT PROCEDURE
   For each input type:
   a. Run 100 warmup iterations (discard)
   b. Run 10,000 timed iterations
   c. Use CUDA events for timing:
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      output = model.forward_and_update(input, target)
      end.record()
      torch.cuda.synchronize()
      elapsed = start.elapsed_time(end)
   d. Record all 10,000 timing measurements

4. STATISTICAL ANALYSIS
   For each input type, compute:
   - Mean execution time
   - Standard deviation
   - Coefficient of variation (CV = std/mean)
   - 95% confidence interval
   - Min/max values
   - Histogram of timing distribution

5. TIMING INVARIANCE TEST
   - ANOVA test across all input types
   - Null hypothesis: All input types have same mean timing
   - Significance level: α = 0.01
   - If p > 0.01, timing is statistically invariant

6. MUTUAL INFORMATION ESTIMATION
   - Bin timing measurements
   - Estimate MI(T; X) using KSG estimator
   - Target: MI < 0.01 bits

7. SUCCESS CRITERIA
   - CV < 1% for all input types
   - ANOVA p-value > 0.01
   - MI(T; X) < 0.01 bits
   - No statistically significant timing difference between input types
```

### Expected Output Format

```json
{
  "experiment": "constant_time_a100",
  "date": "YYYY-MM-DD",
  "hardware": {
    "gpu": "NVIDIA A100-SXM4-80GB",
    "cuda_version": "12.x",
    "driver_version": "xxx.xx"
  },
  "parameters": {
    "d": 4096,
    "r": 512,
    "num_trials": 10000,
    "warmup_trials": 100
  },
  "results": {
    "all_zeros": {"mean_ms": 0.xxx, "std_ms": 0.xxx, "cv": 0.xxx},
    "all_ones": {"mean_ms": 0.xxx, "std_ms": 0.xxx, "cv": 0.xxx},
    "random_normal": {"mean_ms": 0.xxx, "std_ms": 0.xxx, "cv": 0.xxx}
  },
  "statistical_tests": {
    "anova_p_value": 0.xxx,
    "mutual_information_bits": 0.xxx
  },
  "conclusion": "PASS/FAIL"
}
```

---

## Experiment 2: Constant-Time Verification on H100

Same protocol as Experiment 1, but on H100 hardware. This verifies that timing-safety holds across different GPU architectures.

### Additional H100-Specific Considerations
- H100 has different memory hierarchy (HBM3 vs HBM2e)
- Transformer Engine may affect timing
- Test with and without FP8 quantization

---

## Experiment 3: Memory Bandwidth Analysis

### Purpose
Determine whether θMN is compute-bound or memory-bound, which affects actual speedup vs theoretical speedup.

### Protocol

```
1. ROOFLINE ANALYSIS
   - Measure achieved FLOPS
   - Measure memory bandwidth utilization
   - Plot on roofline model
   - Determine operational intensity

2. KERNEL PROFILING
   Using NSight Compute:
   - Profile each operation separately
   - Identify bottlenecks
   - Compare to theoretical peak

3. MEMORY ACCESS PATTERNS
   - Verify all accesses are to fixed addresses
   - Check for cache effects
   - Ensure no data-dependent memory patterns
```

### Required Tools
- NSight Compute
- NSight Systems
- PyTorch Profiler

---

## Experiment 4: Wall-Clock Speedup Measurement

### Purpose
Measure actual speedup over Transformer baseline, accounting for all real-world factors.

### Protocol

```
1. BASELINE IMPLEMENTATION
   - Standard Transformer with FlashAttention-2
   - Optimized KV-cache implementation
   - This is a STRONG baseline (not strawman)

2. θMN IMPLEMENTATION
   - Optimized CUDA kernels (not naive PyTorch)
   - Memory-efficient implementation
   - Fair comparison (same optimization effort)

3. MEASUREMENT
   For context lengths n ∈ [1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K]:
   a. Measure Transformer latency per token
   b. Measure θMN latency per token
   c. Compute speedup ratio
   d. Compare to theoretical prediction (n/r)

4. REPORT
   - Actual vs theoretical speedup
   - Identify gaps and explain causes
   - Memory bandwidth limitations
   - Parallelization inefficiencies
```

### Expected Results

Based on similar work (TTT-E2E reports 2.7× at 128K), we expect:
- Theoretical speedup: n/r (e.g., 256× at 128K with r=512)
- Actual speedup: Likely 2-10× due to memory bandwidth

**This gap between theoretical and actual is expected and must be honestly reported.**

---

## How These Experiments Connect to the Proposal

| Experiment | Validates | Proposal Reference |
|------------|-----------|-------------------|
| Constant-time A100 | Theorem 2 assumption | Assumption 1, Section 4 |
| Constant-time H100 | Theorem 2 generality | Section 9, Phase 2 |
| Memory bandwidth | Practical limitations | Section 8, Caveat |
| Wall-clock speedup | Theorem 3 practicality | Table 12, Section 6 |

---

## What Success Looks Like

### Best Case
- CV < 0.5% across all input types
- ANOVA p > 0.05
- MI < 0.001 bits
- Actual speedup within 50% of theoretical

### Acceptable Case
- CV < 2% across all input types
- ANOVA p > 0.01
- MI < 0.1 bits
- Actual speedup at least 2× over Transformer at 128K

### Failure Cases (Must Report Honestly)
- CV > 5% suggests timing variation
- ANOVA p < 0.001 suggests input-dependent timing
- MI > 1 bit suggests significant leakage
- No speedup over optimized Transformer

**If experiments fail, this must be reported honestly in the thesis.**

---

## Timeline

| Month | Activity |
|-------|----------|
| 7 | Set up measurement infrastructure |
| 8 | Constant-time experiments (A100) |
| 9 | Constant-time experiments (H100) |
| 10 | Memory bandwidth analysis |
| 11 | Wall-clock speedup measurement |
| 12 | Analysis and documentation |

---

## Required Resources

### Hardware
- 1× NVIDIA A100 (80GB) - dedicated access for timing experiments
- 1× NVIDIA H100 - for architecture comparison
- Isolated environment (no other users/processes)

### Software
- CUDA Toolkit 12.x
- PyTorch 2.x with CUDA support
- NSight Compute and NSight Systems
- Python scientific stack (numpy, scipy, matplotlib)

### Compute Time
- Estimated 2-4 weeks of dedicated GPU time
- Statistical significance requires many trials

---

## Files in This Directory

| File | Status | Description |
|------|--------|-------------|
| `README.md` | ✅ Complete | This file - protocols |
| `gpu_timing_a100.py` | 🔬 Placeholder | A100 timing experiment |
| `gpu_timing_h100.py` | 🔬 Placeholder | H100 timing experiment |
| `memory_bandwidth.py` | 🔬 Placeholder | Bandwidth analysis |
| `wallclock_speedup.py` | 🔬 Placeholder | Speedup measurement |
| `statistical_analysis.py` | 🔬 Placeholder | Statistical tests |

---

## Citation

If these protocols are used, cite:

```
θ-Learning Research Proposal, Section 9: Research Plan, Phase 2
```
