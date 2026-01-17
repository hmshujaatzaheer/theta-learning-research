# Phase 3: Security Evaluation

## Status: 🔬 PLACEHOLDER - Requires Lab Execution

This directory contains experimental protocols for Phase 3 of the research plan (Months 13-18).

**These experiments CANNOT be completed without:**
1. Successful completion of Phase 2 (hardware validation)
2. Reproduction of PROMPTPEEK attack baseline
3. Statistical expertise for mutual information estimation
4. Isolated security testing environment

---

## Security Evaluation Overview

The goal of Phase 3 is to **empirically validate** the timing-safety claims proven mathematically in Theorem 2.

### What We're Testing

| Claim | Theorem | Test |
|-------|---------|------|
| Storage-based memory leaks timing | Theorem 1 | Reproduce PROMPTPEEK |
| θMN achieves MI(T;X) = 0 | Theorem 2 | Attack θMN, measure MI |
| No covert channel exists | Theorem 2 corollary | Channel capacity estimation |

---

## Experiment 1: PROMPTPEEK Reproduction

### Purpose
Reproduce the PROMPTPEEK attack (Wu et al., NDSS 2025) to establish a baseline and verify our measurement methodology.

### Protocol

```
1. SETUP
   - Deploy standard Transformer with KV-cache sharing
   - Configure as multi-tenant (victim + attacker)
   - Use Llama-2-7B or similar model

2. ATTACK IMPLEMENTATION
   - Attacker measures TTFT for various prefixes
   - Record timing with microsecond precision
   - Build timing profile database

3. ATTACK EXECUTION
   - Victim submits private prompts
   - Attacker measures TTFT
   - Attempt prompt reconstruction

4. SUCCESS METRIC
   - Must achieve ≥90% accuracy (paper claims 95-99%)
   - If we can't reproduce, our methodology is flawed
```

### Expected Results
- Reconstruction accuracy: 90-99%
- Clear timing difference between cache hit/miss
- This establishes that timing attacks ARE possible on Transformers

---

## Experiment 2: Attack θMN

### Purpose
Attempt the same attack methodology on θMN to demonstrate timing-safety.

### Protocol

```
1. SETUP
   - Deploy θMN with same configuration
   - Same multi-tenant setup as Experiment 1
   - Same measurement methodology

2. ATTACK EXECUTION
   - Attacker measures TTFT for θMN
   - Attempt same reconstruction attack
   - Record all timing measurements

3. ANALYSIS
   - Compare timing distributions for different inputs
   - Attempt prompt reconstruction
   - Calculate attack success rate

4. SUCCESS CRITERIA (for θMN security)
   - Attack accuracy ≤ random guessing
   - No statistically significant timing difference
   - Attacker cannot distinguish between prompts
```

### Expected Results
- Reconstruction accuracy: ~random (depending on vocabulary size)
- No correlation between prompt content and timing
- Timing distribution identical across all inputs

---

## Experiment 3: Mutual Information Estimation

### Purpose
Quantify the information leakage through timing measurements.

### Protocol

```
1. DATA COLLECTION
   For N different inputs x_i:
   - Run θMN M times each
   - Record timing t_ij for each run
   - Build joint distribution P(T, X)

2. MI ESTIMATION
   Use Kraskov-Stögbauer-Grassberger (KSG) estimator:
   - Handles continuous timing measurements
   - Corrects for finite sample bias
   - Provides confidence intervals

3. INTERPRETATION
   - MI = 0: No information leakage (ideal)
   - MI > 0: Some leakage exists
   - Compare to Transformer baseline

4. SAMPLE SIZE
   - Need N ≥ 1000 different inputs
   - Need M ≥ 100 trials per input
   - Total: 100,000+ timing measurements
```

### Mathematical Background

Mutual Information:
```
MI(T; X) = H(T) - H(T|X)
         = Σ_x P(x) Σ_t P(t|x) log[P(t|x) / P(t)]
```

For timing-safety (Theorem 2):
```
MI(T; X) = 0  ⟺  P(t|x) = P(t) for all x
               ⟺  Timing is independent of input
```

### Expected Results

| Architecture | Expected MI | Interpretation |
|-------------|-------------|----------------|
| Transformer + KV-cache | > 1 bit | Significant leakage |
| θMN (under Assumption 1) | < 0.01 bits | Negligible leakage |

---

## Experiment 4: Covert Channel Capacity

### Purpose
Estimate the maximum rate at which information could leak through timing.

### Protocol

```
1. CHANNEL MODEL
   - Treat timing as a noisy channel
   - Input: secret data X
   - Output: timing measurement T
   - Noise: measurement variance

2. CAPACITY ESTIMATION
   C = max_{P(X)} MI(T; X)
   
   - Find input distribution maximizing MI
   - This gives upper bound on leakage rate

3. PRACTICAL IMPLICATIONS
   - If C < 1 bit/query: very slow leakage
   - If C > 10 bits/query: fast leakage, serious concern
```

### Expected Results
- Transformer: High capacity (enables PROMPTPEEK)
- θMN: Near-zero capacity (if Theorem 2 holds)

---

## Adversarial Considerations

### What Adversary Can We Defend Against?

| Adversary Capability | Defended? | Notes |
|---------------------|-----------|-------|
| Measure TTFT | ✓ Yes | Constant-time operations |
| Measure per-token latency | ✓ Yes | Constant-time operations |
| Observe memory access patterns | ✓ Yes | Fixed addresses |
| Side-channel on arithmetic | ? | Depends on Assumption 1 |
| Physical side-channels (power, EM) | ✗ No | Out of scope |

### Assumption 1 Violations

If Assumption 1 (constant-time arithmetic) is violated:

```
Potential violations:
1. Denormalized floating-point numbers
   - Some GPUs handle denorms slowly
   - Mitigation: Flush denorms to zero

2. NaN/Inf handling
   - Special values may take different time
   - Mitigation: Input validation

3. Memory coalescing
   - Unaligned access patterns
   - Mitigation: Ensure aligned access

4. Warp divergence
   - Different threads take different paths
   - θMN has no divergence (all threads same ops)
```

---

## Honest Assessment of Limitations

### What This Evaluation CANNOT Prove

1. **Perfect security**: Real-world systems have many attack surfaces
2. **Implementation correctness**: Bugs could introduce vulnerabilities
3. **Future attacks**: New attack techniques may emerge
4. **Hardware variations**: Different GPUs may behave differently

### What This Evaluation CAN Show

1. **No timing side-channel via TTFT**: Under our threat model
2. **Improvement over Transformer**: Quantified reduction in MI
3. **Assumption validation**: Whether constant-time holds in practice

---

## Files in This Directory

| File | Status | Description |
|------|--------|-------------|
| `README.md` | ✅ Complete | This file - protocols |
| `promptpeek_baseline.py` | 🔬 Placeholder | PROMPTPEEK reproduction |
| `attack_theta_mn.py` | 🔬 Placeholder | Attack attempt on θMN |
| `timing_statistics.py` | 🔬 Placeholder | MI estimation |
| `covert_channel.py` | 🔬 Placeholder | Channel capacity |

---

## Timeline

| Month | Activity |
|-------|----------|
| 13 | Set up security evaluation environment |
| 14 | Reproduce PROMPTPEEK attack |
| 15 | Attack θMN, collect timing data |
| 16 | Mutual information analysis |
| 17 | Covert channel analysis |
| 18 | Documentation and paper writing |

---

## Required Resources

### Hardware
- Multi-tenant GPU setup (to simulate attack scenario)
- High-precision timing measurement capability
- Isolated network (to prevent external interference)

### Software
- PROMPTPEEK reproduction code (from Wu et al.)
- Information-theoretic analysis tools
- Statistical analysis framework

### Expertise
- Security evaluation experience
- Statistical analysis (MI estimation)
- GPU programming (for measurement)

---

## Citation

If these protocols are used, cite:

```
θ-Learning Research Proposal, Section 9: Research Plan, Phase 3

Wu et al., "I Know What You Asked: Prompt Leakage via KV-Cache Sharing 
in Multi-Tenant LLM Serving", NDSS 2025
```
