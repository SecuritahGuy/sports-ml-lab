# Calibration Remediation

*Model: v3.0.0 Frozen QB Overlay*

## Research Question

Can we reduce calibration error for v3.0.0, especially QB-change / gate-active games, without worsening validation log loss or holdout log loss?

## Governance Trigger

Calibration audit (ECE=0.0628, MCE=0.1343) identified a repeatable failure mode: QB-change games have ECE=0.2097 and MCE=0.5690 on 2025 holdout (N=55). This qualifies for remediation under the "repeatable failure mode" trigger.

## Methods Tested

| # | Method | Variants |
|---|--------|----------|
| 1 | **Baseline** | v3.0.0 unchanged |
| 2 | **Global temperature scaling** | T ∈ {0.8, 0.9, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0} |
| 3 | **Gate-aware T** | separate T for gate-active / gate-inactive |
| 4 | **QB-change-aware T** | separate T for QB-change / non-QB-change |
| 5 | **QB-change shrink →0.5** | α ∈ {0.05, 0.10, 0.15, 0.20} |
| 6 | **QB-change shrink →base** | α ∈ {0.05, 0.10, 0.15, 0.20} |

Total variants: 65

## Validation (Rolling-Origin, 3 folds)

Selection criterion: average validation log loss.

| Variant | Avg Val LL | ECE | MCE | Rel |
|---------|-----------|-----|-----|-----|
| Global T=0.9 | 0.6304 | 0.0699 | 0.2491 | 0.0093  ← **SELECTED**|
| Baseline (v3.0.0) | 0.6305 | 0.0595 | 0.2274 | 0.0062 |
| QB shrink →base α=0.05 | 0.6307 | 0.0616 | 0.2816 | 0.0069 |
| QB shrink →0.5 α=0.05 | 0.6308 | 0.0616 | 0.2819 | 0.0069 |
| QB shrink →base α=0.1 | 0.6309 | 0.0629 | 0.2250 | 0.0066 |
| QB shrink →0.5 α=0.1 | 0.6312 | 0.0598 | 0.2250 | 0.0065 |
| QB shrink →base α=0.15 | 0.6313 | 0.0634 | 0.2254 | 0.0068 |
| Global T=1.1 | 0.6315 | 0.0595 | 0.2337 | 0.0061 |
| QB shrink →0.5 α=0.15 | 0.6317 | 0.0652 | 0.2250 | 0.0068 |
| QB shrink →base α=0.2 | 0.6317 | 0.0669 | 0.2315 | 0.0074 |
| Global T=0.8 | 0.6318 | 0.0707 | 0.2504 | 0.0099 |
| QB shrink →0.5 α=0.2 | 0.6323 | 0.0654 | 0.2319 | 0.0071 |
| Gate T: gate=1.0, no_gate=1.5 | 0.6327 | 0.0655 | 0.2466 | 0.0075 |
| Global T=1.2 | 0.6329 | 0.0681 | 0.1911 | 0.0073 |
| QB T: qb=1.5, no_qb=1.0 | 0.6332 | 0.0637 | 0.2525 | 0.0071 |
| Gate T: gate=1.0, no_gate=2.0 | 0.6345 | 0.0657 | 0.2631 | 0.0085 |
| QB T: qb=1.0, no_qb=1.5 | 0.6355 | 0.0715 | 0.2175 | 0.0074 |
| Gate T: gate=1.5, no_gate=1.0 | 0.6360 | 0.0744 | 0.2407 | 0.0085 |
| QB T: qb=2.0, no_qb=1.0 | 0.6361 | 0.0584 | 0.2557 | 0.0072 |
| Gate T: gate=1.0, no_gate=3.0 | 0.6369 | 0.0679 | 0.2631 | 0.0081 |
| Global T=1.5 | 0.6382 | 0.0778 | 0.2452 | 0.0093 |
| Gate T: gate=1.5, no_gate=1.5 | 0.6382 | 0.0778 | 0.2452 | 0.0093 |
| QB T: qb=1.5, no_qb=1.5 | 0.6382 | 0.0778 | 0.2452 | 0.0093 |
| Gate T: gate=1.0, no_gate=5.0 | 0.6393 | 0.0735 | 0.2631 | 0.0087 |
| Gate T: gate=1.5, no_gate=2.0 | 0.6400 | 0.0809 | 0.2438 | 0.0099 |
| QB T: qb=3.0, no_qb=1.0 | 0.6403 | 0.0675 | 0.2555 | 0.0082 |
| QB T: qb=1.0, no_qb=2.0 | 0.6410 | 0.0843 | 0.2498 | 0.0102 |
| QB T: qb=2.0, no_qb=1.5 | 0.6411 | 0.0815 | 0.2252 | 0.0095 |
| Gate T: gate=1.5, no_gate=3.0 | 0.6425 | 0.0847 | 0.2438 | 0.0099 |
| Gate T: gate=2.0, no_gate=1.0 | 0.6426 | 0.0791 | 0.2311 | 0.0096 |
| QB T: qb=1.5, no_qb=2.0 | 0.6437 | 0.0901 | 0.2603 | 0.0116 |
| QB T: qb=5.0, no_qb=1.0 | 0.6446 | 0.0724 | 0.2555 | 0.0081 |
| Gate T: gate=2.0, no_gate=1.5 | 0.6448 | 0.0902 | 0.2876 | 0.0117 |
| Gate T: gate=1.5, no_gate=5.0 | 0.6448 | 0.0903 | 0.2438 | 0.0108 |
| QB T: qb=3.0, no_qb=1.5 | 0.6453 | 0.0858 | 0.2199 | 0.0098 |
| Global T=2.0 | 0.6466 | 0.0924 | 0.2833 | 0.0121 |
| Gate T: gate=2.0, no_gate=2.0 | 0.6466 | 0.0924 | 0.2833 | 0.0121 |
| QB T: qb=2.0, no_qb=2.0 | 0.6466 | 0.0924 | 0.2833 | 0.0121 |
| QB T: qb=1.0, no_qb=3.0 | 0.6487 | 0.0920 | 0.3027 | 0.0125 |
| Gate T: gate=2.0, no_gate=3.0 | 0.6490 | 0.0962 | 0.2833 | 0.0122 |
| QB T: qb=5.0, no_qb=1.5 | 0.6496 | 0.0895 | 0.2163 | 0.0100 |
| QB T: qb=3.0, no_qb=2.0 | 0.6508 | 0.0965 | 0.3172 | 0.0130 |
| QB T: qb=1.5, no_qb=3.0 | 0.6514 | 0.0998 | 0.3175 | 0.0139 |
| Gate T: gate=2.0, no_gate=5.0 | 0.6514 | 0.1018 | 0.2833 | 0.0131 |
| Gate T: gate=3.0, no_gate=1.0 | 0.6521 | 0.0872 | 0.2345 | 0.0111 |
| Gate T: gate=3.0, no_gate=1.5 | 0.6542 | 0.0959 | 0.2890 | 0.0134 |
| QB T: qb=2.0, no_qb=3.0 | 0.6543 | 0.1004 | 0.2357 | 0.0135 |
| QB T: qb=5.0, no_qb=2.0 | 0.6551 | 0.1002 | 0.3190 | 0.0130 |
| Gate T: gate=3.0, no_gate=2.0 | 0.6561 | 0.1000 | 0.2792 | 0.0149 |
| QB T: qb=1.0, no_qb=5.0 | 0.6565 | 0.1068 | 0.2717 | 0.0148 |
| Global T=3.0 | 0.6585 | 0.1037 | 0.2674 | 0.0149 |
| Gate T: gate=3.0, no_gate=3.0 | 0.6585 | 0.1037 | 0.2674 | 0.0149 |
| QB T: qb=3.0, no_qb=3.0 | 0.6585 | 0.1037 | 0.2674 | 0.0149 |
| QB T: qb=1.5, no_qb=5.0 | 0.6591 | 0.1147 | 0.3212 | 0.0170 |
| Gate T: gate=3.0, no_gate=5.0 | 0.6608 | 0.1074 | 0.2862 | 0.0157 |
| Gate T: gate=5.0, no_gate=1.0 | 0.6618 | 0.1037 | 0.2354 | 0.0138 |
| QB T: qb=2.0, no_qb=5.0 | 0.6621 | 0.1148 | 0.2745 | 0.0160 |
| QB T: qb=5.0, no_qb=3.0 | 0.6628 | 0.1078 | 0.2925 | 0.0156 |
| Gate T: gate=5.0, no_gate=1.5 | 0.6639 | 0.1099 | 0.3236 | 0.0150 |
| Gate T: gate=5.0, no_gate=2.0 | 0.6658 | 0.1140 | 0.3075 | 0.0162 |
| QB T: qb=3.0, no_qb=5.0 | 0.6662 | 0.1186 | 0.3097 | 0.0174 |
| Gate T: gate=5.0, no_gate=3.0 | 0.6682 | 0.1177 | 0.3087 | 0.0169 |
| Global T=5.0 | 0.6706 | 0.1214 | 0.3909 | 0.0178 |
| Gate T: gate=5.0, no_gate=5.0 | 0.6706 | 0.1214 | 0.3909 | 0.0178 |
| QB T: qb=5.0, no_qb=5.0 | 0.6706 | 0.1214 | 0.3909 | 0.0178 |

### Fold Details (Best Variant)

**Global T=0.9**

| Fold | Val N | LL | ECE | MCE |
|------|-------|-----|-----|-----|
| 2022 | 275 | 0.6351 | 0.0803 | 0.2496 |
| 2023 | 279 | 0.6633 | 0.0794 | 0.3411 |
| 2024 | 278 | 0.5929 | 0.0499 | 0.1567 |

### Fold Details (Baseline)

| Fold | Val N | LL | ECE | MCE |
|------|-------|-----|-----|-----|
| 2022 | 275 | 0.6360 | 0.0663 | 0.1938 |
| 2023 | 279 | 0.6596 | 0.0602 | 0.3210 |
| 2024 | 278 | 0.5960 | 0.0521 | 0.1675 |

## Holdout (2025) Results

| Metric | Baseline | Selected |
|--------|----------|----------|
| Log loss | 0.6200 | 0.6229 |
| Brier | 0.2157 | 0.2166 |
| AUC | 0.7098 | 0.7098 |
| Accuracy | 0.6630 | 0.6630 |
| ECE | 0.0628 | 0.0654 |
| MCE | 0.1343 | 0.1309 |
| Reliability | 0.0053 | 0.0054 |
| N | 276 | 276 |

### QB-Change Subset Calibration

| Metric | Baseline | Selected |
|--------|----------|----------|
| N | 55 | 55 |
| ECE | 0.2097 | 0.2076 |
| MCE | 0.5690 | 0.5766 |

## Leakage Risk

- Temperature scaling is fit to validation data per-fold (0 parameters — single T is a hyperparameter sweep, not a learned parameter).
- No 2025 holdout data accessed during fold validation.
- No market features used as model inputs.
- No new feature families introduced.
- Selection by validation LL, not by ECE/MCE or holdout.

## Decision

**❌ REJECTED** — no variant beats baseline on both validation and holdout. Val LL: 0.6304 vs 0.6305. Holdout LL: 0.6229 vs 0.6200.

Best variant: **Global T=0.9**

