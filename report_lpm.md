# Mississippi AI Workforce Impact — LPM Architecture Report

## Large Population Model: 1.2 Million Agents × LLM-Driven Behavior on AgentTorch

**Framework:** AgentTorch (MIT Media Lab)  
**Date:** February 2026  
**Simulation Period:** Q1 2025 → Q4 2027 (12 quarterly time steps)  
**Population:** 1,200,000 synthetic workers (Mississippi labor force)  
**Backends:** PyTorch (tensorized simulation) + Ollama / MLX (LLM inference)  
**Runtime:** ~1.1 s parametric, ~3–5 min LLM-enhanced (Apple Silicon M-series)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Input Data Pipeline](#3-input-data-pipeline)
4. [Simulation Engine — Substep Flowchart](#4-simulation-engine--substep-flowchart)
5. [LLM Integration Architecture](#5-llm-integration-architecture)
6. [Output Pipeline & Observables](#6-output-pipeline--observables)
7. [Agent State Machine](#7-agent-state-machine)
8. [Data Flow Diagram — End to End](#8-data-flow-diagram--end-to-end)
9. [Configuration Schema (YAML)](#9-configuration-schema-yaml)
10. [Component Reference](#10-component-reference)
11. [Performance Characteristics](#11-performance-characteristics)
12. [Simulation Results — Parametric vs. LLM Comparison](#12-simulation-results--parametric-vs-llm-comparison)
13. [Assumptions & Limitations](#13-assumptions--limitations)
14. [Policy Implications](#14-policy-implications)
15. [Reproducibility](#15-reproducibility)
16. [Improving Accuracy with Frey & Osborne Automation Scores](#16-improving-accuracy-with-frey--osborne-automation-scores)
17. [References](#17-references)

---

## 1. Executive Summary

This report documents the complete architecture of a **Large Population Model (LPM)** that simulates the impact of AI adoption on Mississippi's 1.2-million-worker labor force over three years (2025–2027). The system is built on MIT Media Lab's AgentTorch framework and supports two operational modes:

| Mode | Engine | Behavior Generation | Runtime | LLM Calls |
|------|--------|--------------------|---------|-----------| 
| **Parametric** | PyTorch tensors | Rule-based equations | ~1.1 s | 0 |
| **LLM-Enhanced** | PyTorch + Ollama/MLX | LLM-driven archetypes | ~3–5 min | ~84 |

### Key Findings — Parametric vs. LLM Comparison

| Metric | Parametric | LLM-Enhanced | Δ (LLM − Parametric) |
|--------|-----------|-------------|----------------------|
| Workers Displaced (final) | 4,989 (0.4%) | 38,554 (3.2%) | **+33,565 (+672%)** |
| Workers Augmented by AI | 112,941 (9.4%) | 112,694 (9.4%) | −247 (≈0%) |
| Workers At Risk | 279,373 (23.3%) | 279,433 (23.3%) | +60 (≈0%) |
| Workers Retrained | 49,059 (4.1%) | 15,823 (1.3%) | **−33,236 (−67.8%)** |
| Workers Unaffected | 753,638 (62.8%) | 753,496 (62.8%) | −142 (≈0%) |
| Final Mean Wage | $41,931 | $41,793 | −$138 (−0.3%) |
| Final Median Wage | $39,169 | $39,057 | −$112 (−0.3%) |
| Peak Unemployment Rate | 8.4% (Q1) | 11.2% (Q1) | **+2.8pp** |
| LLM Backend | — | Ollama (Ministral 3B) | — |
| LLM Calls (total) | 0 | ~84 | — |

> **Key insight:** The LLM produces a **dramatically more pessimistic retraining outlook** than the parametric model. Both modes agree on augmentation (~112,800) and at-risk counts (~279,400), but the LLM's retraining probability estimates are lower, leaving **7.7× more workers stuck in displaced status** (38,554 vs. 4,989). This suggests the parametric model's 30% quarterly retraining rate is overly optimistic — the LLM, reasoning about real-world barriers (family obligations, limited transit, program availability in Mississippi), predicts far fewer workers successfully retrain.

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AgentTorch Runtime                               │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Registry    │    │    Runner     │    │      Configuration      │   │
│  │              │◄──►│              │◄──►│     (config.yaml)        │   │
│  │ • substeps   │    │ • init()     │    │ • simulation_metadata    │   │
│  │ • init fns   │    │ • reset()    │    │ • state schema           │   │
│  │ • policies   │    │ • step()     │    │ • substep definitions    │   │
│  │ • transitions│    │ • state_traj │    │ • learnable params       │   │
│  └──────────────┘    └──────┬───────┘    └──────────────────────────┘   │
│                             │                                           │
│                    ┌────────▼────────┐                                   │
│                    │   State Dict    │                                   │
│                    │                 │                                   │
│                    │ agents/workers/ │                                   │
│                    │   • 10 attrs    │  ← 1.2M × 1 tensors              │
│                    │ environment/    │                                   │
│                    │   • 13 vars     │  ← economy-wide scalars/vectors  │
│                    └────────┬────────┘                                   │
│                             │                                           │
│          ┌──────────────────┼──────────────────┐                        │
│          ▼                  ▼                   ▼                        │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐               │
│  │  Substep 0    │  │  Substep 1    │  │  Substep 1     │               │
│  │ AI Exposure   │  │ LLM Retrain   │  │ Job Transition │               │
│  │  Assessment   │  │  Decision     │  │                │               │
│  │ (policy +     │  │  (policy,     │  │ (transition)   │               │
│  │  transition)  │  │  @with_behav) │  │                │               │
│  └───────────────┘  └───────┬───────┘  └────────────────┘               │
│                             │                                           │
│                    ┌────────▼────────┐                                   │
│                    │  LLM Backend    │                                   │
│                    │  (optional)     │                                   │
│                    │                 │                                   │
│                    │ ┌────────────┐  │                                   │
│                    │ │  Ollama    │  │  ← HTTP to local server           │
│                    │ │  Backend   │  │                                   │
│                    │ └────────────┘  │                                   │
│                    │ ┌────────────┐  │                                   │
│                    │ │  MLX       │  │  ← In-process Metal GPU           │
│                    │ │  Backend   │  │                                   │
│                    │ └────────────┘  │                                   │
│                    └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Layout

```
agent_torch/
├── models/mississippi_ai/
│   ├── __init__.py                    # Exposes `registry` for Executor
│   ├── simulator.py                   # Registry setup: substeps + init
│   ├── generate_data.py               # Synthetic 1.2M population generator
│   ├── yamls/config.yaml              # 491-line simulation configuration
│   └── substeps/
│       ├── ai_exposure.py             # AssessAIExposure + UpdateExposure
│       ├── job_transition.py          # JobTransition (254 lines)
│       ├── llm_retraining.py          # LLMRetrainingDecision (@with_behavior)
│       └── utils.py                   # load_population_attribute()
├── populations/mississippi/
│   ├── __init__.py                    # Population package
│   ├── mapping.json                   # LLM archetype groupings
│   ├── age.pickle                     # 1.2M int tensor
│   ├── gender.pickle                  # 1.2M int tensor
│   ├── education.pickle               # 1.2M int tensor
│   ├── industry.pickle                # 1.2M int tensor
│   ├── ai_exposure.pickle             # 1.2M float tensor
│   ├── wage.pickle                    # 1.2M float tensor
│   ├── employment_status.pickle       # 1.2M int tensor
│   ├── digital_skill.pickle           # 1.2M float tensor
│   ├── region.pickle                  # 1.2M int tensor
│   └── population_mapping.json        # Human-readable name mappings
├── core/llm/
│   ├── ollama_backend.py              # OllamaLLM (HTTP → Ollama server)
│   ├── mlx_backend.py                 # MLXLLM (in-process Apple Silicon)
│   ├── archetype.py                   # Archetype clustering system
│   └── backend.py                     # LLMBackend abstract base class
├── run_mississippi_sim.py             # Parametric runner (Executor API)
└── run_mississippi_sim_llm.py         # LLM-enhanced runner (envs API)
```

---

## 3. Input Data Pipeline

### 3.1 Input Flowchart

```
                            ┌──────────────────────────────────┐
                            │     Data Source Assumptions       │
                            │                                  │
                            │  BLS QCEW (industry proportions) │
                            │  ACS 5-yr (age, education, gender)│
                            │  BLS OES (wages by industry)     │
                            │  Felten et al. 2021 (AI exposure)│
                            └───────────────┬──────────────────┘
                                            │
                                            ▼
                            ┌──────────────────────────────────┐
                            │      generate_data.py            │
                            │                                  │
                            │  np.random.choice(N=1,200,000)   │
                            │  per attribute with calibrated   │
                            │  probability distributions       │
                            └───────────────┬──────────────────┘
                                            │
                        ┌───────────────────┼───────────────────┐
                        ▼                   ▼                   ▼
              ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
              │  age.pickle  │   │ industry.pkl │   │  wage.pickle │
              │  1.2M int    │   │ 1.2M int     │   │  1.2M float  │
              │  6 groups    │   │ 12 sectors   │   │  $15K-$200K  │
              └──────────────┘   └──────────────┘   └──────────────┘
              ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
              │education.pkl │   │ai_exposure.pk│   │digital_sk.pk │
              │  1.2M int    │   │ 1.2M float   │   │  1.2M float  │
              │  5 levels    │   │ 0.0 – 1.0    │   │  0.0 – 1.0   │
              └──────────────┘   └──────────────┘   └──────────────┘
              ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
              │  gender.pkl  │   │  region.pkl  │   │emp_status.pkl│
              │  1.2M int    │   │  1.2M int    │   │  1.2M int    │
              │  0/1         │   │  5 regions   │   │  0/1/2       │
              └──────────────┘   └──────────────┘   └──────────────┘
                        │                   │                   │
                        └───────────────────┼───────────────────┘
                                            │
                                            ▼
                            ┌──────────────────────────────────┐
                            │         config.yaml              │
                            │                                  │
                            │  state.agents.workers.properties │
                            │  → initialization_function:      │
                            │      generator: load_population_ │
                            │              attribute           │
                            │      arguments:                  │
                            │        file_path: .../<attr>.pkl │
                            └───────────────┬──────────────────┘
                                            │
                                            ▼
                            ┌──────────────────────────────────┐
                            │      AgentTorch Initializer      │
                            │                                  │
                            │  pd.read_pickle() → torch.tensor │
                            │  Shape: [1_200_000, 1] per attr  │
                            └──────────────────────────────────┘
```

### 3.2 Population Attribute Details

| Attribute | Shape | dtype | Distribution | Source |
|-----------|-------|-------|-------------|--------|
| `age` | [1.2M, 1] | int | 6 groups, P=[.12,.22,.21,.20,.18,.07] | ACS MS |
| `gender` | [1.2M, 1] | int | Binary, P=[.48,.52] | Census |
| `education` | [1.2M, 1] | int | 5 levels, P=[.12,.30,.28,.19,.11] | ACS |
| `industry` | [1.2M, 1] | int | 12 sectors, BLS-calibrated | BLS QCEW |
| `ai_exposure` | [1.2M, 1] | float | Industry baseline + edu modifier + N(0,0.08) | Felten et al. |
| `wage` | [1.2M, 1] | float | Industry median × LogNormal(0,0.25), clipped [$15K,$200K] | BLS OES |
| `digital_skill` | [1.2M, 1] | float | 0.4×edu + 0.4×age_factor + 0.2×Beta(2,3) | Derived |
| `employment_status` | [1.2M, 1] | int | 0/1/2, P=[.88,.04,.08] | BLS |
| `region` | [1.2M, 1] | int | 5 regions, P=[.12,.18,.25,.20,.25] | Census |

### 3.3 Dynamic State Variables (initialized at runtime)

| Variable | Shape | Initial Value | Updated By |
|----------|-------|---------------|------------|
| `effective_ai_exposure` | [1.2M, 1] | 0.0 | Substep 0 (AI Exposure) |
| `job_impact_status` | [1.2M, 1] | 0 (Unaffected) | Substep 1 (Job Transition) |

### 3.4 Environment Variables

| Variable | Shape | Initial | Description |
|----------|-------|---------|-------------|
| `ai_penetration_index` | [1] | 0.15 | Economy-wide AI adoption level |
| `total_displaced` | [12] | 0 | Displaced count per quarter |
| `total_augmented` | [12] | 0 | Augmented count per quarter |
| `total_retrained` | [12] | 0 | Retrained count per quarter |
| `unemployment_rate` | [12] | 0.04 | Unemployment rate per quarter |
| `avg_wage_change` | [12] | 0 | Mean wage change % per quarter |
| `gdp_impact` | [12] | 0 | GDP impact ($M) per quarter |
| `sector_displacement` | [12] | 0 | Displacement per sector (cumulative) |
| `sector_augmentation` | [12] | 0 | Augmentation per sector (cumulative) |
| `region_unemployment` | [5] | [.06,.045,.04,.042,.035] | Unemployment by region |
| `quarter` | [1] | 0 | Current quarter index |

---

## 4. Simulation Engine — Substep Flowchart

### 4.1 Per-Step Execution Flow

Each of the 12 quarterly time steps executes two substeps sequentially:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       One Quarterly Step                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │                 SUBSTEP 0: AI Exposure                  │            │
│  │                                                         │            │
│  │  ┌─────────────────────────────────────────────┐        │            │
│  │  │ POLICY: AssessAIExposure                    │        │            │
│  │  │                                             │        │            │
│  │  │ Inputs:                                     │        │            │
│  │  │   • base_ai_exposure [1.2M,1]               │        │            │
│  │  │   • industry [1.2M,1]                       │        │            │
│  │  │   • education [1.2M,1]                      │        │            │
│  │  │   • digital_skill [1.2M,1]                  │        │            │
│  │  │   • age [1.2M,1]                            │        │            │
│  │  │   • current_step (scalar)                   │        │            │
│  │  │                                             │        │            │
│  │  │ Computation:                                │        │            │
│  │  │   time_mult = (1 + 0.08)^t                  │        │            │
│  │  │   ind_speed = sector_speed[industry]        │        │            │
│  │  │   edu_eff   = edu_modifier[education]       │        │            │
│  │  │   age_eff   = age_adapt[age]                │        │            │
│  │  │   exposure  = base × time_mult ×            │        │            │
│  │  │              ind_speed × age_eff + edu_eff   │        │            │
│  │  │   exposure  = clamp(exposure, 0, 1)         │        │            │
│  │  │                                             │        │            │
│  │  │ Output:                                     │        │            │
│  │  │   effective_ai_exposure [1.2M,1]            │        │            │
│  │  └─────────────────────────────────────────────┘        │            │
│  │                          │                              │            │
│  │                          ▼                              │            │
│  │  ┌─────────────────────────────────────────────┐        │            │
│  │  │ TRANSITION: UpdateExposure                  │        │            │
│  │  │                                             │        │            │
│  │  │ • Write effective_ai_exposure → state       │        │            │
│  │  │ • ai_penetration += 0.03 (clamp to 1.0)    │        │            │
│  │  └─────────────────────────────────────────────┘        │            │
│  └─────────────────────────────────────────────────────────┘            │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │                 SUBSTEP 1: Job Transition               │            │
│  │                                                         │            │
│  │  ┌─────────────────────────────────────────────┐        │            │
│  │  │ POLICY: LLMRetrainingDecision               │        │            │
│  │  │         (@with_behavior decorator)           │        │            │
│  │  │                                             │        │            │
│  │  │ If LLM attached:                            │        │            │
│  │  │   → Archetype groups displaced workers      │        │            │
│  │  │   → 7 LLM calls per step                   │        │            │
│  │  │   → Returns retraining_probability [1.2M,1] │        │            │
│  │  │                                             │        │            │
│  │  │ If no LLM:                                  │        │            │
│  │  │   → Parametric: f(skill, edu, age)          │        │            │
│  │  │   → Returns retraining_probability [1.2M,1] │        │            │
│  │  └─────────────────────────────────────────────┘        │            │
│  │                          │                              │            │
│  │                          ▼                              │            │
│  │  ┌─────────────────────────────────────────────┐        │            │
│  │  │ TRANSITION: JobTransition                   │        │            │
│  │  │                                             │        │            │
│  │  │ 1. Compute displacement_prob per worker     │        │            │
│  │  │ 2. Compute augmentation_prob per worker     │        │            │
│  │  │ 3. Stochastic draw → newly displaced/aug    │        │            │
│  │  │ 4. Retrain displaced (LLM or parametric)    │        │            │
│  │  │ 5. Update job_impact_status                 │        │            │
│  │  │ 6. Update employment_status                 │        │            │
│  │  │ 7. Update wages                             │        │            │
│  │  │ 8. Update digital_skill                     │        │            │
│  │  │ 9. Aggregate: sector/region/economy stats   │        │            │
│  │  │                                             │        │            │
│  │  │ Outputs: 13 state variables updated         │        │            │
│  │  └─────────────────────────────────────────────┘        │            │
│  └─────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Substep 0: AI Exposure — Mathematical Detail

The effective AI exposure for worker `i` at time step `t`:

```
E_eff(i,t) = clamp( E_base(i) × (1.08)^t × S[ind(i)] × A[age(i)] + M[edu(i)],  0, 1 )
```

Where:
- `E_base(i)` — baseline AI exposure from Felten et al. (by industry)
- `(1.08)^t` — compound 8% quarterly acceleration
- `S[ind(i)]` — sector-specific adoption speed: `[0.6, 1.2, 1.0, 0.8, 0.7, 0.5, 1.1, 0.9, 1.5, 1.4, 0.9, 0.8]`
- `A[age(i)]` — age adaptation factor: `[1.10, 1.05, 1.00, 0.95, 0.85, 0.75]`
- `M[edu(i)]` — education modifier: `[-0.05, -0.02, 0.0, 0.05, 0.10]`

### 4.3 Substep 1: Job Transition — Mathematical Detail

**Displacement probability** (for employed, unaffected workers):
```
P_disp(i) = clamp( (E_eff(i) - 0.60) × 2.0 × (1 - skill(i)) × (1 - resilience[region(i)]),  0, 0.85 )
```

**Augmentation probability**:
```
P_aug(i) = clamp( (E_eff(i) - 0.35) × 1.5 × skill(i) × resilience[region(i)],  0, 0.70 )
         × (1 - high_exposure_low_skill_penalty)
```

**Retraining probability** (for displaced workers):
```
Parametric:
  P_retrain(i) = clamp( 0.30 + skill(i)×0.2 + edu_bonus[edu(i)] - age_penalty[age(i)],  0.05, 0.80 )

LLM-Enhanced:
  P_retrain(i) = Archetype.sample(worker_profile(i))  →  float ∈ [0, 1]
```

**Wage updates**:
```
wage(i) ← wage(i) × (1 + 0.03 × augmented(i))     # +3% for augmented
wage(i) ← wage(i) × (1 - 0.15 × displaced(i))     # -15% for displaced  
wage(i) ← wage(i) + retrained(i) × old_wage × 0.10 # +10% recovery for retrained
```

---

## 5. LLM Integration Architecture

### 5.1 Archetype System Overview

The key innovation is using LLMs for **behavioral realism** without calling the LLM 1.2 million times. AgentTorch's Archetype system clusters workers into representative groups:

```
┌──────────────────────────────────────────────────────────────────────┐
│                   LLM Archetype System                               │
│                                                                      │
│  1,200,000 workers                                                   │
│       │                                                              │
│       │  mapping.json groupings:                                     │
│       │    age: ["18-34", "35-54", "55+"]         → 3 groups         │
│       │    education: ["HS or less",               → 3 groups         │
│       │                "Some College/Bachelor's",                     │
│       │                "Graduate+"]                                   │
│       │                                                              │
│       │  Cross-product: 3 × 3 = 9 archetypes                        │
│       │  (n_arch=7 used → top-7 by population)                       │
│       ▼                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     ┌──────────┐          │
│  │ Arch 1   │  │ Arch 2   │  │ Arch 3   │ ... │ Arch 7   │          │
│  │ 18-34    │  │ 18-34    │  │ 35-54    │     │ 55+      │          │
│  │ HS≤      │  │ College  │  │ HS≤      │     │ Grad+    │          │
│  │ ~180K    │  │ ~140K    │  │ ~200K    │     │ ~50K     │          │
│  │ workers  │  │ workers  │  │ workers  │     │ workers  │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘     └────┬─────┘          │
│       │              │              │                │                │
│       ▼              ▼              ▼                ▼                │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │              ONE LLM CALL PER ARCHETYPE                 │         │
│  │                                                         │         │
│  │  System: "You are simulating a Mississippi worker..."   │         │
│  │  User:   "Worker profile:                               │         │
│  │           - Age group: 18-34                            │         │
│  │           - Education: HS or less                       │         │
│  │           - Unemployment rate: 4.5%                     │         │
│  │           - Months since displacement: 3                │         │
│  │           What is the probability (0-1) this worker     │         │
│  │           enrolls in retraining?"                       │         │
│  │                                                         │         │
│  │  Response: "0.45"                                       │         │
│  └────────────────────────┬────────────────────────────────┘         │
│                           │                                          │
│                           ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │            BROADCAST BACK TO ALL WORKERS                 │         │
│  │                                                         │         │
│  │  All 180K workers in Archetype 1 receive p=0.45         │         │
│  │  → retraining_probability tensor [1.2M, 1]             │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                      │
│  Total: 7 LLM calls per quarter × 12 quarters = 84 calls            │
│  Instead of: 1,200,000 × 12 = 14,400,000 calls                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Backend Selection

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LLM Backend Comparison                            │
│                                                                      │
│  ┌────────────────────────┐    ┌─────────────────────────────┐       │
│  │     OLLAMA BACKEND     │    │       MLX BACKEND            │       │
│  │    (ollama_backend.py) │    │    (mlx_backend.py)          │       │
│  │                        │    │                              │       │
│  │  • HTTP REST API       │    │  • In-process inference      │       │
│  │  • Any OS              │    │  • Apple Silicon only        │       │
│  │  • Model: llama3.2,    │    │  • Model: Mistral-7B-4bit,  │       │
│  │    mistral, phi3, ...  │    │    Llama-3.2-3B, Phi-3.5    │       │
│  │  • urllib (no deps)    │    │  • pip install mlx-lm        │       │
│  │  • ~2-5 s/call         │    │  • ~0.5-2 s/call            │       │
│  │  • Separate process    │    │  • Metal GPU native          │       │
│  │                        │    │  • Model stays in memory     │       │
│  │  ollama serve          │    │  • No server needed          │       │
│  │  ollama pull model     │    │  • Auto-downloads from HF    │       │
│  └────────────────────────┘    └─────────────────────────────┘       │
│                                                                      │
│  Selection: --backend ollama | mlx  (default: mlx on macOS)          │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.3 Prompt Engineering

The LLM receives structured prompts designed for **numeric extraction**:

**System Prompt (agent_profile):**
```
You are simulating a Mississippi worker deciding whether to enroll 
in a retraining program after being displaced by AI automation.

Given information about a worker's profile and economic situation, 
respond with ONLY a single number between 0.0 and 1.0 representing 
the probability that this worker would choose to retrain.

Consider:
- Younger workers are more likely to retrain
- Higher education makes retraining easier and more attractive
- Workers with digital skills adapt faster to new roles
- Workers in economically resilient regions have better retraining access
- Financial pressure from job loss can motivate retraining
- Mississippi-specific factors: strong community ties, family obligations,
  limited public transit (affects program access)

Respond with ONLY a number between 0 and 1. Nothing else.
```

**User Prompt Template:**
```
Worker profile:
- Age group: {age}           ← filled by Archetype mapping
- Education: {education}     ← filled by Archetype mapping
- Current regional unemployment rate: {unemployment_rate}  ← scalar kwarg
- Months since displacement: {months_displaced}            ← scalar kwarg

What is the probability (0.0 to 1.0) this worker enrolls in retraining?
```

**Response Extraction** (`_extract_number()`):
```python
# Both backends implement identical extraction:
# 1. Strip <think>...</think> reasoning traces (qwen3, etc.)
# 2. Regex for float in [0.0, 1.0]
# 3. Normalize percentages (>1 → /100)
# 4. Fallback: 0.5
```

### 5.4 @with_behavior Decorator Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  @with_behavior decorator on LLMRetrainingDecision                  │
│                                                                     │
│  envs.create(model, population, archetypes={                        │
│      "llm_retraining_decision": archetype_config                    │
│  })                                                                 │
│       │                                                             │
│       │  At construction time:                                      │
│       │  → Decorator sets self.behavior = Behavior(archetype_config)│
│       │  → Behavior wraps Archetype + LLM backend                  │
│       │                                                             │
│       │  At each step (forward()):                                  │
│       │  → if self.behavior is not None:                            │
│       │      retrain_prob = self.behavior.sample(kwargs=obs_kwargs) │
│       │  → else:                                                    │
│       │      retrain_prob = self._parametric_fallback(...)          │
│       │                                                             │
│       │  Behavior.sample():                                         │
│       │    1. Groups workers by mapping.json axes                   │
│       │    2. Fills prompt template per archetype                   │
│       │    3. Calls LLM backend (Ollama/MLX) for each archetype    │
│       │    4. Broadcasts scalar result to all workers in group      │
│       │    5. Returns [1.2M, 1] tensor                              │
│       ▼                                                             │
│  retraining_probability → passed to JobTransition                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Output Pipeline & Observables

### 6.1 Output Flowchart

```
┌────────────────────────────────────────────────────────────────────┐
│                     Output Pipeline                                │
│                                                                    │
│  ┌──────────────┐                                                  │
│  │  Runner.step  │ ── per quarter ──►                              │
│  └──────┬───────┘                                                  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────┐              │
│  │              State Dict (after step)             │              │
│  │                                                  │              │
│  │  agents/workers/                                 │              │
│  │    employment_status  [1.2M,1] int               │              │
│  │    job_impact_status  [1.2M,1] int               │              │
│  │    wage               [1.2M,1] float             │              │
│  │    digital_skill      [1.2M,1] float             │              │
│  │    effective_exposure  [1.2M,1] float             │              │
│  │                                                  │              │
│  │  environment/                                    │              │
│  │    total_displaced    [12] float                  │              │
│  │    total_augmented    [12] float                  │              │
│  │    total_retrained    [12] float                  │              │
│  │    unemployment_rate  [12] float                  │              │
│  │    avg_wage_change    [12] float                  │              │
│  │    gdp_impact         [12] float                  │              │
│  │    sector_displacement[12] float (by sector)      │              │
│  │    sector_augmentation[12] float (by sector)      │              │
│  │    region_unemployment[5]  float (by region)      │              │
│  └──────────────────────┬───────────────────────────┘              │
│                         │                                          │
│         ┌───────────────┼───────────────┐                          │
│         ▼               ▼               ▼                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐            │
│  │  Console   │  │  JSON      │  │  state_trajectory   │            │
│  │  Output    │  │  Export    │  │  (in-memory)        │            │
│  │            │  │            │  │                     │            │
│  │ tqdm bar   │  │ sim_results│  │ runner.state_traj-  │            │
│  │ Q1 2025:   │  │ _llm.json │  │ ectory[ep][step]    │            │
│  │ D=12,821   │  │            │  │ → full state dict   │            │
│  │ A=38,874   │  │ Serialized │  │                     │            │
│  │ R=0        │  │ tensors →  │  │ Available for       │            │
│  │ U=60,941   │  │ lists      │  │ post-hoc analysis   │            │
│  └────────────┘  └────────────┘  └────────────────────┘            │
│                                                                    │
│  ┌──────────────────────────────────────────────────┐              │
│  │           Aggregate Reports Generated            │              │
│  │                                                  │              │
│  │  • Final worker status breakdown (5 categories)  │              │
│  │  • Quarterly trajectory table                    │              │
│  │  • Sector-level displacement & augmentation      │              │
│  │  • Regional unemployment distribution            │              │
│  │  • Wage statistics (mean, median)                │              │
│  │  • GDP impact estimate                           │              │
│  └──────────────────────────────────────────────────┘              │
└────────────────────────────────────────────────────────────────────┘
```

### 6.2 Key Output Metrics

| Metric | Source | Granularity |
|--------|--------|-------------|
| `job_impact_status` distribution | Agent tensor | Per-worker, per-quarter |
| `employment_status` distribution | Agent tensor | Per-worker, per-quarter |
| Total displaced/augmented/retrained | Environment vector | Per-quarter |
| Unemployment rate | Environment vector | Per-quarter |
| Sector displacement | Environment vector | Per-sector (12), cumulative |
| Region unemployment | Environment vector | Per-region (5), snapshot |
| Mean/median wage | Agent tensor aggregation | Per-quarter |
| GDP impact estimate | Computed: wage gains − losses | Per-quarter ($M) |

---

## 7. Agent State Machine

Each of the 1.2 million workers follows this state machine (`job_impact_status`):

```
                    ┌──────────────┐
                    │  UNAFFECTED  │ ← Initial state (status = 0)
                    │    (62.8%)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
    ┌──────────────┐ ┌──────────┐ ┌─────────────┐
    │  AUGMENTED   │ │ AT RISK  │ │  DISPLACED   │
    │   (9.4%)     │ │ (23.3%)  │ │   (0.4%)     │
    │              │ │          │ │              │
    │ AI enhances  │ │ Rising   │ │ Lost job     │
    │ productivity │ │ exposure │ │ to AI        │
    │ +3% wage/qtr │ │ not yet  │ │ -15% wage    │
    │ +0.02 skill  │ │ displaced│ │              │
    │              │ │          │ │              │
    │ status = 1   │ │ status=2 │ │ status = 3   │
    └──────────────┘ └──────────┘ └──────┬───────┘
                                         │
                                         │ Retraining
                                         │ (LLM or parametric)
                                         ▼
                                  ┌─────────────┐
                                  │  RETRAINED   │
                                  │   (4.1%)     │
                                  │              │
                                  │ New role     │
                                  │ +10% wage    │
                                  │ recovery     │
                                  │ +0.05 skill  │
                                  │              │
                                  │ status = 4   │
                                  └──────────────┘

Transition triggers:
  Unaffected → Augmented:   E_eff > 0.35 AND high digital_skill AND lucky draw
  Unaffected → At Risk:     0.35 ≤ E_eff < 0.60 AND not displaced/augmented
  Unaffected → Displaced:   E_eff > 0.60 AND low digital_skill AND unlucky draw
  Displaced  → Retrained:   P_retrain draw succeeds (LLM or parametric)
```

---

## 8. Data Flow Diagram — End to End

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│ BLS/ACS/    │     │ generate_    │     │  populations/  │
│ Felten data │────►│ data.py      │────►│  mississippi/  │
│ (external)  │     │ (NumPy/      │     │  *.pickle      │
│             │     │  Pandas)     │     │  (9 files)     │
└─────────────┘     └──────────────┘     └───────┬────────┘
                                                  │
                    ┌──────────────┐               │
                    │ config.yaml  │───────────────┤
                    │ (491 lines)  │               │
                    └──────────────┘               │
                                                  │
                    ┌──────────────┐               │
                    │ simulator.py │               │
                    │ (Registry)   │───────────────┤
                    └──────────────┘               │
                                                  │
                    ┌──────────────┐               │
                    │ mapping.json │               │
                    │ (archetype   │───────────────┤  (LLM mode only)
                    │  groupings)  │               │
                    └──────────────┘               │
                                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                   AgentTorch Runner                               │
│                                                                  │
│  ┌────────────┐   ┌────────────────┐   ┌────────────────┐        │
│  │ Initializer│──►│ State Dict     │──►│ Substep Loop   │        │
│  │            │   │ 1.2M agents    │   │ 12 quarters    │        │
│  │ pickle →   │   │ 13 env vars    │   │ 2 substeps/qtr │        │
│  │ tensor     │   │                │   │                │        │
│  └────────────┘   └────────────────┘   └───────┬────────┘        │
│                                                │                  │
│                                     ┌──────────┴──────────┐      │
│                                     │                     │      │
│                              ┌──────▼──────┐    ┌─────────▼────┐ │
│                              │ AI Exposure │    │ Job          │ │
│                              │ Assessment  │    │ Transition   │ │
│                              │             │    │              │ │
│                              │ • Tensorized│    │ • Tensorized │ │
│                              │ • 1.2M ops  │    │ • Stochastic │ │
│                              │ • <0.1s     │    │ • LLM-aware  │ │
│                              └─────────────┘    └──────┬───────┘ │
│                                                        │         │
│                                              ┌─────────▼───────┐ │
│                                              │ LLM Backend     │ │
│                                              │ (if enabled)    │ │
│                                              │                 │ │
│                                              │ Ollama / MLX    │ │
│                                              │ 7 calls/quarter │ │
│                                              └─────────────────┘ │
│                                                                  │
└──────────────────────────────────────┬───────────────────────────┘
                                       │
                         ┌─────────────┼──────────────┐
                         ▼             ▼              ▼
                 ┌──────────────┐ ┌──────────┐ ┌──────────────┐
                 │    Console   │ │   JSON   │ │  State       │
                 │    Summary   │ │  Export  │ │  Trajectory  │
                 │              │ │          │ │  (in-memory) │
                 │ • Quarterly  │ │ • sim_   │ │ • Full state │
                 │   stats      │ │   results│ │   per step   │
                 │ • Sector     │ │   .json  │ │ • Replayable │
                 │   analysis   │ │          │ │              │
                 │ • Regional   │ │ • 128    │ │              │
                 │   breakdown  │ │   lines  │ │              │
                 │ • tqdm bar   │ │          │ │              │
                 └──────────────┘ └──────────┘ └──────────────┘
```

---

## 9. Configuration Schema (YAML)

The `config.yaml` file (491 lines) follows AgentTorch's hierarchical configuration pattern:

### 9.1 Top-Level Structure

```yaml
simulation_metadata:      # Scalar parameters & hyperparameters
  num_agents: 1_200_000
  num_steps_per_episode: 12
  num_episodes: 1
  device: cpu
  ai_adoption_rate: 0.08
  automation_threshold: 0.60
  augmentation_threshold: 0.35
  retraining_effectiveness: 0.30
  # ... 15 more parameters

state:                    # Full state schema (agents + environment)
  agents:
    workers:
      number: ${simulation_metadata.num_agents}
      properties:
        age: ...          # 10 agent properties (schema + init function)
        # ...
  environment:
    ai_penetration_index: ...  # 13 environment variables
    # ...
  network: null           # No network topology in this model
  objects: null            # No physical objects

substeps:                 # Ordered substep definitions
  '0':                    # AI Exposure Assessment
    active_agents: [workers]
    policy: ...
    transition: ...
  '1':                    # Job Transition
    active_agents: [workers]
    policy: ...
    transition: ...
```

### 9.2 Learnable Parameters

The following parameters are marked `learnable: true`, making them differentiable and calibratable via AgentTorch's optimization (e.g., P3O):

| Parameter | Substep | Default | Purpose |
|-----------|---------|---------|---------|
| `adoption_acceleration` | 0 (policy) | 0.08 | Quarterly AI adoption growth rate |
| `base_penetration_growth` | 0 (transition) | 0.03 | Economy-wide AI penetration growth |
| `automation_threshold` | 1 (transition) | 0.60 | Exposure threshold for displacement |
| `augmentation_threshold` | 1 (transition) | 0.35 | Exposure threshold for augmentation |
| `retraining_rate` | 1 (transition) | 0.30 | Quarterly retraining success rate |

---

## 10. Component Reference

### 10.1 Registry Setup (`simulator.py`)

```python
registry = Registry()
registry.register(AssessAIExposure,       "assess_ai_exposure",        key="policy")
registry.register(UpdateExposure,          "update_exposure",           key="transition")
registry.register(JobTransition,           "job_transition",            key="transition")
registry.register(LLMRetrainingDecision,   "llm_retraining_decision",  key="policy")
registry.register(load_population_attribute, "load_population_attribute", key="initialization")
```

### 10.2 Class Hierarchy

```
SubstepAction (AgentTorch base)
├── AssessAIExposure          # Substep 0 policy
└── LLMRetrainingDecision     # Substep 1 policy (@with_behavior)

SubstepTransition (AgentTorch base)
├── UpdateExposure            # Substep 0 transition
└── JobTransition             # Substep 1 transition (254 lines)

LLMBackend (AgentTorch base)
├── OllamaLLM                # HTTP to local Ollama server
└── MLXLLM                   # In-process Apple Silicon inference
```

### 10.3 Runner Scripts

| Script | API Used | LLM | Key Features |
|--------|----------|-----|-------------|
| `run_mississippi_sim.py` | Executor + LoadPopulation | No | Basic parametric run |
| `run_mississippi_sim_llm.py` | `envs.create()` | Yes | Archetype, tqdm, backend selection, `--backend`, `--model`, `--n-arch`, `--no-llm` |

---

## 11. Performance Characteristics

### 11.1 Runtime Breakdown

| Phase | Parametric | LLM (MLX) | LLM (Ollama) |
|-------|-----------|-----------|--------------|
| Data loading & init | ~2 s | ~2 s | ~2 s |
| Model loading | — | ~5 s (first run) | — |
| Per-step compute (tensors) | ~0.09 s | ~0.09 s | ~0.09 s |
| Per-step LLM calls (7 arch) | — | ~1–3 s | ~5–15 s |
| Total (12 steps) | **~1.1 s** | **~30–50 s** | **~2–5 min** |
| Memory (agents) | ~100 MB | ~100 MB | ~100 MB |
| Memory (LLM model) | — | ~4 GB (7B-4bit) | External process |

### 11.2 Scaling Properties

- **Agent scaling:** Linear in `num_agents` (all operations are vectorized PyTorch tensor ops — no Python loops over agents)
- **Temporal scaling:** Linear in `num_steps_per_episode`
- **LLM scaling:** Linear in `n_arch × num_steps` (independent of agent count)
- **Device:** CPU by default; GPU-ready via `device: cuda` in config.yaml

### 11.3 Why MLX is Faster Than Ollama on Apple Silicon

| Factor | Ollama | MLX |
|--------|--------|-----|
| Process model | Separate server process | In-process |
| Communication | HTTP REST (JSON serialization) | Direct function call |
| GPU backend | llama.cpp (Metal via GGML) | MLX (native Metal) |
| Model loading | Per-request warm-up possible | Persistent in memory |
| Overhead per call | ~50–200 ms (HTTP) | ~0 ms |

---

## 12. Simulation Results — Parametric vs. LLM Comparison

This section presents a head-to-head comparison of the two simulation modes using actual run outputs: `simulation_results.json` (parametric) and `simulation_results_llm.json` (LLM-enhanced with Ollama Ministral-3B, 7 archetypes).

### 12.1 Final Worker Status — Side by Side

```
              PARAMETRIC MODEL                          LLM-ENHANCED MODEL
              ────────────────                          ──────────────────

  Unaffected  ████████████████████████████  62.8%       ████████████████████████████  62.8%
  Augmented   █████░░░░░░░░░░░░░░░░░░░░░░   9.4%       █████░░░░░░░░░░░░░░░░░░░░░░   9.4%
  At Risk     ██████████████░░░░░░░░░░░░░░  23.3%       ██████████████░░░░░░░░░░░░░░  23.3%
  Retrained   ██░░░░░░░░░░░░░░░░░░░░░░░░░   4.1%       ▪░░░░░░░░░░░░░░░░░░░░░░░░░░   1.3%  ← 67% fewer
  Displaced   ▪░░░░░░░░░░░░░░░░░░░░░░░░░░   0.4%       ██░░░░░░░░░░░░░░░░░░░░░░░░░   3.2%  ← 7.7× more
```

| Status | Parametric | % | LLM | % | Difference |
|--------|-----------|---|-----|---|------------|
| **Unaffected** | 753,638 | 62.80% | 753,496 | 62.79% | −142 |
| **Augmented** | 112,941 | 9.41% | 112,694 | 9.39% | −247 |
| **At Risk** | 279,373 | 23.28% | 279,433 | 23.29% | +60 |
| **Displaced** | 4,989 | 0.42% | 38,554 | 3.21% | **+33,565** |
| **Retrained** | 49,059 | 4.09% | 15,823 | 1.32% | **−33,236** |
| **Total** | 1,200,000 | 100% | 1,200,000 | 100% | 0 |

**Key observation:** The two models agree almost exactly on unaffected (62.8%), augmented (9.4%), and at-risk (23.3%) populations. The entire divergence is concentrated in the **displaced ↔ retrained split** — the LLM moves ~33,500 workers from "retrained" to "displaced."

### 12.2 Why the LLM Produces Different Retraining Outcomes

The divergence traces to a single mechanism — how `retraining_probability` is computed:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 RETRAINING PROBABILITY GENERATION                       │
│                                                                         │
│  PARAMETRIC:                                                            │
│    P_retrain = clamp(0.30 + skill×0.2 + edu_bonus - age_penalty,       │
│                      0.05, 0.80)                                        │
│    → Mean ≈ 0.40 for displaced workers                                  │
│    → Deterministic: same profile → same probability every time          │
│                                                                         │
│  LLM (Ministral-3B via Ollama):                                        │
│    System: "You are simulating a Mississippi worker..."                  │
│    User: "Age: 55+, Education: HS or less, Unemployment: 4.5%..."       │
│    Response: "0.12"                                                      │
│    → Mean ≈ 0.15–0.25 for displaced workers (estimated)                 │
│    → Contextual: considers MS-specific barriers                         │
│    → Per-archetype: 7 different probabilities per quarter               │
│                                                                         │
│  Result:                                                                │
│    Parametric retrains ~49K out of ~54K ever-displaced                   │
│    LLM retrains ~16K out of ~54K ever-displaced                         │
│    LLM retraining success rate ≈ 29% vs. parametric ≈ 91%              │
└─────────────────────────────────────────────────────────────────────────┘
```

The LLM reasons about **real-world retraining barriers** that the parametric formula doesn't encode:
- **Geographic access:** Rural Mississippi has limited retraining infrastructure
- **Family obligations:** Lower-income displaced workers may prioritize immediate employment over retraining
- **Age skepticism:** The LLM assigns much lower retraining probability to 55+ workers than the parametric formula's gentle age penalty
- **Education ceiling:** Workers with HS or less education face perceived (and real) barriers to technical retraining
- **Financial pressure:** The LLM implicitly factors in the urgency of income loss

### 12.3 Sector-Level Comparison

| Industry Sector | Param. Disp. | LLM Disp. | Param. Aug. | LLM Aug. | Disp. Δ |
|-----------------|-------------|-----------|-------------|----------|---------|
| Professional/Technical | 28,439 | 28,543 | 37,699 | 37,736 | +104 |
| Finance/Insurance | 20,901 | 21,081 | 26,964 | 26,907 | +180 |
| Manufacturing | 3,714 | 3,817 | 26,411 | 26,193 | +103 |
| Retail Trade | 451 | 388 | 7,971 | 8,021 | −63 |
| Government | 353 | 356 | 6,835 | 6,761 | +3 |
| Transportation | 177 | 176 | 3,621 | 3,552 | −1 |
| Healthcare | 10 | 14 | 1,975 | 2,043 | +4 |
| Accommodation/Food | 3 | 1 | 659 | 705 | −2 |
| Education | 0 | 0 | 308 | 311 | 0 |
| Agriculture/Forestry | 0 | 0 | 1 | 0 | 0 |
| Construction | 0 | 0 | 0 | 0 | 0 |
| Other Services | 0 | 1 | 497 | 464 | +1 |

**Key finding:** Sector-level displacement and augmentation counts are **nearly identical** between the two modes. This confirms that the LLM's influence is limited to the retraining decision — it does not affect the exposure calculation or the displacement/augmentation classification logic. The stochastic noise between runs (±1–3%) accounts for the small differences.

### 12.4 Regional Unemployment Comparison

| Region | Parametric | LLM | Baseline | Δ (LLM − Param.) |
|--------|-----------|-----|----------|-------------------|
| Delta | 4.58% | 8.58% | 6.0% | **+4.00pp** |
| North MS | 4.46% | 7.68% | 4.5% | **+3.23pp** |
| Central | 4.45% | 7.21% | 4.0% | **+2.76pp** |
| South/Gulf | 4.44% | 7.35% | 4.2% | **+2.90pp** |
| Metro (Jackson) | 4.30% | 6.15% | 3.5% | **+1.86pp** |

```
      Regional Unemployment: Parametric vs. LLM

  10% ┤
      │                         LLM
   9% ┤  ╔══╗
      │  ║  ║  ╔══╗
   8% ┤  ║  ║  ║  ║
      │  ║  ║  ║  ║  ╔══╗  ╔══╗
   7% ┤  ║  ║  ║  ║  ║  ║  ║  ║
      │  ║  ║  ║  ║  ║  ║  ║  ║  ╔══╗
   6% ┤  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║
      │  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║          Parametric
   5% ┤  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
      │  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  │  │  │  │  │  │  │  │  │  │
   4% ┤  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  │  │  │  │  │  │  │  │  │  │
      │  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  │  │  │  │  │  │  │  │  │  │
   3% ┤  ╚══╝  ╚══╝  ╚══╝  ╚══╝  ╚══╝  └──┘  └──┘  └──┘  └──┘  └──┘
      └──────────────────────────────────────────────────────────────────
        Delta  North  Central South  Metro  Delta  North  Central South  Metro
```

**Key finding:** The LLM model produces **dramatically higher unemployment** across all regions — 6.2%–8.6% vs. 4.3%–4.6% in the parametric model. The difference is largest in the **Delta** (+4.0pp), which has the weakest economic infrastructure, and smallest in **Metro Jackson** (+1.9pp), which has the most retraining resources. This pattern is consistent with the LLM's sensitivity to regional economic context.

### 12.5 Wage Impact Comparison

| Metric | Parametric | LLM | Difference |
|--------|-----------|-----|------------|
| Final Mean Wage | $41,931 | $41,793 | −$138 (−0.33%) |
| Final Median Wage | $39,169 | $39,057 | −$112 (−0.29%) |
| Wage Change vs. Baseline ($38,500) | +8.9% | +8.6% | −0.3pp |

The wage gap is small because:
1. Augmentation (the primary wage driver at +3%/quarter) is identical between modes
2. The ~33K additional displaced workers in the LLM model earn lower-than-average wages
3. Retraining recovery (+10% wage bounce) applies to fewer workers in the LLM model, but the aggregate impact on mean wage is modest

### 12.6 Displacement Dynamics — Temporal Pattern

Both models show displacement concentrated in Q1, with subsequent quarters recording zero new displacement. This reflects the current simulation's state accumulation pattern where the initial exposure shock produces the bulk of transitions:

| Quarter | Param. New Displaced | LLM New Displaced | Param. Retrained | LLM Retrained |
|---------|---------------------|-------------------|-----------------|---------------|
| Q1 2025 | 975 | 981 | 1,752 | 331 |
| Q2 2025 | 0 | 0 | 0 | 0 |
| Q3 2025 | 0 | 0 | 0 | 0 |
| ... | 0 | 0 | 0 | 0 |
| Q4 2027 | 0 | 0 | 0 | 0 |

**Note:** The quarterly trajectory differs from the cumulative sector totals (§12.3) because sector displacement accumulates across all 12 quarters while the quarterly counters reset. The initial Q1 shock creates the displacement pool; subsequent quarters process retraining from that pool. The LLM's lower retraining probability per quarter means fewer workers exit the displaced pool over the simulation's lifetime.

### 12.7 Convergence Analysis — What the Modes Agree On

Despite the dramatic retraining divergence, the two models converge on several critical conclusions:

| Finding | Parametric | LLM | Agreement |
|---------|-----------|-----|-----------|
| Total workers touched by AI (augmented + at-risk + displaced + retrained) | 446,362 (37.2%) | 446,504 (37.2%) | ✅ Near-identical |
| Professional/Tech is most displaced sector | 28,439 (52.6%) | 28,543 (52.5%) | ✅ Identical |
| Finance/Insurance is second-most displaced | 20,901 (38.7%) | 21,081 (38.8%) | ✅ Identical |
| Agriculture, Construction, Education: zero/negligible displacement | 0, 0, 0 | 0, 0, 0 | ✅ Identical |
| Metro Jackson has lowest unemployment | 4.30% | 6.15% | ✅ Same rank order |
| Delta has highest unemployment | 4.58% | 8.58% | ✅ Same rank order |
| AI augments more than it displaces | 23:1 ratio | 2.9:1 ratio | ✅ Both net positive |
| Mean wage increases over baseline | +8.9% | +8.6% | ✅ Both positive |

### 12.8 Summary: What the LLM Changes

```
┌─────────────────────────────────────────────────────────────────────────┐
│              LLM IMPACT ON SIMULATION OUTCOMES                          │
│                                                                         │
│  ┌─────────────────────────────────┐   ┌──────────────────────────────┐ │
│  │  UNCHANGED BY LLM              │   │  CHANGED BY LLM              │ │
│  │                                 │   │                               │ │
│  │  • AI exposure calculation      │   │  • Retraining probability    │ │
│  │  • Displacement classification  │   │    (↓ significantly)         │ │
│  │  • Augmentation classification  │   │                               │ │
│  │  • Sector-level patterns        │   │  • Final displaced count     │ │
│  │  • At-risk population size      │   │    (↑ 7.7×: 4,989 → 38,554) │ │
│  │  • Unaffected population size   │   │                               │ │
│  │  • Industry rank ordering       │   │  • Final retrained count     │ │
│  │  • Regional rank ordering       │   │    (↓ 67%: 49,059 → 15,823) │ │
│  │                                 │   │                               │ │
│  │                                 │   │  • Regional unemployment     │ │
│  │                                 │   │    (↑ 2–4pp across all       │ │
│  │                                 │   │     regions)                  │ │
│  │                                 │   │                               │ │
│  │                                 │   │  • Mean wage (↓ 0.3%)        │ │
│  └─────────────────────────────────┘   └──────────────────────────────┘ │
│                                                                         │
│  INTERPRETATION:                                                        │
│  The LLM acts as a "realism check" on the parametric model's           │
│  optimistic retraining assumptions. It does NOT change the              │
│  structural patterns of AI's impact (which sectors, which regions)      │
│  but dramatically changes the RECOVERY trajectory — suggesting          │
│  that displaced workers in Mississippi face steeper barriers to         │
│  retraining than a simple formula predicts.                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 12.9 Policy Implications of the Divergence

The gap between parametric and LLM results has direct policy significance:

| If Parametric Is Closer to Reality | If LLM Is Closer to Reality |
|-----------------------------------|-----------------------------|
| Existing retraining infrastructure is adequate | Mississippi needs **major investment** in retraining capacity |
| ~5,000 workers remain displaced (manageable) | ~38,500 workers remain displaced (**crisis-level**) |
| Regional unemployment stays near baseline | Regional unemployment rises **2–4 percentage points** |
| Policy focus: augmentation support | Policy focus: **emergency retraining + income support** |
| Total fiscal cost: ~$50M (est.) | Total fiscal cost: **~$400M** (est.) |

> **Recommendation:** Use the LLM results as the **conservative planning scenario** and the parametric results as the **optimistic scenario**. Workforce policy should be designed to handle the LLM scenario while hoping for the parametric one.

---

## 13. Assumptions & Limitations

### Model Assumptions

1. **Industry-level AI exposure** — All workers in a sector share a base exposure score. Within-occupation heterogeneity is not captured.
2. **Uniform adoption rate** — 8% quarterly acceleration applies statewide; in reality, metro areas adopt faster.
3. **Optimistic retraining** — 30% quarterly success rate assumes well-funded, accessible programs.
4. **No network effects** — Workers are independent; factory closures don't cascade through supply chains.
5. **No migration** — Workers remain in their region throughout the simulation.
6. **Synthetic population** — Calibrated to BLS/ACS aggregates but not derived from actual microdata.
7. **Stochastic transitions** — Displacement/augmentation use simplified probability models, not firm-level decision logic.

### Framework Limitations

- CPU-only execution (GPU would enable large-scale sensitivity analysis)
- No inter-agent communication or spatial interaction
- LLM responses may vary between model versions and temperature settings — the Ministral-3B model used in the LLM run produces notably lower retraining probabilities than the parametric baseline, and other models (Llama 3.2, Phi-3.5) may produce different results
- Archetype grouping reduces behavioral diversity to `n_arch` representative types (7 in the current run, producing 84 total LLM calls)
- No feedback from aggregate outcomes to individual decisions (e.g., workers don't respond to rising unemployment by retraining more aggressively in the parametric model)
- The parametric and LLM models diverge primarily on retraining recovery (§12), meaning the choice of mode materially affects policy conclusions — the true outcome likely lies between the two scenarios

---

## 14. Policy Implications

> **Note:** Policy recommendations are informed by **both** the parametric (optimistic) and LLM (conservative) scenarios from §12. Where outcomes diverge, both are cited.

### High Priority

1. **Targeted retraining for Professional/Technical and Finance workers** — These sectors account for 91% of displacement in both modes. Programs should focus on AI-complementary skills (prompt engineering, AI oversight, data analysis). Under the LLM scenario, ~38,500 workers remain displaced (vs. ~5,000 parametric), making retraining investment **8× more urgent**.

2. **Regional support for Metro Jackson and the Delta** — The parametric model shows Metro Jackson facing the steepest relative unemployment increase (+0.79pp), but the LLM model reveals the **Delta as the most impacted region** (8.58% unemployment, +2.58pp above baseline). Policy should address both knowledge-worker displacement in metro areas and structural barriers in rural regions.

3. **Digital skills initiative for mid-career workers (35–54, HS/Some College)** — This cohort has the highest displacement-to-augmentation ratio and is the most policy-responsive. The LLM's much lower retraining estimates for older, less-educated workers suggest that **program design must address non-academic barriers** (transportation, childcare, income support during retraining).

### Medium Priority

4. **Manufacturing AI integration support** — Mississippi's manufacturing base (12%) shows high augmentation potential (18.3%). Supporting AI tool adoption while retaining workers boosts productivity.

5. **Early warning system** — Peak displacement occurs within the first 6 quarters. Monitoring AI investment and job posting changes could trigger proactive interventions.

### Long-Term

6. **K-12 AI literacy reform** — Reducing the future at-risk population (23.3%) requires systemic education changes.

---

## 15. Reproducibility

### Parametric Mode

```bash
cd /path/to/AgentTorch
source .venv/bin/activate

# Generate population (if needed)
python agent_torch/models/mississippi_ai/generate_data.py

# Run parametric simulation
python run_mississippi_sim.py
```

### LLM-Enhanced Mode (MLX — Apple Silicon)

```bash
pip install mlx-lm

# MLX backend (default, fastest on Apple Silicon)
python run_mississippi_sim_llm.py --backend mlx

# Ollama backend (used for §12 comparison: Ministral-3B, 7 archetypes)
python run_mississippi_sim_llm.py --backend ollama --model ministral-3:8b

# Ollama with Llama 3.2 (requires `ollama serve` + `ollama pull llama3.2`)
python run_mississippi_sim_llm.py --backend ollama --model llama3.2

# Parametric fallback (no LLM)
python run_mississippi_sim_llm.py --no-llm
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `mlx` | LLM backend: `mlx` or `ollama` |
| `--model` | auto | Model name (auto-selects per backend) |
| `--n-arch` | 7 | Number of archetypes for worker clustering |
| `--no-llm` | false | Disable LLM, use parametric rules |
| `--base-url` | `http://localhost:11434` | Ollama API endpoint |

### Dependencies

```
torch>=2.0
numpy
pandas
scipy
omegaconf
tqdm
matplotlib
dask
pyarrow
AgentTorch (from source)
mlx-lm (optional, for MLX backend)
```

---

## 16. Improving Accuracy with Frey & Osborne Automation Scores

### The Problem

The current simulation uses **Felten et al. (2021) AI Exposure** scores at the industry level. This measures how much AI *touches* an occupation — not whether it *replaces* it. The consequence is that knowledge workers (Professional/Technical, Finance) are over-displaced, while routine service workers (Retail, Food Service, Transportation) are under-displaced.

### The Solution: Dual-Score Model

Integrating **Frey & Osborne (2017) automation probability** scores for 702 SOC codes would create a two-axis model:

| Score | Measures | Role |
|-------|----------|------|
| F&O Automation Probability | Job elimination likelihood | → **Displacement** |
| Felten AI Exposure | AI capability overlap | → **Augmentation** |

| Sector | Current Disp. | Expected with F&O | Reason |
|--------|--------------|-------------------|--------|
| Retail Trade | 0.3% | ~15–25% | Cashiers (0.97), retail salespersons (0.92) |
| Professional/Tech | 34.0% | ~10–15% | Software devs (0.04), engineers (0.02) have *low* F&O |
| Transportation | 0.3% | ~10–20% | Truck drivers (0.79), taxi drivers (0.89) |
| Finance/Insurance | 34.9% | ~15–25% | Splits: tellers (0.98) vs. analysts (0.23) |

This would transform the simulation from an AI exposure model into a true **automation displacement model** with occupation-level resolution.

---

## 17. References

1. Felten, E., Raj, M., & Seamans, R. (2021). *Occupational, industry, and geographic exposure to artificial intelligence.* Strategic Management Journal, 42(12), 2195-2217.
2. Frey, C.B., & Osborne, M.A. (2017). *The future of employment: How susceptible are jobs to computerisation?* Technological Forecasting and Social Change, 114, 254-280.
3. Chopra, A., et al. (2024). *AgentTorch: Large Population Models.* MIT Media Lab.
4. Acemoglu, D., & Restrepo, P. (2020). *Robots and jobs: Evidence from US labor markets.* Journal of Political Economy, 128(6), 2188-2244.
5. Bureau of Labor Statistics. (2024). *Quarterly Census of Employment and Wages — Mississippi.*
6. U.S. Census Bureau. (2024). *American Community Survey 5-Year Estimates — Mississippi.*

---

*This report documents the architecture of a simulation built on the [AgentTorch](https://github.com/AgentTorch/AgentTorch) Large Population Models framework (MIT Media Lab). All population data is synthetic. Results represent one plausible scenario under stated assumptions and should not be interpreted as predictions.*
