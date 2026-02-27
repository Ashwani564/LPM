# Dataset Requirements for Mississippi AI Workforce Simulation

## Exact Data Specifications for Improving AgentTorch LPM Accuracy

**Related Reports:** `report_lpm.md` (Architecture), `report.md` (Simulation Results)  
**Date:** February 2026  
**Purpose:** Define the exact datasets — format, schema, source, and integration path — needed to transform the current synthetic simulation into a calibrated, policy-grade model.

---

## Table of Contents

1. [Current Data Gap Analysis](#1-current-data-gap-analysis)
2. [Dataset #1: Occupation-Level Microdata (PUMS)](#2-dataset-1-occupation-level-microdata-pums)
3. [Dataset #2: Frey & Osborne Automation Probabilities](#3-dataset-2-frey--osborne-automation-probabilities)
4. [Dataset #3: Felten AI Exposure Index (Occupation-Level)](#4-dataset-3-felten-ai-exposure-index-occupation-level)
5. [Dataset #4: BLS Occupational Employment & Wage Statistics (OES)](#5-dataset-4-bls-occupational-employment--wage-statistics-oes)
6. [Dataset #5: Quarterly Census of Employment & Wages (QCEW)](#6-dataset-5-quarterly-census-of-employment--wages-qcew)
7. [Dataset #6: O*NET Task-Level Data](#7-dataset-6-onet-task-level-data)
8. [Dataset #7: Current Population Survey (CPS) Microdata](#8-dataset-7-current-population-survey-cps-microdata)
9. [Dataset #8: Mississippi Retraining Program Records](#9-dataset-8-mississippi-retraining-program-records)
10. [Dataset #9: LODES/LEHD Commute & Employment Data](#10-dataset-9-lodeslehd-commute--employment-data)
11. [Dataset #10: Digital Skill Proxies](#11-dataset-10-digital-skill-proxies)
12. [Integration Architecture — How Each Dataset Feeds AgentTorch](#12-integration-architecture--how-each-dataset-feeds-agenttorch)
13. [Data Pipeline Flowchart — Current vs. Improved](#13-data-pipeline-flowchart--current-vs-improved)
14. [Impact Assessment — Expected Accuracy Improvements](#14-impact-assessment--expected-accuracy-improvements)
15. [Dataset Priority Matrix](#15-dataset-priority-matrix)
16. [Implementation Roadmap](#16-implementation-roadmap)
17. [References](#17-references)

---

## 1. Current Data Gap Analysis

### What the simulation currently uses (all synthetic)

The `generate_data.py` script creates 1.2 million agents using `np.random.choice()` with hardcoded probability distributions. **No real microdata is used.** The following table maps every current synthetic input to the real dataset that should replace it:

| Agent Attribute | Current Source | Current Method | Critical Problem | Required Dataset |
|----------------|----------------|----------------|------------------|-----------------|
| `age` | Hardcoded `P=[.12,.22,.21,.20,.18,.07]` | `np.random.choice(6)` | No joint distributions (age×industry ignored) | **ACS PUMS** |
| `gender` | Hardcoded `P=[.48,.52]` | `np.random.choice(2)` | Independent of occupation | **ACS PUMS** |
| `education` | Hardcoded `P=[.12,.30,.28,.19,.11]` | `np.random.choice(5)` | Not conditioned on age or industry | **ACS PUMS** |
| `industry` | Hardcoded `P=[.04,.12,...,.08]` | `np.random.choice(12)` | 12 NAICS-2 sectors only, no occupation within | **BLS OES + PUMS** |
| `ai_exposure` | 12 hardcoded floats | Industry baseline + N(0,0.08) | Industry-level only; no occupation resolution | **Felten et al. + F&O** |
| `wage` | 12 hardcoded medians | Median × LogNormal(0,0.25) | No occupation-level wage variation | **BLS OES** |
| `digital_skill` | Formula: 0.4×edu + 0.4×age_factor + 0.2×Beta(2,3) | Derived | No empirical grounding; invented coefficients | **O\*NET + CPS** |
| `employment_status` | Hardcoded `P=[.88,.04,.08]` | `np.random.choice(3)` | Not stratified by industry, region, or demographics | **CPS / LAUS** |
| `region` | Hardcoded `P=[.12,.18,.25,.20,.25]` | `np.random.choice(5)` | Independent of industry or occupation | **LODES/LEHD** |

| Config Parameter | Current Source | Critical Problem | Required Dataset |
|-----------------|----------------|------------------|-----------------|
| `sector_adoption_speed` | 12 hardcoded values | Invented; no empirical basis | **McKinsey/Brookings AI adoption surveys** |
| `region_resilience` | 5 hardcoded values `[.55,.65,.70,.68,.80]` | Invented | **BEA/BLS regional economic indicators** |
| `retraining_effectiveness` | Single value `0.30` | Not stratified by program type | **MS MDES / WIOA records** |
| `automation_threshold` | Single value `0.60` | Not calibrated against observed displacement | **Calibration against CPS labor flows** |
| `wage_growth_augmented` | Single value `0.03` | Not empirically grounded | **BLS Employer Costs / CPS earnings** |
| `wage_decline_displaced` | Single value `0.15` | Not empirically grounded | **Displaced Workers Survey (CPS)** |

### Core Problem Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ACCURACY GAP HIERARCHY                           │
│                                                                     │
│  CRITICAL (distorts results fundamentally):                         │
│    ❌ No occupation-level resolution (12 industries vs 702 SOCs)    │
│    ❌ Exposure ≠ Automation risk (Felten only, no F&O scores)       │
│    ❌ All attributes independently sampled (no joint distributions) │
│                                                                     │
│  HIGH (materially affects policy conclusions):                      │
│    ⚠ No real wage distributions (industry medians only)            │
│    ⚠ Digital skill is entirely fabricated                          │
│    ⚠ Retraining rate not empirically calibrated                    │
│    ⚠ Region assignments independent of industry/occupation         │
│                                                                     │
│  MODERATE (affects precision but not direction):                    │
│    △ Employment status not stratified                               │
│    △ Sector adoption speed not empirically sourced                  │
│    △ Region resilience not data-driven                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset #1: Occupation-Level Microdata (PUMS)

### What It Is

The **American Community Survey Public Use Microdata Sample (ACS PUMS)** provides individual-level records for ~1% of the US population, including demographic, employment, occupation, industry, income, and education data. The Mississippi subset contains approximately **28,000 person-records** that can be reweighted to represent the full 1.2M labor force.

### Exact Dataset Specification

| Field | Value |
|-------|-------|
| **Name** | ACS 1-Year or 5-Year PUMS (Mississippi) |
| **Source** | US Census Bureau, https://data.census.gov/mdat/ |
| **Direct Download** | https://www2.census.gov/programs-surveys/acs/data/pums/ |
| **Year** | 2023 or 2024 5-Year (latest available) |
| **File** | `csv_pms.zip` (Mississippi person-level file) |
| **Format** | CSV, ~300 MB compressed |
| **Access** | Public, free, no application needed |
| **License** | Public domain (US Government work) |

### Exact Columns Required

| PUMS Column | Description | Maps To Agent Attribute | dtype |
|------------|-------------|------------------------|-------|
| `AGEP` | Age (integer 0–99) | `age` (bucketed into 6 groups) | int |
| `SEX` | Sex (1=Male, 2=Female) | `gender` | int |
| `SCHL` | Educational attainment (24 codes) | `education` (mapped to 5 levels) | int |
| `NAICSP` | NAICS industry code (4–6 digit) | `industry` (mapped to 12 sectors) | str → int |
| `SOCP` | SOC occupation code (6-digit) | **NEW: `occupation`** | str → int |
| `WAGP` | Wages/salary income (annual $) | `wage` | float |
| `ESR` | Employment status recode | `employment_status` | int |
| `PUMA` | Public Use Microdata Area | `region` (mapped to 5 MS regions) | int |
| `PWGTP` | Person weight | Used for reweighting to 1.2M | float |
| `COW` | Class of worker | Used for filtering (exclude military, etc.) | int |

### Why It Is Required

1. **Joint distributions:** Currently, age, education, industry, and wage are sampled independently. In reality, a 22-year-old in Agriculture almost never has a Graduate degree. PUMS preserves these **correlations** — each record is a real person with internally consistent attributes.

2. **Occupation codes:** The single most important missing dimension. PUMS includes the 6-digit SOC code (`SOCP`), which enables linking to Frey & Osborne automation scores and Felten AI exposure at the **occupation level** instead of the industry level.

3. **Real wage distributions:** Instead of `industry_median × LogNormal`, we get actual wage distributions by occupation × education × age, including the long tail of high earners and the mass of minimum-wage workers.

4. **Geographic precision:** PUMA codes map directly to Mississippi's sub-state regions with much higher accuracy than our current 5-region random assignment.

### How It Will Be Used in AgentTorch

```python
# REPLACEMENT for generate_data.py
# Instead of np.random.choice(), draw from real PUMS records

import pandas as pd

pums = pd.read_csv("psam_p28.csv", usecols=[
    "AGEP", "SEX", "SCHL", "NAICSP", "SOCP", "WAGP", "ESR", "PUMA", "PWGTP"
])

# Filter to working-age, civilian employed/unemployed
pums = pums[(pums.AGEP >= 18) & (pums.AGEP <= 67) & (pums.COW.isin([1,2,3,4,5,6,7]))]

# Resample with replacement using PWGTP weights to get 1.2M records
agents = pums.sample(n=1_200_000, weights="PWGTP", replace=True, random_state=42)

# Map SOCP → 6-digit SOC → Frey & Osborne automation score
# Map NAICSP → 12-sector industry code
# Map PUMA → 5-region code
# Map SCHL → 5-level education
# Map AGEP → 6 age groups

# Save as pickle files (same format as current generate_data.py)
for attr in ["age", "gender", "education", "industry", "occupation",
             "wage", "employment_status", "region"]:
    pd.Series(agents[attr].values.astype(np.float32)).to_pickle(f"{OUT_DIR}/{attr}.pickle")
```

### Expected Impact on Accuracy

| Metric | Before (Synthetic) | After (PUMS) | Change |
|--------|-------------------|--------------|--------|
| Attribute correlations | Zero (independent draws) | Real joint distribution | **Fundamental** |
| Occupation resolution | 12 industries | ~200 occupations (MS labor force) | **17× finer** |
| Wage accuracy | ±$10K RMSE (estimated) | ~±$2K RMSE | **5× improvement** |
| Regional assignment | Random | PUMA-based | **Geographically correct** |

---

## 3. Dataset #2: Frey & Osborne Automation Probabilities

### What It Is

The seminal **Frey & Osborne (2017)** study provides a **probability of computerization** for **702 6-digit SOC occupations**. Scores range from 0.0 (safe from automation) to 1.0 (near-certain automation), based on expert assessment of bottlenecks in perception, manipulation, creative intelligence, and social intelligence.

### Exact Dataset Specification

| Field | Value |
|-------|-------|
| **Name** | Frey & Osborne Automation Probability Scores |
| **Source** | Appendix Table A.1 of the published paper |
| **Paper** | Frey, C.B. & Osborne, M.A. (2017). "The future of employment." *Technological Forecasting and Social Change*, 114, 254–280 |
| **Direct Access** | Table A.1 in the paper, or pre-extracted CSV from https://data.world/quanticdata/frey-and-osborne |
| **Format** | CSV: `soc_code, occupation_title, probability` |
| **Records** | 702 occupations |
| **Access** | Open access (published academic paper) |

### Exact Schema

```csv
soc_code,occupation_title,probability
11-1011,"Chief Executives",0.015
13-2051,"Financial Analysts",0.23
43-3071,"Tellers",0.98
41-2011,"Cashiers",0.97
53-3032,"Heavy and Tractor-Trailer Truck Drivers",0.79
15-1252,"Software Developers",0.042
29-1141,"Registered Nurses",0.009
...
```

| Column | Type | Description |
|--------|------|-------------|
| `soc_code` | str | 6-digit SOC-2010 occupation code (e.g., `43-3071`) |
| `occupation_title` | str | Official BLS occupation title |
| `probability` | float | Probability of computerization [0.0, 1.0] |

### Why It Is Required

This is the **single most consequential dataset missing** from the simulation. Our current model uses the Felten AI Exposure Index, which measures how much AI *touches* a job — not whether it *eliminates* the job. The distinction is critical:

| Occupation | Felten AI Exposure | F&O Automation Prob. | Current Model Result | Correct Result |
|------------|-------------------|---------------------|---------------------|----------------|
| Financial Analyst | 0.78 (high) | 0.23 (low) | **Displaced** ❌ | **Augmented** ✅ |
| Bank Teller | 0.65 (high) | 0.98 (very high) | Displaced ✅ | Displaced ✅ |
| Cashier | 0.30 (low) | 0.97 (very high) | **Unaffected** ❌ | **Displaced** ✅ |
| Software Developer | 0.72 (high) | 0.04 (very low) | **Displaced** ❌ | **Augmented** ✅ |
| Truck Driver | 0.25 (low) | 0.79 (high) | **Unaffected** ❌ | **At Risk/Displaced** ✅ |
| Registered Nurse | 0.45 (moderate) | 0.009 (negligible) | At Risk | **Unaffected** ✅ |

### How It Will Be Used in AgentTorch

```python
# In generate_data.py — after assigning SOC codes from PUMS:

fo_scores = pd.read_csv("frey_osborne_2017.csv")
# Columns: soc_code, probability

# Merge with agent data via SOC code
agents["automation_probability"] = agents["soc_code"].map(
    fo_scores.set_index("soc_code")["probability"]
).fillna(0.5)  # Default for unmatched codes

# Save as new agent attribute
pd.Series(agents["automation_probability"].values.astype(np.float32)).to_pickle(
    f"{OUT_DIR}/automation_probability.pickle"
)
```

```yaml
# In config.yaml — add new agent property:
state:
  agents:
    workers:
      properties:
        automation_probability:
          dtype: float
          name: "Frey & Osborne Automation Probability"
          shape: [1200000, 1]
          initialization_function:
            generator: load_population_attribute
            arguments:
              file_path: .../automation_probability.pickle
              attribute: automation_probability
```

```python
# In job_transition.py — DUAL-SCORE displacement model:

# CURRENT (single-score):
# displacement_prob = f(effective_ai_exposure > 0.60, low_skill, low_resilience)

# IMPROVED (dual-score):
# automation_risk drives DISPLACEMENT
# ai_exposure drives AUGMENTATION
automation_risk = get_by_path(state, "agents/workers/automation_probability")

displacement_prob = torch.clamp(
    (automation_risk - 0.50) * 2.0 * (1.0 - digital_skill) * (1.0 - worker_resilience),
    0.0, 0.85
) * eligible

augmentation_prob = torch.clamp(
    (effective_exposure - 0.35) * 1.5 * digital_skill * worker_resilience
    * (1.0 - automation_risk),  # LOW automation risk + HIGH exposure = augmentation
    0.0, 0.70
) * eligible
```

### Expected Impact on Results

| Sector | Current Displacement | With F&O | Explanation |
|--------|---------------------|----------|-------------|
| Professional/Technical | 34.0% (28,529) | **~10–15%** | Most professionals (devs, engineers, analysts) have F&O < 0.10 |
| Finance/Insurance | 34.9% (20,910) | **~15–25%** | Splits: tellers (0.98) vs. analysts (0.23) |
| Retail Trade | 0.3% (400) | **~15–25%** | Cashiers (0.97), salespersons (0.92) |
| Transportation | 0.3% (200) | **~10–20%** | Truck drivers (0.79), taxi drivers (0.89) |
| Accommodation/Food | <0.1% (5) | **~5–15%** | Food prep (0.87), waitstaff (0.94) |
| Healthcare | <0.1% (10) | **~1–3%** | Mostly safe (nurses 0.009), but medical records clerks (0.98) |

---

## 4. Dataset #3: Felten AI Exposure Index (Occupation-Level)

### What It Is

The **Felten, Raj & Seamans (2021) AI Occupational Exposure Index** measures how much AI application capabilities overlap with occupational task requirements. Our simulation already uses this — but at the **industry level** (12 hardcoded floats). The original dataset provides scores for **774 SOC occupations**.

### Exact Dataset Specification

| Field | Value |
|-------|-------|
| **Name** | AI Occupational Exposure (AIOE) Index |
| **Source** | Felten, Raj & Seamans (2021) — supplementary data |
| **Direct Download** | https://github.com/mfelten/aioe (or Zenodo) |
| **Format** | CSV: `soc_code, soc_title, aioe_score` |
| **Records** | 774 occupations |
| **Access** | Open access (academic publication supplementary material) |

### Exact Schema

```csv
soc_code,soc_title,aioe_score
11-1011,"Chief Executives",0.523
13-2051,"Financial Analysts",0.784
43-3071,"Tellers",0.651
15-1252,"Software Developers",0.718
29-1141,"Registered Nurses",0.452
...
```

| Column | Type | Description |
|--------|------|-------------|
| `soc_code` | str | 6-digit SOC occupation code |
| `soc_title` | str | Occupation title |
| `aioe_score` | float | AI Occupational Exposure Index [0.0, 1.0] |

### Why It Is Required

Currently, `generate_data.py` uses 12 hardcoded industry-level exposure values:

```python
# CURRENT — 12 values for 12 sectors
AI_EXPOSURE_BY_INDUSTRY = np.array([0.15, 0.42, 0.38, 0.35, 0.30, 0.18, 0.32, 0.25, 0.65, 0.70, 0.45, 0.28])
```

This means a **software developer** (AIOE = 0.718) and a **janitor** (AIOE = 0.12) in the same "Professional/Technical" industry both get the same exposure score of 0.65. The occupation-level dataset would give each their correct score.

### How It Will Be Used

```python
# In generate_data.py — replace industry-level scores with occupation-level:

felten = pd.read_csv("felten_aioe_2021.csv")
agents["ai_exposure"] = agents["soc_code"].map(
    felten.set_index("soc_code")["aioe_score"]
).fillna(0.35)  # Sector median for unmatched

# Result: each of 1.2M agents has their own occupation-specific AI exposure
# Instead of: all workers in an industry sharing the same value
```

### Expected Impact

- **Within-sector variance** becomes visible (currently zero)
- **Augmentation predictions** improve: high-exposure, low-automation-risk workers correctly classified as augmented
- **At-risk population** size and composition changes significantly

---

## 5. Dataset #4: BLS Occupational Employment & Wage Statistics (OES)

### What It Is

The **BLS OES** survey provides employment counts and wage distributions for ~800 occupations at the state level. This is the authoritative source for "how many cashiers work in Mississippi, and what do they earn?"

### Exact Dataset Specification

| Field | Value |
|-------|-------|
| **Name** | Occupational Employment and Wage Statistics (OES), State Level |
| **Source** | Bureau of Labor Statistics |
| **Direct Download** | https://www.bls.gov/oes/tables.htm → "State" → "Mississippi" |
| **File** | `oesm28.xlsx` or `state_M2024_dl.xlsx` |
| **Format** | Excel/CSV |
| **Year** | May 2024 (latest) |
| **Records** | ~600 occupations × wage percentiles |
| **Access** | Public, free |

### Exact Columns Required

| Column | Type | Description | Maps To |
|--------|------|-------------|---------|
| `OCC_CODE` | str | 6-digit SOC code | Agent `occupation` |
| `OCC_TITLE` | str | Occupation name | Reference |
| `TOT_EMP` | int | Total employment in MS | Sampling weights for `generate_data.py` |
| `H_MEAN` / `A_MEAN` | float | Mean hourly/annual wage | Agent `wage` distribution |
| `H_PCT10`–`H_PCT90` | float | Wage percentiles (10th–90th) | Wage distribution shape |
| `NAICS` | str | Industry code (for occupation×industry cross-tab) | Agent `industry` |

### Why It Is Required

1. **Occupation sampling weights:** Tells us exactly how many people in Mississippi work in each occupation, enabling realistic sampling when creating the 1.2M agent population.

2. **Wage distributions:** Instead of `industry_median × LogNormal(0, 0.25)`, we get real 10th/25th/50th/75th/90th percentile wages **per occupation**. A cashier in Mississippi earns $21,190 (median), not $30,000 (our current Retail industry median).

3. **Industry×occupation cross-tabulation:** Critical for assigning occupations to agents who already have an industry code, ensuring internal consistency (no cashiers assigned to construction).

### How It Will Be Used

```python
# In generate_data.py — occupation-aware agent creation:

oes = pd.read_excel("state_M2024_dl.xlsx", sheet_name="Mississippi")

# Step 1: For each industry, get occupation distribution
# Step 2: For each agent's industry, sample an occupation code
#         weighted by OES employment counts
# Step 3: Assign wage from occupation-specific distribution
#         (using percentiles to create realistic spread)

# Example: Agent in Retail Trade
#   → Sample from [Cashiers: 45K, Retail Salespersons: 38K, Stock Clerks: 22K, ...]
#   → Cashier selected → wage ~ Uniform(21K_p10, 28K_p90) or fit distribution
```

### Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Occupation resolution | 0 (none) | ~600 SOC codes | **New dimension** |
| Wage RMSE per agent | ~$10K (estimated) | ~$3K | **3× better** |
| Sector labor composition | Flat (equal within sector) | Realistic (weighted by employment) | **Fundamental** |

---

## 6. Dataset #5: Quarterly Census of Employment & Wages (QCEW)

### What It Is

The **QCEW** provides quarterly establishment-level employment and wage data, covering 95% of US jobs. It gives industry×county resolution — exactly what we need for the Mississippi sub-state regions.

### Exact Dataset Specification

| Field | Value |
|-------|-------|
| **Name** | Quarterly Census of Employment and Wages |
| **Source** | Bureau of Labor Statistics |
| **Direct Download** | https://data.bls.gov/cew/apps/data_views/data_views.htm |
| **Filter** | State: Mississippi, Area Type: County or MSA |
| **Format** | CSV |
| **Year** | 2024Q4 (latest) |
| **Access** | Public, free |

### Exact Columns Required

| Column | Type | Description | Maps To |
|--------|------|-------------|---------|
| `area_fips` | str | County FIPS code (28001–28163) | Agent `region` (county → 5 MS regions) |
| `industry_code` | str | NAICS code (2-digit to 6-digit) | Agent `industry` |
| `month3_emplvl` | int | Employment count end-of-quarter | Region×industry employment weights |
| `avg_wkly_wage` | float | Average weekly wage | Wage calibration |
| `qtrly_estabs` | int | Number of establishments | Firm-level context |

### Why It Is Required

1. **Region×industry cross-tabulation:** Currently, region and industry are sampled independently. In reality, the Delta has far more agriculture and far fewer Professional/Technical workers than Metro Jackson. QCEW gives the exact county×industry employment matrix.

2. **Regional economic indicators:** Average wages by region×industry provide the basis for calibrating `region_resilience` (currently 5 hardcoded values: `[0.55, 0.65, 0.70, 0.68, 0.80]`).

3. **Temporal calibration:** QCEW is quarterly — matching our simulation's quarterly time step — enabling direct calibration of employment trajectories.

### How It Will Be Used

```python
# Map 82 Mississippi counties to 5 simulation regions:
COUNTY_TO_REGION = {
    # Delta: Bolivar, Coahoma, Humphreys, Leflore, Sunflower, Washington, ...
    "28011": 0, "28027": 0, "28053": 0, ...
    # North MS: Alcorn, DeSoto, Lee, Pontotoc, Tishomingo, Union, ...
    "28003": 1, "28033": 1, ...
    # Central: Hinds, Madison, Rankin, Scott, ...
    "28049": 2, "28089": 2, ...
    # South/Gulf: Harrison, Jackson, Forrest, Jones, ...
    "28047": 3, "28059": 3, ...
    # Metro Jackson: Hinds, Madison, Rankin (overlaps Central)
    "28049": 4, "28089": 4, "28121": 4, ...
}

# Build industry×region employment matrix from QCEW
# → Replace independent np.random.choice() with joint sampling
# → Automatically ensures Delta gets agriculture, Jackson gets finance
```

### Expected Impact

| Parameter | Before | After |
|-----------|--------|-------|
| `region_resilience` | 5 hardcoded values | Data-derived from regional wage growth & employment stability |
| Industry×region distribution | Independent sampling | QCEW-calibrated joint distribution |
| Regional unemployment baseline | Hardcoded `[.06,.045,.04,.042,.035]` | Actual QCEW quarterly rates |

---

## 7. Dataset #6: O*NET Task-Level Data

### What It Is

**O\*NET** (Occupational Information Network) provides detailed task, skill, ability, and technology requirements for ~1,000 occupations. It is the primary source for understanding *what workers actually do* in each job.

### Exact Dataset Specification

| Field | Value |
|-------|-------|
| **Name** | O*NET 29.0 Database (Task Statements, Technology Skills, Abilities) |
| **Source** | US DOL / O*NET Resource Center |
| **Direct Download** | https://www.onetcenter.org/database.html → "O*NET 29.0 Database" |
| **Files Required** | `Task Statements.xlsx`, `Technology Skills.xlsx`, `Abilities.xlsx`, `Skills.xlsx` |
| **Format** | Excel/CSV (tab-delimited) |
| **Records** | ~20,000 task statements across ~1,000 occupations |
| **Access** | Public, free, CC BY 4.0 license |

### Exact Columns Required — `Technology Skills.xlsx`

| Column | Type | Description | Purpose |
|--------|------|-------------|---------|
| `O*NET-SOC Code` | str | 8-digit O*NET occupation code | Join to SOC-6 |
| `Example` | str | Technology tool/skill name | Digital skill calibration |
| `Hot Technology` | str | Whether it's a high-demand tech skill | Identify AI-adjacent skills |

### Exact Columns Required — `Abilities.xlsx`

| Column | Type | Description | Purpose |
|--------|------|-------------|---------|
| `O*NET-SOC Code` | str | Occupation code | Join key |
| `Element Name` | str | Ability name (e.g., "Mathematical Reasoning") | Compute digital_skill proxy |
| `Scale ID` | str | Level / Importance | Weighting |
| `Data Value` | float | Score (1–7 or 1–5 depending on scale) | Numeric input |

### Why It Is Required

The `digital_skill` attribute is currently **entirely fabricated**:

```python
# CURRENT — invented formula with invented coefficients
digital_skill = 0.4 * edu_dig[education] + 0.4 * age_dig[age] + 0.2 * Beta(2,3)
```

O\*NET provides empirical data to construct a real digital skill score. Specifically:

1. **Technology Skills** lists actual software, tools, and technologies used in each occupation (e.g., "Python", "Microsoft Excel", "CAD software")
2. **Abilities** includes "Mathematical Reasoning", "Information Ordering", "Deductive Reasoning" — cognitive abilities that correlate with digital adaptability
3. **Hot Technologies** flags high-demand skills, allowing identification of workers already in AI-adjacent roles

### How It Will Be Used

```python
# Construct data-driven digital_skill score per occupation:

tech_skills = pd.read_excel("Technology Skills.xlsx")
abilities = pd.read_excel("Abilities.xlsx")

# Count "digital/computational" technology skills per occupation
digital_keywords = ["software", "python", "sql", "data", "cloud", "programming",
                     "database", "analytics", "machine learning", "excel"]
tech_skills["is_digital"] = tech_skills["Example"].str.lower().str.contains(
    "|".join(digital_keywords)
)
digital_count = tech_skills.groupby("O*NET-SOC Code")["is_digital"].mean()

# Get cognitive ability scores (Mathematical Reasoning, Info Ordering)
cognitive = abilities[abilities["Element Name"].isin([
    "Mathematical Reasoning", "Information Ordering", "Deductive Reasoning"
])]
cog_score = cognitive.groupby("O*NET-SOC Code")["Data Value"].mean()

# Composite digital skill: 0.6 × digital_tech_fraction + 0.4 × normalized_cognitive
# Then map SOC-8 → SOC-6 → agent
```

### Expected Impact

- **digital_skill** goes from a fabricated number to an **occupation-grounded empirical measure**
- Augmentation/displacement predictions become more accurate (digital_skill is a core input to both probability calculations)
- LLM prompt context becomes richer (can include actual technology skills in worker profile)

---

## 8. Dataset #7: Current Population Survey (CPS) Microdata

### What It Is

The **CPS** is a monthly survey of ~60,000 households providing the official US unemployment rate. The **Annual Social and Economic Supplement (ASEC)** adds detailed income and health insurance data. The **Displaced Workers Supplement (DWS)** is conducted every two years and provides data on workers who lost jobs due to plant closings, position elimination, or insufficient work.

### Exact Dataset Specification

| File | Description | URL |
|------|-------------|-----|
| **CPS Basic Monthly** | Employment status, occupation, industry, demographics | https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html |
| **CPS ASEC** | Annual earnings, poverty, health insurance | https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html |
| **CPS DWS** | Displaced workers: reason, duration, reemployment, wage change | https://www.bls.gov/cps/cps_displaced.htm |

### Key Variables from DWS (Displaced Workers Supplement)

| Variable | Type | Description | Maps To |
|----------|------|-------------|---------|
| `PEDWRSN` | int | Reason for displacement (plant closing, position eliminated, etc.) | Displacement cause validation |
| `PEDWLKO` | int | Duration of joblessness (weeks) | Retraining timing calibration |
| `PRWKSCH` | float | Pre-displacement weekly earnings | Wage decline calibration |
| `PRERNWA` | float | Current weekly earnings (if reemployed) | Post-displacement wage recovery |
| `PEDWAVL` | int | Whether sought retraining | Retraining uptake baseline |
| `PRDTOCC1` | str | Pre-displacement occupation | Occupation-level displacement rates |
| `PRMJOCC1` | str | Current occupation (if reemployed) | Occupation transition matrix |

### Why It Is Required

Three critical config parameters are currently hardcoded without empirical basis:

| Parameter | Current Value | What CPS DWS Provides |
|-----------|--------------|----------------------|
| `wage_decline_displaced` | 0.15 (15%) | Actual median wage loss for displaced workers: ~**17%** for those reemployed full-time, ~**30%** if including part-time/non-reemployed (BLS DWS 2024) |
| `retraining_effectiveness` | 0.30 (30%/quarter) | Actual reemployment rates by duration: ~60% within 1 year, ~75% within 2 years — calibrate quarterly transition probabilities |
| `wage_growth_augmented` | 0.03 (3%/quarter) | Wage premium for workers in AI-augmented occupations vs. baseline CPS earnings growth |

### How It Will Be Used

```python
# Calibrate wage_decline_displaced from DWS:
dws = pd.read_csv("cps_displaced_2024.csv")
dws_ms = dws[dws.STATE == 28]  # Mississippi

# Median wage change for displaced workers who found new job
wage_before = dws_ms["PRWKSCH"]
wage_after = dws_ms["PRERNWA"]
median_decline = 1 - (wage_after / wage_before).median()
# → Updates config.yaml: wage_decline_displaced: 0.17 (or actual value)

# Calibrate quarterly retraining rate from reemployment duration:
# P(reemployed within 3 months) ≈ quarterly retraining_effectiveness
quarterly_rate = (dws_ms["PEDWLKO"] <= 13).mean()  # ≤13 weeks
# → Updates config.yaml: retraining_effectiveness: {actual value}
```

---

## 9. Dataset #8: Mississippi Retraining Program Records

### What It Is

State-level records from the **Mississippi Department of Employment Security (MDES)** and **Workforce Innovation and Opportunity Act (WIOA)** programs, documenting actual retraining enrollment, completion, and post-program employment outcomes.

### Exact Dataset Specification

| Field | Value |
|-------|-------|
| **Name** | WIOA Annual Performance Reports — Mississippi |
| **Source** | US DOL Employment & Training Administration (ETA) |
| **Direct Access** | https://www.dol.gov/agencies/eta/performance/results |
| **State Source** | Mississippi Department of Employment Security (MDES), https://mdes.ms.gov |
| **Format** | PDF reports (public), CSV/database (FOIA request to MDES) |
| **Access** | Public aggregates; individual records require data-sharing agreement |

### Key Metrics Needed

| Metric | Description | Maps To |
|--------|-------------|---------|
| Enrollment rate | % of eligible displaced workers who enroll | Base `retraining_effectiveness` |
| Completion rate | % of enrolled who complete program | Modifies `retraining_effectiveness` |
| Post-training employment rate | % employed 2 & 4 quarters after exit | Calibrates retrained → employed transition |
| Post-training median wage | Median quarterly earnings after exit | Calibrates wage recovery for retrained workers |
| Program type breakdown | IT/digital, healthcare, manufacturing, etc. | Sector-specific retraining rates |
| Demographic breakdown | By age, education, race, gender | Archetype-specific retraining rates |
| Regional breakdown | By WDA (Workforce Development Area) | Region-specific retraining capacity |

### Why It Is Required

The single parameter `retraining_effectiveness: 0.30` assumes all displaced workers across all demographics, occupations, and regions retrain at the same rate. In reality:

- Mississippi's WIOA programs had a **67% employment rate** at Q2 after exit (2023 data) but only ~15% of eligible workers enroll
- Completion rates vary: **IT programs: ~55%**, **healthcare credentialing: ~72%**, **manufacturing: ~60%**
- Workers 55+ complete at roughly **half the rate** of workers 25–34
- Rural areas (Delta) have **40% fewer** program slots per capita than Metro Jackson

### How It Will Be Used

```python
# Replace single retraining_effectiveness with stratified rates:
# In config.yaml:
#   retraining_effectiveness → retraining_base_enrollment: 0.15
#                            → retraining_completion_rate: [0.72, 0.55, 0.60, ...]  # by program
#                            → retraining_employment_rate: 0.67

# In job_transition.py — stratified retraining:
#   effective_retrain_prob = enrollment × completion × employment_rate
#   Adjusted by: age_penalty, region_capacity, sector_match
```

---

## 10. Dataset #9: LODES/LEHD Commute & Employment Data

### What It Is

**LODES** (LEHD Origin-Destination Employment Statistics) provides block-level data on where people live and where they work, enabling precise geographic modeling of Mississippi's labor market.

### Exact Dataset Specification

| Field | Value |
|-------|-------|
| **Name** | LODES Version 8 — Mississippi |
| **Source** | US Census Bureau, Center for Economic Studies |
| **Direct Download** | https://lehd.ces.census.gov/data/ → LODES → Version 8 |
| **Files** | `ms_od_main_JT00_2022.csv.gz` (origin-destination), `ms_wac_S000_JT00_2022.csv.gz` (workplace area characteristics) |
| **Format** | Gzipped CSV |
| **Access** | Public, free |

### Key Files and Columns

**`ms_wac_*.csv`** (Workplace Area Characteristics):

| Column | Type | Description | Maps To |
|--------|------|-------------|---------|
| `w_geocode` | str | Workplace Census block (15-digit) | Agent `region` (block → county → region) |
| `C000` | int | Total jobs | Regional employment density |
| `CNS01`–`CNS20` | int | Jobs by NAICS sector | Industry×region employment matrix |
| `CA01`–`CA03` | int | Jobs by age group | Age×region distribution |
| `CE01`–`CE03` | int | Jobs by earnings class | Wage×region calibration |

### Why It Is Required

1. **Industry×region joint distribution:** Replaces independent sampling of region and industry. LODES shows exactly how many Manufacturing jobs are in each Census block (and therefore county/region).

2. **Commute patterns:** Workers in the Delta may commute to Jackson for professional jobs — LODES captures this, affecting how displacement in one region impacts another.

3. **Employment density:** Informs `region_resilience` — denser employment areas have more reemployment options.

### How It Will Be Used

```python
# Build industry×region employment matrix:
wac = pd.read_csv("ms_wac_S000_JT00_2022.csv.gz")
wac["county"] = wac["w_geocode"].astype(str).str[:5]
wac["region"] = wac["county"].map(COUNTY_TO_REGION)

# Sum by region × industry (CNS01-CNS20 → map to 12 sectors)
region_industry = wac.groupby("region")[["CNS01","CNS02",...,"CNS20"]].sum()

# → Joint probability table for sampling (replaces independent np.random.choice)
# → Data-driven region_resilience from employment diversity
```

---

## 11. Dataset #10: Digital Skill Proxies

### What It Is

Multiple datasets can serve as proxies for the `digital_skill` attribute, which is currently fabricated. The best proxies come from **NTIA Internet Use Survey** (supplement to CPS), **OECD PIAAC** (adult skills assessment), and **Burning Glass / Lightcast** job posting data.

### Exact Dataset Specifications

#### Option A: NTIA Internet Use Survey

| Field | Value |
|-------|-------|
| **Name** | NTIA Internet Use Survey (CPS Computer and Internet Use Supplement) |
| **Source** | NTIA / Census Bureau |
| **Download** | https://www.ntia.gov/data/digital-nation-data-explorer |
| **Format** | CPS microdata with supplement variables |
| **Key Variables** | Internet use, device ownership, online activities, telecommuting |
| **Access** | Public, free |

#### Option B: Lightcast (formerly Burning Glass) Job Posting Data

| Field | Value |
|-------|-------|
| **Name** | Lightcast Job Postings Analytics |
| **Source** | Lightcast (commercial) |
| **Access** | Commercial license required; some academic access via NBER/Upjohn Institute |
| **Key Variables** | Skills demanded by occupation×region, digital skill requirements |
| **Granularity** | Occupation × MSA × quarter |

### Why It Is Required

`digital_skill` directly controls both displacement and augmentation probabilities:

```python
# From job_transition.py:
displacement_prob = (exposure - 0.60) * 2.0 * (1.0 - digital_skill) * ...  # low skill → displaced
augmentation_prob = (exposure - 0.35) * 1.5 * digital_skill * ...           # high skill → augmented
```

An error of ±0.15 in digital_skill translates to roughly ±30% change in displacement probability for borderline workers. The current fabricated formula has no empirical validation.

### How It Will Be Used

```python
# From O*NET + NTIA, construct per-occupation digital_skill:

# Method: 
# 1. O*NET technology skills → compute digital tool density per SOC
# 2. O*NET abilities → weight cognitive/analytical scores
# 3. NTIA → state-level correction for MS digital infrastructure gap
# 4. Composite score, normalized [0, 1]

onet_digital = compute_onet_digital_score(soc_code)  # per occupation
ntia_state_factor = 0.85  # MS is ~15% below national average in digital access
digital_skill_empirical = onet_digital * ntia_state_factor

# Apply age modifier from NTIA data (not hardcoded):
# NTIA shows internet use by age: 18-29: 95%, 30-49: 91%, 50-64: 83%, 65+: 62%
```

---

## 12. Integration Architecture — How Each Dataset Feeds AgentTorch

### Data Integration Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IMPROVED DATA PIPELINE                               │
│                                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │  ACS PUMS  │  │  BLS OES   │  │   QCEW     │  │   LODES    │        │
│  │ (microdata)│  │ (occ×wage) │  │ (ind×county)│  │ (geo×ind)  │        │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘        │
│        │               │               │               │                │
│        └───────────────┬┼───────────────┘               │                │
│                        ││                               │                │
│                        ▼▼                               │                │
│             ┌─────────────────────┐                     │                │
│             │  generate_data.py   │◄────────────────────┘                │
│             │  (IMPROVED)         │                                      │
│             │                     │                                      │
│             │  1. Load PUMS micro-│                                      │
│             │     data (28K recs) │                                      │
│             │  2. Resample to     │◄──── OES occupation weights          │
│             │     1.2M agents     │                                      │
│             │  3. Assign SOC code │◄──── PUMS SOCP column               │
│             │  4. Join F&O score  │◄──── frey_osborne_2017.csv          │
│             │  5. Join AIOE score │◄──── felten_aioe_2021.csv           │
│             │  6. Join O*NET skill│◄──── O*NET Technology Skills         │
│             │  7. Apply QCEW/LODES│◄──── Region×industry calibration    │
│             │     regional weights│                                      │
│             │  8. Calibrate wages │◄──── OES percentiles                 │
│             │  9. Save 12 pickles │                                      │
│             └──────────┬──────────┘                                      │
│                        │                                                 │
│        ┌───────────────┼────────────────┐                                │
│        ▼               ▼                ▼                                │
│  ┌──────────┐   ┌──────────────┐  ┌──────────────┐                      │
│  │ EXISTING  │   │  NEW ATTRS   │  │ CALIBRATED   │                      │
│  │ PICKLES   │   │              │  │ CONFIG       │                      │
│  │ (improved)│   │ occupation.pk│  │              │                      │
│  │           │   │ auto_prob.pk │  │ config.yaml  │                      │
│  │ age       │   │ aioe_score.pk│  │              │                      │
│  │ education │   │              │  │ Updated:     │                      │
│  │ industry  │   │ digital_skill│  │  - wage_decl │◄── CPS DWS          │
│  │ wage      │   │ (empirical)  │  │  - retrain   │◄── MS WIOA          │
│  │ region    │   │              │  │  - resilience│◄── QCEW/BEA         │
│  │ gender    │   │              │  │  - sector_spd│◄── McKinsey/AI       │
│  │ emp_status│   │              │  │  - thresholds│◄── Calibrated        │
│  └──────────┘   └──────────────┘  └──────────────┘                      │
│        │               │                │                                │
│        └───────────────┼────────────────┘                                │
│                        │                                                 │
│                        ▼                                                 │
│             ┌─────────────────────┐                                      │
│             │   AgentTorch Runner │                                      │
│             │                     │                                      │
│             │  State Dict: 12+3   │                                      │
│             │  agent attributes   │                                      │
│             │  + calibrated params│                                      │
│             └─────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mapping: Dataset → Agent Attribute → Substep Impact

| Dataset | Agent Attribute Created/Improved | Substep That Consumes It | Impact on Simulation |
|---------|--------------------------------|-------------------------|---------------------|
| ACS PUMS | All 9 existing (with joint distributions) | All substeps | Correlated demographics |
| F&O Scores | **NEW:** `automation_probability` | `JobTransition` displacement_prob | Correct displacement targeting |
| Felten AIOE | **IMPROVED:** `ai_exposure` (occupation-level) | `AssessAIExposure`, `JobTransition` augmentation_prob | Correct augmentation targeting |
| BLS OES | **IMPROVED:** `wage`, `industry`, **NEW:** `occupation` | `JobTransition` wage updates | Realistic wage impacts |
| QCEW | **IMPROVED:** `region`×`industry` joint distribution | `JobTransition` regional stats | Correct geographic patterns |
| O\*NET | **IMPROVED:** `digital_skill` (empirical) | `AssessAIExposure`, `JobTransition` | Accurate skill-based sorting |
| CPS DWS | **CONFIG:** `wage_decline_displaced`, `retraining_effectiveness` | `JobTransition` | Calibrated transition rates |
| MS WIOA | **CONFIG:** stratified `retraining_effectiveness` | `JobTransition`, `LLMRetrainingDecision` | Realistic retraining dynamics |
| LODES | **IMPROVED:** `region` (block-level precision) | `JobTransition` regional stats | Accurate geography |
| NTIA/Lightcast | **IMPROVED:** `digital_skill` (regional correction) | All substeps | MS-specific digital gap |

---

## 13. Data Pipeline Flowchart — Current vs. Improved

### Current Pipeline (Synthetic)

```
  Hardcoded probabilities → np.random.choice(1.2M) → 9 independent pickles
                                     │
                                     ▼
                              config.yaml (hardcoded params)
                                     │
                                     ▼
                              AgentTorch Runner
                                     │
                                     ▼
                              Results (low confidence)
```

### Improved Pipeline (Data-Driven)

```
  ┌────────────────────────────────────────────────────────────────┐
  │                    EXTERNAL DATA LAYER                         │
  │                                                                │
  │  Census PUMS ─┬─► Joint demographic × occupation × wage       │
  │  BLS OES ─────┤                                                │
  │  QCEW ────────┤─► Region × industry employment matrix          │
  │  LODES ───────┘                                                │
  │                                                                │
  │  Felten AIOE ──┬─► Dual exposure scores per occupation         │
  │  F&O Scores ───┘                                                │
  │                                                                │
  │  O*NET ────────┬─► Empirical digital skill per occupation      │
  │  NTIA ─────────┘                                                │
  │                                                                │
  │  CPS DWS ──────┬─► Calibrated transition parameters            │
  │  MS WIOA ──────┘                                                │
  └───────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
                      ┌──────────────────────┐
                      │  generate_data_v2.py │
                      │                      │
                      │  Resampling +        │
                      │  merging +           │
                      │  calibration         │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  12 improved pickles │
                      │  + 3 new pickles     │
                      │  + calibrated config │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  AgentTorch Runner   │
                      │  (same code, better  │
                      │   inputs)            │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  Results             │
                      │  (policy-grade       │
                      │   confidence)        │
                      └──────────────────────┘
```

---

## 14. Impact Assessment — Expected Accuracy Improvements

### Quantitative Impact Estimates

| Improvement | Dataset(s) | Metric Affected | Current Error (est.) | Expected Error After | Improvement Factor |
|------------|-----------|-----------------|---------------------|---------------------|--------------------|
| Occupation-level resolution | PUMS + OES | Displacement distribution by sector | >200% (wrong sectors displaced) | ~30% | **~7×** |
| Dual-score displacement | F&O + Felten | Which workers are displaced vs. augmented | ~50% misclassification rate | ~15% | **~3×** |
| Joint demographic distributions | PUMS | Attribute correlations (age×edu×industry) | 100% error (zero correlation) | ~5% (real correlations) | **~20×** |
| Wage accuracy | OES + PUMS | Per-agent wage RMSE | ~$10K | ~$3K | **~3×** |
| Digital skill grounding | O\*NET + NTIA | Displacement/augmentation probability accuracy | Unquantifiable (fabricated) | Empirically grounded | **Qualitative leap** |
| Geographic precision | QCEW + LODES | Regional unemployment patterns | Region×industry independent | Correct joint distribution | **Structural fix** |
| Transition parameter calibration | CPS DWS + WIOA | Retraining rate, wage decline | ±50% (guessed) | ±10% (calibrated) | **~5×** |

### Qualitative Impact on Policy Conclusions

| Policy Conclusion | Current (Synthetic) | Expected (Data-Driven) |
|-------------------|--------------------|-----------------------|
| "Which sectors need intervention?" | Professional/Technical & Finance (91% of displacement) | Retail, Transportation, Food Service, Clerical roles (routine-task displacement) |
| "Which regions are most impacted?" | Metro Jackson (+0.79pp unemployment) | Delta & rural areas (more routine/manual labor) |
| "What type of retraining?" | Knowledge-worker upskilling | Lower-wage worker digital basics, credential programs |
| "How many workers displaced total?" | 4,930 (0.4%) | Likely **higher** (10K–30K) due to routine-task workers |
| "What is the wage impact?" | Net positive (+8.9% mean wage) | Likely **net negative for bottom quartile** (inequality worsens) |

---

## 15. Dataset Priority Matrix

### Priority × Effort × Impact

```
                        IMPACT ON ACCURACY
                    Low          Medium         High
                ┌────────────┬─────────────┬────────────────┐
        Easy    │            │ NTIA Digital│ BLS OES        │
  E     (hours) │            │ Skill       │ (occ wages)    │
  F             │            │             │                │
  F     ────────┼────────────┼─────────────┼────────────────┤
  O     Medium  │            │ QCEW        │ ACS PUMS       │
  R     (days)  │            │ (region×ind)│ (microdata)    │
  T             │            │ LODES       │ F&O Scores     │
        ────────┼────────────┼─────────────┼────────────────┤
        Hard    │            │ Lightcast   │ MS WIOA Records│
        (weeks) │            │ (commercial)│ (FOIA needed)  │
                │            │             │ CPS DWS        │
                └────────────┴─────────────┴────────────────┘
```

### Recommended Implementation Order

| Phase | Datasets | Timeline | Agent Attributes Improved | Config Params Improved |
|-------|---------|----------|--------------------------|----------------------|
| **Phase 1** (Critical) | ACS PUMS + F&O Scores + Felten AIOE (occupation-level) | 1–2 weeks | All demographics, `occupation`, `automation_probability`, `ai_exposure` | — |
| **Phase 2** (High) | BLS OES + O\*NET | 1 week | `wage`, `digital_skill` | — |
| **Phase 3** (Medium) | QCEW + LODES | 1 week | `region` (joint distribution) | `region_resilience`, regional unemployment baselines |
| **Phase 4** (Calibration) | CPS DWS + MS WIOA | 2–3 weeks | — | `wage_decline_displaced`, `retraining_effectiveness`, `automation_threshold` |
| **Phase 5** (Polish) | NTIA + Lightcast | 1–2 weeks | `digital_skill` (regional correction) | `sector_adoption_speed` |

---

## 16. Implementation Roadmap

### Phase 1: Occupation-Level Agent Resolution

```
Week 1–2:
  □ Download ACS 5-Year PUMS for Mississippi (csv_pms.zip)
  □ Download Frey & Osborne Table A.1 (702 SOC scores)
  □ Download Felten AIOE dataset (774 SOC scores)
  □ Build SOC-6 crosswalk (PUMS SOCP → SOC-6 → F&O → Felten)
  □ Rewrite generate_data.py to use PUMS resampling
  □ Add `occupation`, `automation_probability`, `aioe_score` pickles
  □ Update config.yaml with new agent properties
  □ Modify job_transition.py for dual-score model
  □ Run simulation, compare results to current baseline
```

### Phase 2: Wage & Skill Calibration

```
Week 3:
  □ Download BLS OES Mississippi (oesm28.xlsx)
  □ Download O*NET 29.0 (Technology Skills + Abilities)
  □ Replace wage sampling with OES percentile distributions
  □ Compute empirical digital_skill from O*NET
  □ Validate digital_skill distribution against NTIA benchmarks
```

### Phase 3: Geographic Precision

```
Week 4:
  □ Download QCEW Mississippi (county × industry)
  □ Download LODES v8 Mississippi (ms_wac_*.csv.gz)
  □ Build county → region mapping (82 counties → 5 regions)
  □ Construct industry×region joint probability matrix
  □ Derive data-driven region_resilience from employment diversity
  □ Update generate_data.py for correlated region×industry sampling
```

### Phase 4: Parameter Calibration

```
Week 5–7:
  □ Download CPS Displaced Workers Supplement (2024)
  □ Request MS WIOA program data from MDES
  □ Calibrate wage_decline_displaced from DWS
  □ Calibrate retraining_effectiveness (stratified by age, education, region)
  □ Calibrate automation_threshold against observed displacement patterns
  □ Run sensitivity analysis across calibrated parameter ranges
  □ Document calibration methodology
```

### Phase 5: Validation

```
Week 8:
  □ Compare simulation output to actual MS labor statistics (2024–2025)
  □ Run with calibration=true in config.yaml
  □ Use AgentTorch's P3O optimizer to fine-tune learnable parameters
  □ Document accuracy improvements with before/after comparison
  □ Produce updated report_lpm.md with data-driven results
```

---

## 17. References

### Datasets

1. US Census Bureau. *American Community Survey Public Use Microdata Sample (PUMS)*. https://data.census.gov/mdat/
2. Bureau of Labor Statistics. *Occupational Employment and Wage Statistics (OES)*. https://www.bls.gov/oes/
3. Bureau of Labor Statistics. *Quarterly Census of Employment and Wages (QCEW)*. https://www.bls.gov/cew/
4. US Census Bureau. *LEHD Origin-Destination Employment Statistics (LODES)*. https://lehd.ces.census.gov/data/
5. US DOL. *O\*NET OnLine Database*. https://www.onetcenter.org/database.html
6. Bureau of Labor Statistics. *Current Population Survey — Displaced Workers Supplement*. https://www.bls.gov/cps/cps_displaced.htm
7. NTIA. *Internet Use Survey*. https://www.ntia.gov/data/digital-nation-data-explorer
8. Mississippi Department of Employment Security (MDES). *WIOA Performance Reports*. https://mdes.ms.gov

### Academic Papers

9. Frey, C.B. & Osborne, M.A. (2017). *The future of employment: How susceptible are jobs to computerisation?* Technological Forecasting and Social Change, 114, 254–280.
10. Felten, E., Raj, M., & Seamans, R. (2021). *Occupational, industry, and geographic exposure to artificial intelligence.* Strategic Management Journal, 42(12), 2195–2217.
11. Acemoglu, D. & Restrepo, P. (2020). *Robots and jobs: Evidence from US labor markets.* Journal of Political Economy, 128(6), 2188–2244.
12. Chopra, A., et al. (2024). *AgentTorch: Large Population Models.* MIT Media Lab.

---

*This dataset requirements document accompanies `report_lpm.md` (architecture) and `report.md` (results) for the Mississippi AI Workforce Impact Simulation built on the [AgentTorch](https://github.com/AgentTorch/AgentTorch) framework. All recommended datasets are publicly accessible unless otherwise noted.*
