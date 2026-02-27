# Mississippi AI Job Impact Simulation — Report

## Large Population Model: Impact of AI Exposure on 1.2 Million Mississippi Workers

**Framework:** AgentTorch (Large Population Models)  
**Date:** February 2026  
**Simulation Period:** Q1 2025 → Q4 2027 (12 quarters, 3 years)  
**Population:** 1,200,000 synthetic workers representing Mississippi's labor force  
**Runtime:** ~1.1 seconds on CPU (Apple Silicon)

---

## 1. Executive Summary

This simulation models the impact of accelerating AI adoption on Mississippi's workforce of approximately 1.2 million workers over a 3-year period (2025–2027). Using the AgentTorch Large Population Model (LPM) framework, we simulate individual-level job transitions for every worker — accounting for their industry, education, age, digital skills, and geographic region — as AI penetration grows across the economy.

### Key Findings

| Metric | Value |
|--------|-------|
| **Workers Displaced (final)** | **4,930** (0.4% of workforce) |
| **Workers Augmented by AI** | **112,868** (9.4% of workforce) |
| **Workers At Risk** | **279,290** (23.3% of workforce) |
| **Workers Successfully Retrained** | **49,311** (4.1% of workforce) |
| **Workers Unaffected** | **753,601** (62.8% of workforce) |
| **Peak Displacement** | **17,058 workers** (Q3 2025) |
| **Final Unemployment Rate** | **4.29%–4.54%** (varies by region) |
| **Final Mean Wage** | **$41,930** (up from ~$38,500 baseline) |

> **Bottom line:** AI primarily *augments* Mississippi's workforce rather than displacing it. For every worker displaced, approximately **23 workers** are augmented. However, nearly **1 in 4 workers** (23.3%) end up in an "at-risk" category — meaning their jobs face growing AI exposure but haven't yet been fully automated.

---

## 2. Simulation Architecture

This simulation follows the AgentTorch LPM architecture, consisting of:

### 2.1 Agents (Workers)
Each of the 1,200,000 agents has the following properties:

| Property | Description | Distribution Source |
|----------|-------------|-------------------|
| **Age Group** | 6 groups (18–24 through 65+) | ACS Mississippi estimates |
| **Gender** | Binary (48% M, 52% F) | Census proportions |
| **Education** | 5 levels (< HS through Graduate+) | ACS educational attainment |
| **Industry** | 12 sectors (NAICS-2 mapped) | BLS QCEW for Mississippi |
| **AI Exposure** | 0–1 score per worker | Felten et al. (2021) AI Occupational Exposure Index |
| **Wage** | Annual ($15K–$200K) | BLS OES medians by industry |
| **Digital Skill** | 0–1 score | Correlated with education & age |
| **Region** | 5 MS economic regions | Population distribution |
| **Employment Status** | Employed / Unemployed / Underemployed | Dynamic |
| **Job Impact Status** | Unaffected / Augmented / At-Risk / Displaced / Retrained | Dynamic |

### 2.2 Substeps (per quarter)

**Substep 0 — AI Exposure Assessment:**
- Computes each worker's *effective* AI exposure as a function of:
  - Base AI exposure score × time-compounding adoption rate
  - Industry-specific adoption speed multiplier
  - Education modifier (knowledge workers = higher exposure)
  - Age adaptation factor (younger workers adapt faster)
- Updates the economy-wide AI Penetration Index (+3% per quarter)

**Substep 1 — Job Transition:**
- **Displacement probability:** `f(exposure > 0.6, low digital skill, low regional resilience)`
- **Augmentation probability:** `f(exposure ∈ [0.35, 0.6], high digital skill, high resilience)`
- **Retraining:** Displaced workers retrain with probability modulated by digital skill, education, and age
- **Wage updates:** Augmented +3%/quarter, Displaced −15%, Retrained +10% recovery
- **Tracks:** sector-level displacement, regional unemployment, GDP impact

### 2.3 Environment Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| AI Adoption Rate | 8% quarterly acceleration | Industry estimates |
| Automation Threshold | 0.60 exposure score | Calibrated |
| Augmentation Threshold | 0.35 exposure score | Calibrated |
| Retraining Effectiveness | 30% per quarter | Policy research |
| Wage Growth (Augmented) | 3% per quarter | BLS wage growth data |
| Wage Decline (Displaced) | 15% | Labor economics literature |

---

## 3. Results: Quarterly Trajectory

### 3.1 Worker Status Over Time

| Quarter | Displaced | Augmented | Retrained | Unemployed |
|---------|-----------|-----------|-----------|------------|
| Q1 2025 | 12,821 | 38,874 | 0 | 60,941 |
| Q2 2025 | 16,818 | 56,066 | 5,669 | 64,938 |
| Q3 2025 | 17,058 | 69,059 | 12,871 | 65,178 |
| Q4 2025 | 15,775 | 79,063 | 19,930 | 63,895 |
| Q1 2026 | 14,095 | 87,023 | 26,063 | 62,215 |
| Q2 2026 | 12,299 | 93,311 | 31,455 | 60,419 |
| Q3 2026 | 10,617 | 98,405 | 35,942 | 58,737 |
| Q4 2026 | 9,146 | 102,602 | 39,648 | 57,266 |
| Q1 2027 | 7,819 | 105,969 | 42,764 | 55,939 |
| Q2 2027 | 6,650 | 108,713 | 45,398 | 54,770 |
| Q3 2027 | 5,767 | 110,971 | 47,503 | 53,887 |
| Q4 2027 | 4,930 | 112,868 | 49,311 | 53,050 |

**Key observations:**
- **Peak displacement occurs in Q3 2025** (17,058 workers), then steadily declines as the most vulnerable workers are already displaced and retraining kicks in.
- **Augmentation grows continuously** throughout the period, from 38,874 to 112,868, reflecting AI tools becoming productivity enhancers for digitally skilled workers.
- **Retraining absorbs displaced workers:** By Q4 2027, 49,311 workers have successfully retrained — roughly **10× the final displaced count**.
- **Unemployment trends downward** after peaking around Q3 2025, dropping from ~65,000 to ~53,000 as retraining and new AI-adjacent jobs absorb displaced workers.

---

## 4. Sector Analysis

### 4.1 Displacement & Augmentation by Industry

| Industry Sector | Workers | Displaced | Augmented | Disp. Rate | Aug. Rate |
|----------------|---------|-----------|-----------|------------|-----------|
| **Professional/Technical** | ~84,000 | **28,529** | 37,681 | **34.0%** | 44.9% |
| **Finance/Insurance** | ~60,000 | **20,910** | 26,915 | **34.9%** | 44.9% |
| **Manufacturing** | ~144,000 | 3,848 | 26,322 | 2.7% | 18.3% |
| **Retail Trade** | ~132,000 | 400 | 8,088 | 0.3% | 6.1% |
| **Government** | ~96,000 | 339 | 6,794 | 0.4% | 7.1% |
| **Transportation** | ~72,000 | 200 | 3,596 | 0.3% | 5.0% |
| **Healthcare** | ~168,000 | 10 | 1,995 | <0.1% | 1.2% |
| **Accommodation/Food** | ~108,000 | 5 | 688 | <0.1% | 0.6% |
| **Education** | ~108,000 | 0 | 304 | 0% | 0.3% |
| **Agriculture/Forestry** | ~48,000 | 0 | 0 | 0% | 0% |
| **Construction** | ~84,000 | 0 | 0 | 0% | 0% |
| **Other Services** | ~96,000 | 0 | 485 | 0% | 0.5% |

### 4.2 Key Sector Insights

- **Professional/Technical and Finance/Insurance** bear the overwhelming burden of displacement — together accounting for **91%** of all displaced workers. These are knowledge-intensive sectors with the highest baseline AI exposure scores (0.65 and 0.70 respectively).
- **However, these same sectors also have the highest augmentation rates** (45%), meaning AI is simultaneously creating productivity gains for those with strong digital skills.
- **Manufacturing** shows significant augmentation (18.3%) but low displacement (2.7%), suggesting AI primarily enhances existing production processes rather than eliminating jobs outright.
- **Healthcare, Education, Agriculture, and Construction** are largely insulated, with regulatory barriers, physical labor requirements, and human-interaction needs protecting them from displacement.

---

## 5. Regional Analysis

### 5.1 Unemployment by Region

| Region | Final Unemployment | Baseline | Change |
|--------|-------------------|----------|--------|
| **Delta** | **4.54%** | 6.0% | −1.46pp |
| **North MS** | 4.45% | 4.5% | −0.05pp |
| **Central** | 4.47% | 4.0% | +0.47pp |
| **South/Gulf** | 4.43% | 4.2% | +0.23pp |
| **Metro (Jackson)** | **4.29%** | 3.5% | +0.79pp |

### 5.2 Regional Insights

- **The Delta region**, despite having the lowest economic resilience (0.55), sees a net *decrease* in unemployment — primarily because its workforce is concentrated in agriculture and low-AI-exposure industries.
- **Metro Jackson** experiences the largest *increase* (+0.79pp) because it hosts the highest concentration of Professional/Technical and Finance workers — the sectors most exposed to AI displacement.
- The **convergence of regional unemployment rates** (all clustering around 4.3–4.5%) is a notable finding: AI's impact redistributes employment stress from historically disadvantaged areas toward economically stronger metro regions.

---

## 6. Demographic Analysis

### 6.1 Age Group Impact
- **Younger workers (18–34)** experience higher AI exposure due to their concentration in tech-adjacent roles but also have the highest retraining success rates.
- **Workers 55+** face the most acute displacement risk because their lower digital skills and reduced retraining capacity compound the exposure effect.
- **The "at-risk" population** is spread broadly across age groups, with the 35–54 cohort representing the largest absolute number.

### 6.2 Education Level Impact
- **Graduate+ workers** have the highest AI exposure scores but their strong digital skills channel this into augmentation rather than displacement.
- **< High School workers** have low exposure and are largely unaffected, but also miss the wage-growth benefits of augmentation.
- **HS/GED and Some College workers** represent the most policy-critical group: moderate exposure, limited digital skills, and the highest displacement-to-augmentation ratio.

---

## 7. Economic Impact

| Metric | Value |
|--------|-------|
| **Final Mean Wage** | $41,930 (↑ 8.9% from baseline $38,500) |
| **Final Median Wage** | $39,165 |
| **Net GDP Impact (3-year)** | Positive (augmentation wage gains exceed displacement losses) |
| **Wage Growth for Augmented Workers** | ~+36% cumulative (3% compounding × 12 quarters) |
| **Wage Loss for Displaced Workers** | −15% initial; partially recovered through retraining |

The overall wage picture is **net positive** because the augmented population (112,868 workers × ~$50K average × 36% growth) generates far more aggregate income than the displaced population loses. However, this **masks significant inequality** — the gains accrue to already-skilled workers while the losses hit the most vulnerable.

---

## 8. Assumptions & Limitations

### Broad Assumptions Made

1. **AI exposure scores** are mapped from Felten et al. (2021) at the industry-sector level, not the individual occupation level. Reality is far more heterogeneous within sectors.
2. **Adoption rates** (8% quarterly acceleration) are assumed uniform across the state. In practice, Metro Jackson would adopt faster than the Delta.
3. **Retraining effectiveness** (30% per quarter) is optimistic and assumes accessible, funded programs. Mississippi's actual retraining infrastructure may not support this.
4. **Stochastic displacement** uses simplified probability models. Real displacement depends on firm-level decisions, capital availability, and regulatory environments.
5. **No network effects** are modeled (e.g., a factory closure affecting supply chain workers). The simulation treats workers independently.
6. **Synthetic population data** — while calibrated to BLS/ACS aggregate statistics, individual worker profiles are generated stochastically rather than from actual microdata.
7. **No migration** is modeled — workers don't leave Mississippi or move between regions.

### Framework Limitations

- The simulation uses CPU-only execution. For production-grade analysis, GPU acceleration (supported by AgentTorch) would enable sensitivity analysis across thousands of parameter combinations.
- No LLM-driven behavioral modeling is included; worker decisions (e.g., whether to retrain) follow parametric rules rather than LLM-generated behavior.
- Calibration against ground-truth labor data was not performed (the `calibration` flag is set to `false`).

---

## 9. Policy Implications

Based on the simulation results, several policy priorities emerge for Mississippi:

### High Priority
1. **Targeted retraining for Professional/Technical and Finance workers** — These sectors account for 91% of displacement. Digital upskilling programs focused on AI-complementary skills (prompt engineering, AI system oversight, data analysis) would convert at-risk workers into augmented workers.

2. **Regional support for Metro Jackson** — Counter-intuitively, the state's economic hub faces the steepest unemployment increase. State economic development efforts should include AI transition support alongside traditional rural development.

3. **Digital skills initiative for mid-career workers** — The 35–54 age cohort with HS/GED or Some College education is the most vulnerable. Community college programs focused on practical AI literacy could shift these workers from the at-risk category.

### Medium Priority
4. **Manufacturing AI integration support** — Mississippi's significant manufacturing base (12% of workforce) shows high augmentation potential. Supporting firms in adopting AI tools while retaining workers could boost state productivity.

5. **Early warning system** — The simulation shows that peak displacement occurs within the first 6 quarters. Monitoring leading indicators (industry AI investment, job posting changes) could trigger proactive interventions.

### Long-Term
6. **Education pipeline reform** — Ensuring K-12 and higher education integrate AI literacy would reduce the future at-risk population from the current 23.3%.

---

## 10. Technical Implementation

### Model Architecture (AgentTorch Pattern)

```
mississippi_ai/
├── __init__.py              # Module entry point (exposes registry)
├── simulator.py             # Registry setup (substeps + init helpers)
├── generate_data.py         # Synthetic population generator
├── yamls/
│   └── config.yaml          # Full simulation configuration
├── substeps/
│   ├── __init__.py
│   ├── ai_exposure.py       # Substep 0: AssessAIExposure + UpdateExposure
│   ├── job_transition.py    # Substep 1: JobTransition
│   └── utils.py             # Initialization helpers
└── data/
    └── simulation_results.json
```

### Key Design Decisions
- **Vectorized operations:** All 1.2M agents are processed simultaneously using PyTorch tensor operations (no Python loops over agents), enabling sub-second execution.
- **Follows AgentTorch conventions:** `SubstepAction` for policies, `SubstepTransition` for state updates, YAML-driven configuration, Registry-based module discovery.
- **Learnable parameters:** The `automation_threshold`, `augmentation_threshold`, `retraining_rate`, and `adoption_acceleration` are marked as learnable — meaning they could be calibrated against real labor market data using AgentTorch's differentiable optimization.

---

## 11. How to Reproduce

```bash
# 1. Activate virtual environment
cd /path/to/AgentTorch
source .venv/bin/activate

# 2. Generate population data (if needed)
python agent_torch/models/mississippi_ai/generate_data.py

# 3. Run simulation
python run_mississippi_sim.py
```

---

## 12. Data Requirements for Improved Simulation Accuracy

To enhance the precision and reliability of the simulation outcomes, the following data improvements are recommended:

1. **Granular AI Exposure Metrics:**
   - Obtain industry- and occupation-specific AI exposure data at a more granular level than the Felten et al. (2021) index. This could involve proprietary data sources or detailed surveys from industry groups.

2. **Localized Adoption Rates:**
   - Collect data on actual AI adoption rates by industry and region within Mississippi, potentially from state economic development agencies or industry reports.

3. **Detailed Retraining Program Efficacy:**
   - Gather data on the effectiveness of existing retraining programs in Mississippi, including completion rates and post-training employment outcomes, to better calibrate the retraining effectiveness parameter.

4. **Displacement Case Studies:**
   - Conduct case studies or surveys of firms in Mississippi that have undergone AI-driven displacement to understand the nuances of job transition dynamics, including the role of digital skills and regional economic conditions.

5. **Longitudinal Wage and Employment Data:**
   - Acquire longitudinal data on wages and employment status for a representative sample of Mississippi workers to validate and calibrate the simulation's wage growth and displacement recovery trajectories.

6. **Sectoral AI Impact Studies:**
   - Commission or access studies on the impact of AI in specific sectors prevalent in Mississippi (e.g., manufacturing, agriculture) to refine the sectoral displacement and augmentation parameters.

7. **Behavioral Response Data:**
   - Collect data on worker and employer behavioral responses to AI integration, including mobility, retraining uptake, and changes in job search behavior, to inform the simulation's behavioral modeling components.

---

## 12.1 How Frey & Osborne (2017) Automation Scores Would Improve This Simulation

### Background

Frey & Osborne (2013, published 2017) estimated the probability of computerization for **702 detailed occupations** (6-digit SOC codes) in the US labor market. Their scores range from 0 (no automation risk) to 1 (near-certain automation), derived from expert assessments of bottlenecks in three areas: perception/manipulation, creative intelligence, and social intelligence. This remains one of the most widely cited occupation-level automation risk datasets in labor economics.

This is fundamentally different from the **Felten et al. (2021) AI Exposure Index** currently used in our simulation, which measures the degree to which AI *capabilities* overlap with an occupation's task requirements — not the probability that the occupation will be *eliminated*.

### Current Limitation in Our Model

Our simulation maps AI exposure at the **industry level** (12 NAICS-2 sectors), assigning a single exposure score to all workers within a sector. This creates two critical problems:

1. **Within-sector heterogeneity is lost.** A bank teller (SOC 43-3071, F&O score: 0.98) and a financial analyst (SOC 13-2051, F&O score: 0.23) both work in "Finance/Insurance" but face radically different automation risks. Our simulation treats them identically.

2. **Exposure ≠ Displacement.** The Felten index measures how much AI *touches* an occupation, not whether it *replaces* it. A radiologist has high AI exposure but low displacement risk (AI assists diagnosis). The F&O scores directly estimate displacement probability, which is what our `job_transition.py` substep actually needs.

### Specific Improvements with F&O Integration

#### Improvement 1: Occupation-Level Agent Resolution

Instead of assigning each agent an industry-level AI exposure score, we would assign each of the 1.2M agents both a **6-digit SOC code** (drawn from Mississippi-specific occupation distributions within each industry via BLS OES data) and the corresponding **F&O automation probability**:

| Current Model | Improved Model |
|--------------|----------------|
| 12 industry-level exposure scores | 702 occupation-level automation scores |
| All Finance workers share score 0.70 | Bank tellers = 0.98, Financial analysts = 0.23, Loan officers = 0.61 |
| Single dimension (exposure) | Two dimensions (exposure + automation risk) |

This would require adding an `occupation_code` and `automation_probability` attribute to each agent in the YAML config, and sourcing Mississippi's occupation-by-industry matrix from BLS Occupational Employment Statistics (OES).

#### Improvement 2: Dual-Score Displacement Model

The `JobTransition` substep would use **both** scores in a composite model:

| Score | Measures | Role in Simulation |
|-------|----------|-------------------|
| **F&O Automation Probability** | Likelihood the entire job is *automated away* | Drives **displacement** probability |
| **Felten AI Exposure Index** | Degree AI capabilities *overlap with* job tasks | Drives **augmentation** probability |

The key insight: a worker with **high F&O score + high Felten score** faces displacement, while a worker with **low F&O score + high Felten score** is a prime candidate for augmentation. Our current single-score model cannot make this distinction.

#### Improvement 3: Corrected Sector Outcomes

The F&O scores would materially change our sector-level results:

| Sector | Current Disp. Rate | Expected with F&O | Reason |
|--------|-------------------|-------------------|--------|
| **Retail Trade** | 0.3% | **~15–25%** | Cashiers (0.97), retail salespersons (0.92) have very high F&O scores; our industry-level score (0.30) dramatically underestimates risk |
| **Transportation** | 0.3% | **~10–20%** | Truck drivers (0.79), taxi drivers (0.89) have high F&O scores hidden by the low industry average |
| **Accommodation/Food** | <0.1% | **~5–15%** | Food prep (0.87), waitstaff (0.94) have high F&O scores despite low AI "exposure" |
| **Professional/Technical** | 34.0% | **~10–15%** | Many professional roles (lawyers 0.04, software devs 0.04, engineers 0.02) have *very low* F&O scores; our model over-displaces this sector |
| **Finance/Insurance** | 34.9% | **~15–25%** | Splits into high-risk clerical (tellers 0.98) vs. low-risk analytical (analysts 0.23) |

This is the **most consequential improvement**: our current simulation likely **over-displaces** knowledge workers and **under-displaces** routine service and manual workers — exactly the opposite of what F&O predicts for automation risk.

#### Improvement 4: Better Policy Targeting

With occupation-level resolution, policy recommendations would shift:

- **Current model** → Focus retraining on Professional/Technical and Finance (broad sectors)
- **F&O-enhanced model** → Focus retraining on *specific occupations*: bank tellers, cashiers, truck drivers, data entry clerks, food service workers — the actual high-automation-risk roles, many of which are lower-wage and require different retraining interventions than knowledge workers

#### Improvement 5: Wage Impact Recalibration

Since F&O high-risk occupations skew toward **lower-wage, routine jobs** (median wage ~$28K) rather than the high-wage knowledge jobs our current model displaces, the economic impact would shift:
- **Aggregate wage loss per displaced worker** would decrease (lower-wage workers displaced)
- **Inequality impact** would worsen (losses concentrated among workers with fewer resources)
- **Retraining cost-effectiveness** would change (different skill gaps to bridge)

### Data Required for Implementation

| Data Source | What It Provides | Access |
|-------------|-----------------|--------|
| **Frey & Osborne (2017) — Table A.1** | Automation probability for 702 SOC codes | Published (open access) |
| **BLS OES — Mississippi** | Occupation counts by industry, by state | Public (bls.gov) |
| **SOC-to-NAICS crosswalk** | Maps occupations to industries | Public (BLS/Census) |
| **Mississippi MDES data** | State-specific occupation distributions by region | State agency request |

### Expected Impact on Results

If F&O scores were integrated, we would expect:
- **Total displacement count** to increase moderately (more routine jobs flagged)
- **Displacement distribution** to shift dramatically — from knowledge sectors toward retail, transportation, food service, and clerical occupations
- **Augmentation** to increase in professional/technical sectors (high exposure, low automation risk = augmentation)
- **Regional patterns** to shift — the Delta and rural areas (with more routine/manual labor) would face *higher* displacement than our current model shows, while Metro Jackson (knowledge economy) would see *less*
- **Policy priorities** to refocus on lower-wage worker support and routine-task retraining rather than knowledge-worker upskilling

### Conclusion

The Frey & Osborne automation scores would transform this simulation from an **AI exposure model** into a true **automation displacement model**. The current Felten-only approach answers "which jobs interact with AI?" — the F&O scores answer the more critical question: "which jobs will AI *replace*?" Combining both would yield a dual-axis model that distinguishes augmentation from displacement at the occupation level, producing substantially more accurate and actionable results for Mississippi workforce policy.

---

## 13. References

1. Felten, E., Raj, M., & Seamans, R. (2021). *Occupational, industry, and geographic exposure to artificial intelligence: A novel dataset and its potential uses.* Strategic Management Journal, 42(12), 2195-2217.
2. Bureau of Labor Statistics. (2024). *Quarterly Census of Employment and Wages — Mississippi.*
3. U.S. Census Bureau. (2024). *American Community Survey 5-Year Estimates — Mississippi.*
4. Chopra, A., et al. (2024). *AgentTorch: Large Population Models.* MIT Media Lab.
5. Acemoglu, D., & Restrepo, P. (2020). *Robots and jobs: Evidence from US labor markets.* Journal of Political Economy, 128(6), 2188-2244.
6. Frey, C.B., & Osborne, M.A. (2017). *The future of employment: How susceptible are jobs to computerisation?* Technological Forecasting and Social Change, 114, 254-280.

---

*This report was generated from a simulation built on the [AgentTorch](https://github.com/AgentTorch/AgentTorch) Large Population Models framework developed at MIT Media Lab. All population data is synthetic. Results represent one plausible scenario under the stated assumptions and should not be interpreted as predictions.*
