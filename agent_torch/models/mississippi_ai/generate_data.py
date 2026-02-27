"""
Generate synthetic population data for 1.2 million Mississippi workers.

Data sources & assumptions (broad):
- Mississippi total labor force ~1.2M (BLS 2024 estimates).
- Age distribution: aligned with ACS 5-year estimates for MS working-age population.
- Industry distribution: BLS QCEW proportions for Mississippi.
- Education levels: ACS educational attainment for MS (25+).
- AI exposure scores: derived from Felten, Raj & Seamans (2021) AI Occupational Exposure Index,
  mapped to 2-digit NAICS industry sectors.
- Wage bands: approximate MS median wages by industry from BLS OES.

All values are synthetic and stochastic; no PII is involved.
"""

import os
import pickle
import numpy as np
import pandas as pd

SEED = 42
NUM_WORKERS = 1_200_000
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "populations", "mississippi")

np.random.seed(SEED)

# ── 1. Age (working-age 18–67, bucketed into 6 groups) ──────────────────────
# 0: 18-24, 1: 25-34, 2: 35-44, 3: 45-54, 4: 55-64, 5: 65+
AGE_PROBS = np.array([0.12, 0.22, 0.21, 0.20, 0.18, 0.07])
age_groups = np.random.choice(6, size=NUM_WORKERS, p=AGE_PROBS)

# ── 2. Gender (0: Male, 1: Female) ──────────────────────────────────────────
GENDER_PROBS = np.array([0.48, 0.52])
gender = np.random.choice(2, size=NUM_WORKERS, p=GENDER_PROBS)

# ── 3. Education level ──────────────────────────────────────────────────────
# 0: < HS, 1: HS/GED, 2: Some College, 3: Bachelor's, 4: Graduate+
EDU_PROBS = np.array([0.12, 0.30, 0.28, 0.19, 0.11])
education = np.random.choice(5, size=NUM_WORKERS, p=EDU_PROBS)

# ── 4. Industry sector (12 sectors, roughly NAICS-2 mapped) ─────────────────
# 0: Agriculture/Forestry   1: Manufacturing
# 2: Retail Trade           3: Healthcare
# 3: Education              5: Construction
# 6: Transportation         7: Accommodation/Food
# 8: Professional/Technical 9: Finance/Insurance
# 10: Government            11: Other Services
INDUSTRY_NAMES = [
    "Agriculture/Forestry", "Manufacturing", "Retail Trade", "Healthcare",
    "Education", "Construction", "Transportation", "Accommodation/Food",
    "Professional/Technical", "Finance/Insurance", "Government", "Other Services"
]
INDUSTRY_PROBS = np.array([0.04, 0.12, 0.11, 0.14, 0.09, 0.07, 0.06, 0.09, 0.07, 0.05, 0.08, 0.08])
INDUSTRY_PROBS = INDUSTRY_PROBS / INDUSTRY_PROBS.sum()
industry = np.random.choice(12, size=NUM_WORKERS, p=INDUSTRY_PROBS)

# ── 5. AI Exposure Score (0-1) per industry ──────────────────────────────────
# Based on Felten et al. (2021) AI Occupational Exposure + adjustments
# Higher = more tasks exposed to AI automation/augmentation
AI_EXPOSURE_BY_INDUSTRY = np.array([
    0.15,  # Agriculture/Forestry
    0.42,  # Manufacturing
    0.38,  # Retail Trade
    0.35,  # Healthcare
    0.30,  # Education
    0.18,  # Construction
    0.32,  # Transportation
    0.25,  # Accommodation/Food
    0.65,  # Professional/Technical
    0.70,  # Finance/Insurance
    0.45,  # Government
    0.28,  # Other Services
])

# Individual exposure = industry baseline + noise + education modifier
# Higher education → slightly higher exposure (knowledge work)
edu_modifier = np.array([-0.05, -0.02, 0.0, 0.05, 0.10])
base_exposure = AI_EXPOSURE_BY_INDUSTRY[industry]
noise = np.random.normal(0, 0.08, NUM_WORKERS)
ai_exposure = np.clip(base_exposure + edu_modifier[education] + noise, 0.0, 1.0)

# ── 6. Current wage (annual, approximate MS medians by industry) ─────────────
WAGE_BY_INDUSTRY = np.array([
    28000,  # Agriculture
    42000,  # Manufacturing
    30000,  # Retail
    45000,  # Healthcare
    40000,  # Education
    38000,  # Construction
    40000,  # Transportation
    24000,  # Accommodation/Food
    58000,  # Professional/Technical
    52000,  # Finance
    44000,  # Government
    32000,  # Other
])
base_wage = WAGE_BY_INDUSTRY[industry].astype(float)
wage_noise = np.random.lognormal(0, 0.25, NUM_WORKERS)
wage = np.clip(base_wage * wage_noise, 15000, 200000)

# ── 7. Employment status (0: employed, 1: unemployed, 2: underemployed) ──────
# MS unemployment ~4%, underemployment ~8%
emp_probs = np.array([0.88, 0.04, 0.08])
employment_status = np.random.choice(3, size=NUM_WORKERS, p=emp_probs)

# ── 8. Digital skill level (0-1), correlated with education and age ──────────
edu_dig = np.array([0.15, 0.30, 0.45, 0.65, 0.80])
age_dig = np.array([0.75, 0.70, 0.60, 0.50, 0.35, 0.25])  # younger = higher
digital_skill = np.clip(
    0.4 * edu_dig[education] + 0.4 * age_dig[age_groups] + 0.2 * np.random.beta(2, 3, NUM_WORKERS),
    0.0, 1.0
)

# ── 9. Region within Mississippi (0-4, rough economic regions) ───────────────
# 0: Delta, 1: North MS, 2: Central, 3: South/Gulf, 4: Metro (Jackson area)
REGION_PROBS = np.array([0.12, 0.18, 0.25, 0.20, 0.25])
region = np.random.choice(5, size=NUM_WORKERS, p=REGION_PROBS)

# ── Save as pickle files (AgentTorch population format) ──────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

def save_pickle(name, data):
    series = pd.Series(data.astype(np.float32))
    path = os.path.join(OUT_DIR, f"{name}.pickle")
    series.to_pickle(path)
    print(f"  Saved {name}.pickle  shape={series.shape}")

save_pickle("age", age_groups)
save_pickle("gender", gender)
save_pickle("education", education)
save_pickle("industry", industry)
save_pickle("ai_exposure", ai_exposure)
save_pickle("wage", wage)
save_pickle("employment_status", employment_status)
save_pickle("digital_skill", digital_skill)
save_pickle("region", region)

# Save industry names mapping
mapping = {
    "industry_names": INDUSTRY_NAMES,
    "age_groups": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "education_levels": ["< High School", "HS/GED", "Some College", "Bachelor's", "Graduate+"],
    "regions": ["Delta", "North MS", "Central", "South/Gulf", "Metro (Jackson)"],
    "employment_status": ["Employed", "Unemployed", "Underemployed"],
}
import json
with open(os.path.join(OUT_DIR, "population_mapping.json"), "w") as f:
    json.dump(mapping, f, indent=2)

print(f"\n✅  Generated {NUM_WORKERS:,} synthetic Mississippi workers → {OUT_DIR}")
