#!/usr/bin/env python3
"""
Mississippi AI Job Impact Simulation — Main Runner

Simulates the impact of AI exposure on 1.2 million workers across Mississippi
over 12 quarterly time steps (3 years: 2025-2027).

Usage:
    python run_mississippi_sim.py

This script:
  1. Generates synthetic population data (if not already present)
  2. Loads the model config and population
  3. Runs the simulation using AgentTorch's Executor API
  4. Collects trajectory data and prints summary statistics
  5. Generates a comprehensive report
"""

import os
import sys
import json
import time
import numpy as np

# ── Step 0: Generate population data if needed ───────────────────────────────
POP_DIR = os.path.join(
    os.path.dirname(__file__), "agent_torch", "populations", "mississippi"
)
if not os.path.exists(os.path.join(POP_DIR, "age.pickle")):
    print("📊 Generating synthetic Mississippi workforce data...")
    # Run the data generation script
    gen_script = os.path.join(
        os.path.dirname(__file__),
        "agent_torch", "models", "mississippi_ai", "generate_data.py"
    )
    exec(open(gen_script).read())
    print()

# ── Step 1: Import AgentTorch components ─────────────────────────────────────
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
from agent_torch.models import mississippi_ai
from agent_torch.populations import mississippi

print("=" * 70)
print("  MISSISSIPPI AI JOB IMPACT SIMULATION")
print("  1.2 Million Workers | 12 Quarters (2025-2027)")
print("=" * 70)
print()

# ── Step 2: Setup simulation ─────────────────────────────────────────────────
print("🔧 Setting up simulation...")
t0 = time.time()

loader = LoadPopulation(mississippi)
simulation = Executor(model=mississippi_ai, pop_loader=loader)
runner = simulation.runner

print(f"   Population size: {loader.population_size:,} workers")
print(f"   Device: {runner.config['simulation_metadata']['device']}")

runner.init()
setup_time = time.time() - t0
print(f"   ✅ Initialization complete ({setup_time:.1f}s)")
print()

# ── Step 3: Run simulation ───────────────────────────────────────────────────
num_episodes = runner.config["simulation_metadata"]["num_episodes"]
num_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]

print(f"🚀 Running simulation: {num_episodes} episode(s), {num_steps} steps each")
print()

t1 = time.time()
for episode in range(num_episodes):
    runner.reset()
    for step in range(num_steps):
        runner.step(1)
        
        # Print quarterly progress
        state = runner.state
        if state is not None:
            q_label = f"Q{(step % 4) + 1} {2025 + step // 4}"
            
            # Get current stats
            emp_status = state.get("agents", {}).get("workers", {}).get("employment_status")
            job_impact = state.get("agents", {}).get("workers", {}).get("job_impact_status")
            
            if emp_status is not None and hasattr(emp_status, 'float'):
                n_unemployed = (emp_status == 1).float().sum().item()
                n_displaced = (job_impact == 3).float().sum().item()
                n_augmented = (job_impact == 1).float().sum().item()
                n_retrained = (job_impact == 4).float().sum().item()
                
                print(f"   {q_label}: Displaced={int(n_displaced):>7,} | "
                      f"Augmented={int(n_augmented):>7,} | "
                      f"Retrained={int(n_retrained):>7,} | "
                      f"Unemployed={int(n_unemployed):>7,}")

sim_time = time.time() - t1
print()
print(f"⏱  Simulation completed in {sim_time:.1f}s")
print()

# ── Step 4: Extract final results ────────────────────────────────────────────
print("📈 Extracting results...")

final_state = runner.state_trajectory[-1][-1] if runner.state_trajectory else runner.state
results = {}

try:
    env = final_state.get("environment", {})
    agents = final_state.get("agents", {}).get("workers", {})
    
    results["total_displaced_by_quarter"] = env.get("total_displaced", None)
    results["total_augmented_by_quarter"] = env.get("total_augmented", None)
    results["total_retrained_by_quarter"] = env.get("total_retrained", None)
    results["unemployment_rate_by_quarter"] = env.get("unemployment_rate", None)
    results["avg_wage_change_by_quarter"] = env.get("avg_wage_change", None)
    results["gdp_impact_by_quarter"] = env.get("gdp_impact", None)
    results["sector_displacement"] = env.get("sector_displacement", None)
    results["sector_augmentation"] = env.get("sector_augmentation", None)
    results["region_unemployment"] = env.get("region_unemployment", None)
    
    # Final agent-level stats
    if agents.get("job_impact_status") is not None:
        jis = agents["job_impact_status"]
        if hasattr(jis, 'float'):
            results["final_unaffected"] = int((jis == 0).float().sum().item())
            results["final_augmented"] = int((jis == 1).float().sum().item())
            results["final_at_risk"] = int((jis == 2).float().sum().item())
            results["final_displaced"] = int((jis == 3).float().sum().item())
            results["final_retrained"] = int((jis == 4).float().sum().item())
    
    if agents.get("wage") is not None and agents.get("employment_status") is not None:
        wage = agents["wage"]
        if hasattr(wage, 'mean'):
            results["final_mean_wage"] = float(wage.mean().item())
            results["final_median_wage"] = float(wage.median().item())

except Exception as e:
    print(f"   ⚠ Error extracting results: {e}")
    import traceback
    traceback.print_exc()

# ── Step 5: Print summary ────────────────────────────────────────────────────
print()
print("=" * 70)
print("  SIMULATION RESULTS SUMMARY")
print("=" * 70)

if results.get("final_displaced") is not None:
    total_pop = 1_200_000
    print(f"\n  Final Worker Status (after 3 years):")
    print(f"    Unaffected:  {results.get('final_unaffected', 0):>10,}  ({results.get('final_unaffected', 0)/total_pop*100:.1f}%)")
    print(f"    Augmented:   {results.get('final_augmented', 0):>10,}  ({results.get('final_augmented', 0)/total_pop*100:.1f}%)")
    print(f"    At Risk:     {results.get('final_at_risk', 0):>10,}  ({results.get('final_at_risk', 0)/total_pop*100:.1f}%)")
    print(f"    Displaced:   {results.get('final_displaced', 0):>10,}  ({results.get('final_displaced', 0)/total_pop*100:.1f}%)")
    print(f"    Retrained:   {results.get('final_retrained', 0):>10,}  ({results.get('final_retrained', 0)/total_pop*100:.1f}%)")

if results.get("sector_displacement") is not None:
    sd = results["sector_displacement"]
    sa = results["sector_augmentation"]
    INDUSTRY_NAMES = [
        "Agriculture/Forestry", "Manufacturing", "Retail Trade", "Healthcare",
        "Education", "Construction", "Transportation", "Accommodation/Food",
        "Professional/Technical", "Finance/Insurance", "Government", "Other Services"
    ]
    print(f"\n  Impact by Industry Sector:")
    print(f"    {'Sector':<25} {'Displaced':>10} {'Augmented':>10}")
    print(f"    {'-'*25} {'-'*10} {'-'*10}")
    if hasattr(sd, '__getitem__'):
        for i, name in enumerate(INDUSTRY_NAMES):
            d = int(sd[i].item()) if hasattr(sd[i], 'item') else int(sd[i])
            a = int(sa[i].item()) if hasattr(sa[i], 'item') else int(sa[i])
            print(f"    {name:<25} {d:>10,} {a:>10,}")

if results.get("region_unemployment") is not None:
    ru = results["region_unemployment"]
    REGIONS = ["Delta", "North MS", "Central", "South/Gulf", "Metro (Jackson)"]
    print(f"\n  Final Unemployment by Region:")
    if hasattr(ru, '__getitem__'):
        for i, name in enumerate(REGIONS):
            r = float(ru[i].item()) if hasattr(ru[i], 'item') else float(ru[i])
            print(f"    {name:<20} {r*100:.2f}%")

print()
print(f"  Total simulation time: {setup_time + sim_time:.1f}s")
print("=" * 70)

# ── Step 6: Save results to JSON for report generation ──────────────────────
results_path = os.path.join(os.path.dirname(__file__), "simulation_results.json")
serializable_results = {}
for k, v in results.items():
    if hasattr(v, 'tolist'):
        serializable_results[k] = v.tolist()
    elif hasattr(v, 'item'):
        serializable_results[k] = v.item()
    else:
        serializable_results[k] = v

with open(results_path, "w") as f:
    json.dump(serializable_results, f, indent=2)
print(f"\n💾 Results saved to {results_path}")
