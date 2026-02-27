#!/usr/bin/env python3
"""
Mississippi AI Job Impact Simulation — LLM-Enhanced Runner (Ollama)

This runner integrates the AgentTorch Archetype system with a local Ollama LLM
to generate behaviorally realistic retraining decisions for displaced workers.

Instead of parametric rules, the LLM receives worker profiles and generates
context-sensitive retraining probabilities. The Archetype system clusters
1.2M workers into ~7 representative groups, so only ~7 LLM calls per quarter
are needed (84 total for 12 quarters).

Prerequisites:
    1. Ollama installed: https://ollama.ai
    2. A model pulled: `ollama pull llama3.2`
    3. Ollama server running: `ollama serve`

Usage:
    python run_mississippi_sim_llm.py
    python run_mississippi_sim_llm.py --model mistral
    python run_mississippi_sim_llm.py --no-llm   # parametric fallback
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

# ── Parse arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Mississippi AI Sim with LLM (Ollama or MLX)")
parser.add_argument("--backend", default="mlx", choices=["ollama", "mlx"], help="LLM backend (default: mlx)")
parser.add_argument("--model", default=None, help="Model name (default: auto-select per backend)")
parser.add_argument("--n-arch", type=int, default=7, help="Number of archetypes (default: 7)")
parser.add_argument("--no-llm", action="store_true", help="Disable LLM, use parametric fallback")
parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama API URL")
args = parser.parse_args()

# Auto-select model if not specified
if args.model is None:
    if args.backend == "mlx":
        args.model = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    else:
        args.model = "ministral-3:8b"

# ── Step 0: Generate population data if needed ───────────────────────────────
POP_DIR = os.path.join(
    os.path.dirname(__file__), "agent_torch", "populations", "mississippi"
)
if not os.path.exists(os.path.join(POP_DIR, "age.pickle")):
    print("📊 Generating synthetic Mississippi workforce data...")
    gen_script = os.path.join(
        os.path.dirname(__file__),
        "agent_torch", "models", "mississippi_ai", "generate_data.py"
    )
    exec(open(gen_script).read())
    print()

# ── Step 1: Import components ────────────────────────────────────────────────
from agent_torch.core.environment import envs
from agent_torch.core.llm.archetype import Archetype
from agent_torch.models import mississippi_ai
from agent_torch.populations import mississippi

print("=" * 70)
print("  MISSISSIPPI AI JOB IMPACT SIMULATION")
print(f"  🧠 LLM-Enhanced with {args.backend.upper()} Archetypes")
print("=" * 70)
print()

# ── Step 2: Configure LLM + Archetype ───────────────────────────────────────
archetypes = None

if not args.no_llm:
    # System prompt: instructs the LLM on its role
    agent_profile = (
        "You are simulating a Mississippi worker deciding whether to enroll "
        "in a retraining program after being displaced by AI automation.\n\n"
        "Given information about a worker's profile and economic situation, "
        "respond with ONLY a single number between 0.0 and 1.0 representing "
        "the probability that this worker would choose to retrain.\n\n"
        "Consider:\n"
        "- Younger workers are more likely to retrain\n"
        "- Higher education makes retraining easier and more attractive\n"
        "- Workers with digital skills adapt faster to new roles\n"
        "- Workers in economically resilient regions have better retraining access\n"
        "- Financial pressure from job loss can motivate retraining\n"
        "- Mississippi-specific factors: strong community ties, family obligations, "
        "limited public transit (affects program access)\n\n"
        "Respond with ONLY a number between 0 and 1. Nothing else."
    )

    # User prompt template — {age} and {education} are grouped via mapping.json
    # Other variables are filled as scalar kwargs per-quarter
    user_prompt = (
        "Worker profile:\n"
        "- Age group: {age}\n"
        "- Education: {education}\n"
        "- Current regional unemployment rate: {unemployment_rate}\n"
        "- Months since displacement: {months_displaced}\n\n"
        "What is the probability (0.0 to 1.0) this worker enrolls in retraining?"
    )

    # ── Create the LLM backend ───────────────────────────────────────────
    if args.backend == "mlx":
        from agent_torch.core.llm.mlx_backend import MLXLLM

        print(f"🍎 Setting up MLX backend (Apple Silicon native)...")
        print(f"   Model: {args.model}")
        print(f"   Archetypes: {args.n_arch}")

        llm_backend = MLXLLM(
            model=args.model,
            agent_profile=agent_profile,
            max_tokens=8,
        )
        llm_backend.initialize_llm()

    else:  # ollama
        from agent_torch.core.llm.ollama_backend import OllamaLLM

        print(f"🔌 Connecting to Ollama ({args.base_url})...")
        print(f"   Model: {args.model}")
        print(f"   Archetypes: {args.n_arch}")

        llm_backend = OllamaLLM(
            model=args.model,
            agent_profile=agent_profile,
            base_url=args.base_url,
            temperature=0.3,
        )
        llm_backend.initialize_llm()

    # Create the Archetype using AgentTorch's unified API
    archetype = Archetype(
        prompt=user_prompt,
        llm=llm_backend,
        n_arch=args.n_arch,
    )

    # Build archetype mapping: substep_name → archetype
    archetypes_config = archetype.llm(llm=llm_backend, user_prompt=user_prompt)

    archetypes = {
        "llm_retraining_decision": archetypes_config,
    }

    print(f"   ✅ Archetype configured ({args.backend.upper()})")
    print()
else:
    print("⚙ Running in parametric mode (no LLM)")
    print()

# ── Step 3: Create environment with Archetypes ──────────────────────────────
print("🔧 Setting up simulation...")
t0 = time.time()

runner = envs.create(
    model=mississippi_ai,
    population=mississippi,
    archetypes=archetypes,
)

num_agents = runner.config["simulation_metadata"]["num_agents"]
print(f"   Population size: {num_agents:,} workers")
print(f"   Device: {runner.config['simulation_metadata']['device']}")
if not args.no_llm:
    print(f"   LLM Backend: {args.backend.upper()} ({args.model})")
    print(f"   Estimated LLM calls: ~{args.n_arch * 12} (7 archetypes × 12 quarters)")

setup_time = time.time() - t0
print(f"   ✅ Initialization complete ({setup_time:.1f}s)")
print()

# ── Step 4: Run simulation ───────────────────────────────────────────────────
num_episodes = runner.config["simulation_metadata"]["num_episodes"]
num_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]

print(f"🚀 Running simulation: {num_episodes} episode(s), {num_steps} steps each")
if not args.no_llm:
    print(f"   ⏳ Each step queries {args.backend.upper()} for retraining decisions...")
print()

t1 = time.time()
for episode in range(num_episodes):
    runner.reset()
    pbar = tqdm(range(num_steps), desc="🚀 Simulating", unit="quarter",
                bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for step in pbar:
        step_t0 = time.time()

        # Suppress noisy prints from Behavior.sample() during step
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            runner.step(1)
        step_time = time.time() - step_t0

        # Print quarterly progress
        state = runner.state
        if state is not None:
            q_label = f"Q{(step % 4) + 1} {2025 + step // 4}"

            emp_status = state.get("agents", {}).get("workers", {}).get("employment_status")
            job_impact = state.get("agents", {}).get("workers", {}).get("job_impact_status")

            if emp_status is not None and hasattr(emp_status, 'float'):
                n_unemployed = (emp_status == 1).float().sum().item()
                n_displaced = (job_impact == 3).float().sum().item()
                n_augmented = (job_impact == 1).float().sum().item()
                n_retrained = (job_impact == 4).float().sum().item()

                llm_tag = " 🧠" if not args.no_llm else ""
                pbar.set_postfix_str(
                    f"{q_label} D={int(n_displaced):,} A={int(n_augmented):,} "
                    f"R={int(n_retrained):,} ({step_time:.1f}s/q){llm_tag}"
                )
    pbar.close()

sim_time = time.time() - t1
print()
print(f"⏱  Simulation completed in {sim_time:.1f}s")
if not args.no_llm:
    print(f"   (includes ~{args.n_arch * num_steps} {args.backend.upper()} LLM calls)")
print()

# ── Step 5: Extract final results ────────────────────────────────────────────
print("📈 Extracting results...")

final_state = runner.state_trajectory[-1][-1] if hasattr(runner, 'state_trajectory') and runner.state_trajectory else runner.state
results = {}

try:
    env = final_state.get("environment", {})
    agents = final_state.get("agents", {}).get("workers", {})

    results["total_displaced_by_quarter"] = env.get("total_displaced", None)
    results["total_augmented_by_quarter"] = env.get("total_augmented", None)
    results["total_retrained_by_quarter"] = env.get("total_retrained", None)
    results["unemployment_rate_by_quarter"] = env.get("unemployment_rate", None)
    results["sector_displacement"] = env.get("sector_displacement", None)
    results["sector_augmentation"] = env.get("sector_augmentation", None)
    results["region_unemployment"] = env.get("region_unemployment", None)

    if agents.get("job_impact_status") is not None:
        jis = agents["job_impact_status"]
        if hasattr(jis, 'float'):
            results["final_unaffected"] = int((jis == 0).float().sum().item())
            results["final_augmented"] = int((jis == 1).float().sum().item())
            results["final_at_risk"] = int((jis == 2).float().sum().item())
            results["final_displaced"] = int((jis == 3).float().sum().item())
            results["final_retrained"] = int((jis == 4).float().sum().item())

    if agents.get("wage") is not None:
        wage = agents["wage"]
        if hasattr(wage, 'mean'):
            results["final_mean_wage"] = float(wage.mean().item())
            results["final_median_wage"] = float(wage.median().item())

except Exception as e:
    print(f"   ⚠ Error extracting results: {e}")
    import traceback
    traceback.print_exc()

# ── Step 6: Print summary ────────────────────────────────────────────────────
print()
print("=" * 70)
print("  SIMULATION RESULTS SUMMARY")
if not args.no_llm:
    print(f"  🧠 LLM-Enhanced ({args.backend.upper()} {args.model}, {args.n_arch} archetypes)")
else:
    print("  ⚙ Parametric Mode")
print("=" * 70)

total_pop = num_agents

if results.get("final_displaced") is not None:
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

# ── Step 7: Save results ────────────────────────────────────────────────────
results_path = os.path.join(os.path.dirname(__file__), "simulation_results_llm.json")
serializable_results = {"mode": "llm" if not args.no_llm else "parametric"}
if not args.no_llm:
    serializable_results["llm_backend"] = args.backend
    serializable_results["llm_model"] = args.model
    serializable_results["n_archetypes"] = args.n_arch

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
