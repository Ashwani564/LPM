# Mississippi AI Workforce Impact — Large Population Model

A **1.2-million-agent simulation** of AI's impact on Mississippi's labor force over 3 years (2025–2027), built on MIT Media Lab's [AgentTorch](https://github.com/AgentTorch/AgentTorch) framework.

The simulation supports two modes: a fast **parametric** mode (rule-based, ~1 second) and an **LLM-enhanced** mode that uses a local LLM via Ollama or MLX to generate behaviorally realistic retraining decisions.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installing Ollama](#installing-ollama)
  - [macOS](#macos)
  - [Windows](#windows)
  - [Linux](#linux)
  - [Pulling a Model](#pulling-a-model)
- [Project Setup](#project-setup)
- [Running the Simulation](#running-the-simulation)
  - [Parametric Mode (No LLM)](#1-parametric-mode-no-llm)
  - [LLM Mode with Ollama](#2-llm-mode-with-ollama)
  - [LLM Mode with MLX (Apple Silicon)](#3-llm-mode-with-mlx-apple-silicon-only)
  - [CLI Options](#cli-options)
- [Project Structure](#project-structure)
- [Simulation Output](#simulation-output)
- [Reports](#reports)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Ashwani564/LPM.git
cd LPM

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install agent-torch torch numpy pandas scipy omegaconf tqdm matplotlib dask pyarrow

# 4a. Run parametric simulation (no LLM needed)
python run_mississippi_sim.py

# 4b. Or run with LLM (requires Ollama — see below)
python run_mississippi_sim_llm.py --backend ollama --model llama3.2
```

---

## Installing Ollama

[Ollama](https://ollama.ai) runs open-source LLMs locally on your machine. It is required only for the LLM-enhanced simulation mode.

### macOS

**Option A — Download the app (recommended):**

1. Go to [https://ollama.ai/download](https://ollama.ai/download)
2. Click **"Download for macOS"**
3. Open the downloaded `Ollama.dmg`
4. Drag **Ollama** to your Applications folder
5. Launch Ollama from Applications — it will appear in your menu bar

**Option B — Homebrew:**

```bash
brew install ollama
```

**Verify installation:**

```bash
ollama --version
# ollama version 0.x.x
```

### Windows

**Option A — Download the installer (recommended):**

1. Go to [https://ollama.ai/download](https://ollama.ai/download)
2. Click **"Download for Windows"**
3. Run the downloaded `OllamaSetup.exe`
4. Follow the installer prompts
5. Ollama will run in the system tray after installation

**Option B — winget:**

```powershell
winget install Ollama.Ollama
```

**Verify installation** (open PowerShell or Command Prompt):

```powershell
ollama --version
```

> **Note (Windows):** Ollama runs as a background service automatically after installation. If you need GPU acceleration, make sure your NVIDIA drivers are up to date.

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Verify:**

```bash
ollama --version
```

### Pulling a Model

After installing Ollama, you need to download a model. Pull one of these recommended models:

```bash
# Small & fast (recommended for first run, ~2 GB)
ollama pull llama3.2

# Larger, higher quality (~4 GB)
ollama pull mistral

# Very small, used in the LLM results in this repo (~2 GB)
ollama pull ministral:3b
```

**Start the Ollama server** (if it isn't running already):

```bash
ollama serve
```

> On macOS (app install) and Windows, Ollama starts automatically. You only need `ollama serve` if you installed via CLI on Linux or Homebrew.

**Test that it works:**

```bash
ollama run llama3.2 "Hello, are you working?"
```

You should see a response from the model. Press `Ctrl+D` to exit.

---

## Project Setup

### Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **Git**
- **Ollama** (only for LLM mode — see above)

### Install Python Dependencies

```bash
# Clone the repo
git clone https://github.com/Ashwani564/LPM.git
cd LPM

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (Command Prompt)
# .venv\Scripts\Activate.ps1     # Windows (PowerShell)

# Install AgentTorch and dependencies
pip install agent-torch
pip install torch numpy pandas scipy omegaconf tqdm matplotlib dask pyarrow

# (Optional) For MLX backend on Apple Silicon Macs:
pip install mlx-lm
```

### Verify Setup

```bash
python -c "import agent_torch; print('AgentTorch OK')"
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"
```

---

## Running the Simulation

### 1. Parametric Mode (No LLM)

The fastest way to run. Uses rule-based equations for all worker decisions. No Ollama required.

```bash
python run_mississippi_sim.py
```

- **Runtime:** ~1–2 seconds
- **Output:** Prints quarterly stats to console + saves `simulation_results.json`

### 2. LLM Mode with Ollama

Uses a local LLM to generate retraining probabilities for displaced workers. Workers are clustered into 7 archetypes, so only ~7 LLM calls are made per quarter (84 total).

**Step 1 — Make sure Ollama is running with a model pulled:**

```bash
# Pull a model (one-time)
ollama pull llama3.2

# Ensure the server is running
ollama serve
```

**Step 2 — Run the simulation:**

```bash
# Default Ollama model (ministral:3b)
python run_mississippi_sim_llm.py --backend ollama

# Specify a different model
python run_mississippi_sim_llm.py --backend ollama --model llama3.2

# Use more archetypes for finer behavioral granularity
python run_mississippi_sim_llm.py --backend ollama --model mistral --n-arch 12

# Parametric fallback (no LLM calls, same as run_mississippi_sim.py)
python run_mississippi_sim_llm.py --no-llm
```

- **Runtime:** ~2–5 minutes (depends on model and hardware)
- **Output:** Prints quarterly stats with tqdm progress bar + saves `simulation_results_llm.json`

### 3. LLM Mode with MLX (Apple Silicon Only)

If you have a Mac with an M1/M2/M3/M4 chip, MLX runs the LLM natively on the Metal GPU — significantly faster than Ollama.

```bash
# Install MLX
pip install mlx-lm

# Run with MLX (default backend)
python run_mississippi_sim_llm.py --backend mlx

# Specify a different MLX model
python run_mississippi_sim_llm.py --backend mlx --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

- **Runtime:** ~30–50 seconds
- **Requires:** Apple Silicon Mac (M1+), `mlx-lm` package

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `mlx` | LLM backend: `ollama` or `mlx` |
| `--model` | auto | Model name (auto-selects per backend) |
| `--n-arch` | `7` | Number of archetypes for worker clustering |
| `--no-llm` | `false` | Disable LLM, use parametric rules instead |
| `--base-url` | `http://localhost:11434` | Ollama API endpoint |

---

## Project Structure

```
LPM/
├── run_mississippi_sim.py                   # Parametric simulation runner
├── run_mississippi_sim_llm.py               # LLM-enhanced simulation runner
├── simulation_results.json                  # Parametric run output
├── simulation_results_llm.json              # LLM run output (Ministral-3B)
│
├── agent_torch/
│   ├── core/llm/
│   │   ├── ollama_backend.py                # Ollama LLM backend
│   │   └── mlx_backend.py                   # MLX LLM backend (Apple Silicon)
│   │
│   ├── models/mississippi_ai/
│   │   ├── generate_data.py                 # Synthetic population generator
│   │   ├── simulator.py                     # Registry setup
│   │   ├── yamls/config.yaml                # Simulation configuration (491 lines)
│   │   └── substeps/
│   │       ├── ai_exposure.py               # AI exposure assessment substep
│   │       ├── job_transition.py            # Job transition substep (254 lines)
│   │       ├── llm_retraining.py            # LLM retraining decision substep
│   │       └── utils.py                     # Population data loader
│   │
│   └── populations/mississippi/
│       ├── age.pickle                       # 1.2M agent ages
│       ├── gender.pickle                    # 1.2M agent genders
│       ├── education.pickle                 # 1.2M education levels
│       ├── industry.pickle                  # 1.2M industry codes
│       ├── wage.pickle                      # 1.2M wages
│       ├── ai_exposure.pickle               # 1.2M AI exposure scores
│       ├── digital_skill.pickle             # 1.2M digital skill scores
│       ├── employment_status.pickle         # 1.2M employment statuses
│       ├── region.pickle                    # 1.2M region codes
│       ├── mapping.json                     # Archetype grouping definitions
│       └── population_mapping.json          # Human-readable label mappings
│
├── report_lpm.md                            # Architecture & parametric vs LLM comparison
├── report.md                                # Simulation results analysis
└── report-dataset.md                        # Dataset requirements for accuracy improvements
```

---

## Simulation Output

Both runners produce a JSON file with the following structure:

```json
{
  "total_displaced_by_quarter": [975, 0, 0, ...],
  "total_augmented_by_quarter": [1955, 0, 0, ...],
  "total_retrained_by_quarter": [1752, 0, 0, ...],
  "unemployment_rate_by_quarter": [0.084, 0.04, ...],
  "sector_displacement": [0, 3714, 451, ...],
  "sector_augmentation": [1, 26411, 7971, ...],
  "region_unemployment": [0.046, 0.045, 0.044, 0.044, 0.043],
  "final_unaffected": 753638,
  "final_augmented": 112941,
  "final_at_risk": 279373,
  "final_displaced": 4989,
  "final_retrained": 49059,
  "final_mean_wage": 41931.31,
  "final_median_wage": 39169.18
}
```

### Key Results (Parametric vs LLM)

| Metric | Parametric | LLM (Ministral-3B) |
|--------|-----------|---------------------|
| Workers Displaced | 4,989 (0.4%) | 38,554 (3.2%) |
| Workers Augmented | 112,941 (9.4%) | 112,694 (9.4%) |
| Workers Retrained | 49,059 (4.1%) | 15,823 (1.3%) |
| Final Mean Wage | $41,931 | $41,793 |

> The LLM produces dramatically lower retraining rates, reflecting real-world barriers like limited transit, family obligations, and sparse rural infrastructure in Mississippi.

---

## Reports

| Report | Description |
|--------|-------------|
| [`report_lpm.md`](report_lpm.md) | Full architecture documentation with parametric vs LLM comparison |
| [`report.md`](report.md) | Simulation results analysis and policy implications |
| [`report-dataset.md`](report-dataset.md) | Dataset requirements for upgrading from synthetic to real data |

---

## Troubleshooting

### Ollama connection refused

```
Error: Connection refused to http://localhost:11434
```

**Fix:** Make sure Ollama is running:
```bash
ollama serve
```

### Ollama model not found

```
Error: model 'llama3.2' not found
```

**Fix:** Pull the model first:
```bash
ollama pull llama3.2
```

### ModuleNotFoundError: agent_torch

**Fix:** Install AgentTorch:
```bash
pip install agent-torch
```

Or if working from the full AgentTorch source:
```bash
pip install -e .
```

### MLX not available (non-Apple hardware)

```
Error: No module named 'mlx'
```

**Fix:** MLX only works on Apple Silicon Macs. Use Ollama instead:
```bash
python run_mississippi_sim_llm.py --backend ollama
```

### Out of memory with large models

**Fix:** Use a smaller model:
```bash
ollama pull llama3.2       # ~2 GB (recommended)
# instead of
ollama pull llama3.1:70b   # ~40 GB
```

### Windows: `source .venv/bin/activate` doesn't work

**Fix:** Use the Windows activation command:
```powershell
.venv\Scripts\activate          # Command Prompt
.venv\Scripts\Activate.ps1      # PowerShell
```

---

## License

This simulation is built on [AgentTorch](https://github.com/AgentTorch/AgentTorch) (MIT License). All population data is synthetic.
