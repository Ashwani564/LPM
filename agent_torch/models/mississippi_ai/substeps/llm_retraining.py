"""
Substep: LLM-Driven Retraining Decision

Uses AgentTorch's Archetype system with an Ollama backend to generate
behaviorally realistic retraining decisions for displaced workers.

Instead of parametric rules (retrain_prob = f(skill, education, age)),
the LLM receives a contextual prompt describing the worker's situation
and returns a probability (0-1) of whether the worker would retrain.

The @with_behavior decorator allows an Archetype/Behavior to be injected
at runtime via `envs.create(..., archetypes={"llm_retraining_decision": ...})`.

Key design: The Archetype groups workers into representative archetypes
(n_arch=7), so we make ~7 LLM calls per quarter, not 1.2M.
"""

import torch
import re

from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_by_path
from agent_torch.core.decorators import with_behavior


# ─── Prompt templates ─────────────────────────────────────────────────────────

AGENT_PROFILE = """You are simulating a Mississippi worker deciding whether to enroll in a retraining program after being displaced by AI automation.

Given information about a worker's profile and economic situation, respond with ONLY a single number between 0.0 and 1.0 representing the probability that this worker would choose to retrain. 

Consider these factors:
- Younger workers are more likely to retrain
- Higher education makes retraining easier
- Workers with digital skills adapt faster
- Workers in economically resilient regions have better retraining access
- Financial pressure (low savings, dependents) can both motivate and hinder retraining
- Cultural and social factors in Mississippi (community ties, family obligations)

Respond with ONLY a number between 0 and 1. Nothing else."""

USER_PROMPT_TEMPLATE = (
    "Worker profile:\n"
    "- Age group: {age_group}\n"
    "- Education: {education_level}\n"
    "- Previous industry: {industry}\n"
    "- Digital skill level: {digital_skill}/10\n"
    "- Previous annual wage: ${wage:,.0f}\n"
    "- Region: {region}\n"
    "- Months since displacement: {months_displaced}\n"
    "- Regional unemployment rate: {unemployment_pct:.1f}%\n"
    "- AI penetration in their sector: {sector_ai_pct:.0f}%\n\n"
    "What is the probability (0.0 to 1.0) this worker enrolls in retraining?"
)

AGE_GROUP_NAMES = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
EDUCATION_NAMES = ["Less than HS", "HS/GED", "Some College", "Bachelor's", "Graduate+"]
INDUSTRY_NAMES = [
    "Agriculture/Forestry", "Manufacturing", "Retail Trade", "Healthcare",
    "Education", "Construction", "Transportation", "Accommodation/Food",
    "Professional/Technical", "Finance/Insurance", "Government", "Other Services"
]
REGION_NAMES = ["Delta", "North Mississippi", "Central", "South/Gulf Coast", "Metro Jackson"]


@with_behavior
class LLMRetrainingDecision(SubstepAction):
    """
    Policy: Generate retraining probability for each displaced worker.

    When a Behavior is attached (via @with_behavior + envs.create), the LLM
    is queried through the Archetype system. When no behavior is attached,
    falls back to the parametric model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.num_agents = self.config["simulation_metadata"]["num_agents"]

    def _parametric_fallback(self, digital_skill, education, age):
        """Fallback parametric retraining probability (same as original model)."""
        retrain_rate = 0.30
        edu_bonus = torch.tensor(
            [0.0, 0.05, 0.10, 0.20, 0.30], dtype=torch.float32, device=self.device
        )
        age_penalty = torch.tensor(
            [0.0, 0.0, 0.05, 0.10, 0.20, 0.30], dtype=torch.float32, device=self.device
        )
        retrain_prob = torch.clamp(
            retrain_rate
            + digital_skill * 0.2
            + edu_bonus[education.squeeze(-1)].unsqueeze(-1)
            - age_penalty[age.squeeze(-1)].unsqueeze(-1),
            0.05, 0.80
        )
        return retrain_prob

    def forward(self, state, observation):
        input_variables = self.input_variables
        t = int(state["current_step"])

        # Retrieve worker state
        job_impact = get_by_path(state, re.split("/", input_variables["job_impact_status"]))
        digital_skill = get_by_path(state, re.split("/", input_variables["digital_skill"]))
        education = get_by_path(state, re.split("/", input_variables["education"])).long()
        age = get_by_path(state, re.split("/", input_variables["age"])).long()
        industry = get_by_path(state, re.split("/", input_variables["industry"])).long()
        region = get_by_path(state, re.split("/", input_variables["region"])).long()
        wage = get_by_path(state, re.split("/", input_variables["wage"]))
        unemployment_rate = get_by_path(state, re.split("/", input_variables["unemployment_rate"]))
        effective_exposure = get_by_path(state, re.split("/", input_variables["effective_ai_exposure"]))

        # Only displaced workers (job_impact == 3) are candidates
        displaced_mask = (job_impact == 3).float()

        if self.behavior is not None:
            # ── LLM Path: use Archetype system ───────────────────────────
            # The PromptManager fills {age} and {education} from mapping.json groupings.
            # Other template variables ({unemployment_rate}, {months_displaced})
            # are filled from scalar kwargs below.
            current_unemp = unemployment_rate[min(t, len(unemployment_rate) - 1)].item() * 100

            obs_kwargs = {
                "device": str(self.device),
                "current_memory_dir": ".agent_torch_memory",
                # Scalar kwargs matching prompt template variables not in mapping.json
                "unemployment_rate": f"{current_unemp:.1f}%",
                "months_displaced": str(t * 3),
            }

            # Call the behavior (Archetype handles grouping + LLM calls)
            llm_retrain_prob = self.behavior.sample(kwargs=obs_kwargs)

            # Ensure correct shape and apply displaced mask
            if llm_retrain_prob.dim() == 1:
                llm_retrain_prob = llm_retrain_prob.unsqueeze(-1)
            retrain_prob = llm_retrain_prob * displaced_mask

            print(f"    🧠 LLM retraining: {int(displaced_mask.sum().item())} displaced workers, "
                  f"mean prob={retrain_prob[displaced_mask.bool()].mean().item():.3f}")
        else:
            # ── Parametric fallback ──────────────────────────────────────
            retrain_prob = self._parametric_fallback(digital_skill, education, age)
            retrain_prob = retrain_prob * displaced_mask

        return {"retraining_probability": retrain_prob}
