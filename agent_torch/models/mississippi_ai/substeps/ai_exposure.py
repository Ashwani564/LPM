"""
Substep 0: AI Exposure Assessment

Computes each worker's effective AI exposure based on their baseline exposure score,
industry sector AI adoption speed, education level, digital skills, age, and the
evolving economy-wide AI penetration index.

Follows AgentTorch SubstepAction + SubstepTransition pattern.
"""

import torch
import torch.nn.functional as F
import re

from agent_torch.core.substep import SubstepAction, SubstepTransition
from agent_torch.core.helpers import get_by_path


class AssessAIExposure(SubstepAction):
    """Policy: compute each worker's effective AI exposure this quarter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.num_agents = self.config["simulation_metadata"]["num_agents"]

        # Sector-specific adoption speed multipliers
        self.sector_speed = torch.tensor(
            self.config["simulation_metadata"]["sector_adoption_speed"],
            dtype=torch.float32, device=self.device
        )

    def forward(self, state, observation):
        t = int(state["current_step"])
        input_variables = self.input_variables

        # Retrieve worker attributes
        base_ai_exposure = get_by_path(state, re.split("/", input_variables["ai_exposure"]))
        industry = get_by_path(state, re.split("/", input_variables["industry"])).long()
        education = get_by_path(state, re.split("/", input_variables["education"])).long()
        digital_skill = get_by_path(state, re.split("/", input_variables["digital_skill"]))
        age = get_by_path(state, re.split("/", input_variables["age"])).long()

        # AI adoption acceleration: compound quarterly growth
        adoption_rate = self.args.get("adoption_acceleration", 0.08)
        if isinstance(adoption_rate, torch.Tensor):
            adoption_rate = adoption_rate.item()
        time_multiplier = (1.0 + adoption_rate) ** t

        # Industry-specific adoption speed
        industry_speed = self.sector_speed[industry.squeeze(-1)].unsqueeze(-1)

        # Education modifier: higher ed → more exposed to knowledge-work AI
        edu_modifier = torch.tensor(
            [-0.05, -0.02, 0.0, 0.05, 0.10],
            dtype=torch.float32, device=self.device
        )
        edu_effect = edu_modifier[education.squeeze(-1)].unsqueeze(-1)

        # Age modifier: younger workers adapt faster but also get more new AI tools
        age_adapt = torch.tensor(
            [1.10, 1.05, 1.00, 0.95, 0.85, 0.75],
            dtype=torch.float32, device=self.device
        )
        age_effect = age_adapt[age.squeeze(-1)].unsqueeze(-1)

        # Digital skill acts as a buffer (high skill → exposure shifts toward augmentation)
        # but we still compute raw effective exposure here
        effective_exposure = base_ai_exposure * time_multiplier * industry_speed * age_effect + edu_effect
        effective_exposure = torch.clamp(effective_exposure, 0.0, 1.0)

        return {"effective_ai_exposure": effective_exposure}


class UpdateExposure(SubstepTransition):
    """Transition: write effective exposure back to state and update AI penetration index."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.config["simulation_metadata"]["device"])

    def forward(self, state, action):
        input_variables = self.input_variables

        # The action from the policy carries the new effective exposure
        new_exposure = action["workers"]["effective_ai_exposure"]

        # Update AI penetration index (economy-wide measure)
        current_penetration = get_by_path(
            state, re.split("/", input_variables["ai_penetration_index"])
        )
        growth = self.args.get("base_penetration_growth", 0.03)
        if isinstance(growth, torch.Tensor):
            growth = growth.item()

        new_penetration = torch.clamp(current_penetration + growth, 0.0, 1.0)

        return {
            "effective_ai_exposure": new_exposure,
            "ai_penetration_index": new_penetration,
        }
