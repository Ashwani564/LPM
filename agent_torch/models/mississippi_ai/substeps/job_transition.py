"""
Substep 1: Job Transition

Each quarter, workers transition between states based on their effective AI exposure,
digital skills, regional economic resilience, and stochastic factors.

Worker states (job_impact_status):
  0 = Unaffected
  1 = Augmented (AI helps, wage grows)
  2 = At Risk (exposure rising, not yet displaced)
  3 = Displaced (lost job to AI automation)
  4 = Retrained (successfully transitioned to new role)

Follows AgentTorch SubstepTransition pattern.
"""

import torch
import torch.nn.functional as F
import re

from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_by_path


class JobTransition(SubstepTransition):
    """Transition: update worker employment, wages, and aggregate statistics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]

        # Economic parameters
        self.wage_growth_augmented = self.config["simulation_metadata"]["wage_growth_augmented"]
        self.wage_decline_displaced = self.config["simulation_metadata"]["wage_decline_displaced"]
        self.new_job_creation_rate = self.config["simulation_metadata"]["new_job_creation_rate"]

        # Region resilience
        self.region_resilience = torch.tensor(
            self.config["simulation_metadata"]["region_resilience"],
            dtype=torch.float32, device=self.device
        )

        # Sector adoption speeds
        self.sector_speed = torch.tensor(
            self.config["simulation_metadata"]["sector_adoption_speed"],
            dtype=torch.float32, device=self.device
        )

    def forward(self, state, action):
        t = int(state["current_step"])
        input_variables = self.input_variables

        # ── Check for LLM-generated retraining probabilities ─────────────
        llm_retrain_prob = None
        if action is not None:
            llm_retrain_prob = action.get("workers", {}).get("retraining_probability", None)

        # ── Retrieve state variables ─────────────────────────────────────
        effective_exposure = get_by_path(state, re.split("/", input_variables["effective_ai_exposure"]))
        employment_status = get_by_path(state, re.split("/", input_variables["employment_status"]))
        job_impact = get_by_path(state, re.split("/", input_variables["job_impact_status"]))
        wage = get_by_path(state, re.split("/", input_variables["wage"]))
        digital_skill = get_by_path(state, re.split("/", input_variables["digital_skill"]))
        industry = get_by_path(state, re.split("/", input_variables["industry"])).long()
        region = get_by_path(state, re.split("/", input_variables["region"])).long()
        age = get_by_path(state, re.split("/", input_variables["age"])).long()
        education = get_by_path(state, re.split("/", input_variables["education"])).long()

        # Environment trackers
        total_displaced = get_by_path(state, re.split("/", input_variables["total_displaced"]))
        total_augmented = get_by_path(state, re.split("/", input_variables["total_augmented"]))
        total_retrained = get_by_path(state, re.split("/", input_variables["total_retrained"]))
        unemployment_rate = get_by_path(state, re.split("/", input_variables["unemployment_rate"]))
        avg_wage_change = get_by_path(state, re.split("/", input_variables["avg_wage_change"]))
        gdp_impact = get_by_path(state, re.split("/", input_variables["gdp_impact"]))
        sector_displacement = get_by_path(state, re.split("/", input_variables["sector_displacement"]))
        sector_augmentation = get_by_path(state, re.split("/", input_variables["sector_augmentation"]))
        region_unemployment = get_by_path(state, re.split("/", input_variables["region_unemployment"]))

        # ── Thresholds ───────────────────────────────────────────────────
        auto_thresh = self.args.get("automation_threshold", 0.60)
        aug_thresh = self.args.get("augmentation_threshold", 0.35)
        retrain_rate = self.args.get("retraining_rate", 0.30)
        if isinstance(auto_thresh, torch.Tensor):
            auto_thresh = auto_thresh.item()
        if isinstance(aug_thresh, torch.Tensor):
            aug_thresh = aug_thresh.item()
        if isinstance(retrain_rate, torch.Tensor):
            retrain_rate = retrain_rate.item()

        # ── Region resilience per worker ─────────────────────────────────
        worker_resilience = self.region_resilience[region.squeeze(-1)].unsqueeze(-1)

        # ── Stochastic displacement/augmentation ─────────────────────────
        # Workers currently employed (status 0) and unaffected (impact 0)
        employed_mask = (employment_status == 0).float()
        unaffected_mask = (job_impact == 0).float()
        eligible = employed_mask * unaffected_mask

        # Displacement probability: high exposure + low digital skill + low resilience
        displacement_prob = torch.clamp(
            (effective_exposure - auto_thresh) * 2.0 * (1.0 - digital_skill) * (1.0 - worker_resilience),
            0.0, 0.85
        ) * eligible

        # Augmentation probability: moderate exposure + high digital skill
        augmentation_prob = torch.clamp(
            (effective_exposure - aug_thresh) * 1.5 * digital_skill * worker_resilience,
            0.0, 0.70
        ) * eligible * (1.0 - (effective_exposure > auto_thresh).float() * (1.0 - digital_skill))

        # Draw stochastic outcomes
        rand_disp = torch.rand_like(displacement_prob)
        rand_aug = torch.rand_like(augmentation_prob)

        newly_displaced = (rand_disp < displacement_prob).float() * eligible
        newly_augmented = (rand_aug < augmentation_prob).float() * eligible * (1.0 - newly_displaced)

        # At-risk: moderate exposure, not yet displaced or augmented
        at_risk = (
            (effective_exposure >= aug_thresh).float()
            * (effective_exposure < auto_thresh).float()
            * eligible
            * (1.0 - newly_displaced)
            * (1.0 - newly_augmented)
        )

        # ── Retraining: previously displaced workers can retrain ─────────
        displaced_mask = (job_impact == 3).float()

        if llm_retrain_prob is not None:
            # Use LLM-generated retraining probabilities (from Archetype)
            retrain_prob = llm_retrain_prob * displaced_mask
        else:
            # Parametric fallback
            # Retraining probability modulated by digital skill, education, age
            edu_retrain_bonus = torch.tensor(
                [0.0, 0.05, 0.10, 0.20, 0.30], dtype=torch.float32, device=self.device
            )
            age_retrain_penalty = torch.tensor(
                [0.0, 0.0, 0.05, 0.10, 0.20, 0.30], dtype=torch.float32, device=self.device
            )
            retrain_prob = torch.clamp(
                retrain_rate
                + digital_skill * 0.2
                + edu_retrain_bonus[education.squeeze(-1)].unsqueeze(-1)
                - age_retrain_penalty[age.squeeze(-1)].unsqueeze(-1),
                0.05, 0.80
            ) * displaced_mask

        rand_retrain = torch.rand_like(retrain_prob)
        newly_retrained = (rand_retrain < retrain_prob).float() * displaced_mask

        # ── Update job_impact_status ─────────────────────────────────────
        new_job_impact = job_impact.clone()
        new_job_impact[newly_augmented.bool()] = 1
        new_job_impact[at_risk.bool()] = 2
        new_job_impact[newly_displaced.bool()] = 3
        new_job_impact[newly_retrained.bool()] = 4

        # ── Update employment_status ─────────────────────────────────────
        new_employment = employment_status.clone()
        new_employment[newly_displaced.bool()] = 1  # unemployed
        new_employment[newly_retrained.bool()] = 0  # re-employed

        # ── Update wages ─────────────────────────────────────────────────
        new_wage = wage.clone()
        # Augmented workers get wage growth
        aug_mask = (new_job_impact == 1).float()
        new_wage = new_wage * (1.0 + self.wage_growth_augmented * aug_mask)
        # Displaced workers lose income
        disp_mask = newly_displaced
        new_wage = new_wage * (1.0 - self.wage_decline_displaced * disp_mask)
        # Retrained workers recover to 85% of original wage
        retrain_mask = newly_retrained
        # Approximate: retrained get a bump back
        new_wage = new_wage + retrain_mask * wage * 0.10

        # ── Update digital skill (slow improvement over time) ────────────
        new_digital_skill = digital_skill.clone()
        # Augmented workers gain digital skill
        new_digital_skill = torch.clamp(
            new_digital_skill + 0.02 * aug_mask + 0.05 * retrain_mask,
            0.0, 1.0
        )

        # ── Aggregate statistics ─────────────────────────────────────────
        new_total_displaced = total_displaced.clone()
        new_total_augmented = total_augmented.clone()
        new_total_retrained = total_retrained.clone()
        new_unemployment = unemployment_rate.clone()
        new_avg_wage = avg_wage_change.clone()
        new_gdp = gdp_impact.clone()

        n_displaced_now = newly_displaced.sum().item()
        n_augmented_now = newly_augmented.sum().item()
        n_retrained_now = newly_retrained.sum().item()

        if t < self.num_steps:
            new_total_displaced[t] = n_displaced_now
            new_total_augmented[t] = n_augmented_now
            new_total_retrained[t] = n_retrained_now

            # Unemployment rate
            total_unemployed = (new_employment == 1).float().sum().item()
            total_underemployed = (new_employment == 2).float().sum().item()
            new_unemployment[t] = (total_unemployed + 0.5 * total_underemployed) / self.num_agents

            # Average wage change
            wage_change_pct = ((new_wage - wage) / (wage + 1e-8)).mean().item() * 100
            new_avg_wage[t] = wage_change_pct

            # GDP impact (rough estimate: displaced wages lost - augmented wage gains)
            displaced_wage_loss = (newly_displaced * wage * self.wage_decline_displaced).sum().item()
            augmented_wage_gain = (newly_augmented * wage * self.wage_growth_augmented).sum().item()
            retrained_recovery = (newly_retrained * wage * 0.10).sum().item()
            # Convert to millions
            new_gdp[t] = (augmented_wage_gain + retrained_recovery - displaced_wage_loss) / 1e6

        # ── Sector-level displacement and augmentation ───────────────────
        new_sector_disp = sector_displacement.clone()
        new_sector_aug = sector_augmentation.clone()
        for s in range(12):
            s_mask = (industry.squeeze(-1) == s)
            new_sector_disp[s] += (newly_displaced.squeeze(-1)[s_mask]).sum()
            new_sector_aug[s] += (newly_augmented.squeeze(-1)[s_mask]).sum()

        # ── Region-level unemployment ────────────────────────────────────
        new_region_unemp = region_unemployment.clone()
        for r in range(5):
            r_mask = (region.squeeze(-1) == r)
            r_total = r_mask.float().sum().item()
            if r_total > 0:
                r_unemployed = ((new_employment.squeeze(-1) == 1).float() * r_mask.float()).sum().item()
                new_region_unemp[r] = r_unemployed / r_total

        return {
            "employment_status": new_employment,
            "job_impact_status": new_job_impact,
            "wage": new_wage,
            "digital_skill": new_digital_skill,
            "total_displaced": new_total_displaced,
            "total_augmented": new_total_augmented,
            "total_retrained": new_total_retrained,
            "unemployment_rate": new_unemployment,
            "avg_wage_change": new_avg_wage,
            "gdp_impact": new_gdp,
            "sector_displacement": new_sector_disp,
            "sector_augmentation": new_sector_aug,
            "region_unemployment": new_region_unemp,
        }
