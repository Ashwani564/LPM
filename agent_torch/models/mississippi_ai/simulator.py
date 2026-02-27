"""
Mississippi AI Job Impact Model — Simulator Module

Sets up the Registry with all substeps and initialization helpers,
following the same pattern as the COVID model's simulator.py.
"""

from agent_torch.core import Runner, Registry


def get_registry():
    reg = Registry()

    # ── Substeps ──────────────────────────────────────────────────────────
    from .substeps.ai_exposure import AssessAIExposure, UpdateExposure
    from .substeps.job_transition import JobTransition
    from .substeps.llm_retraining import LLMRetrainingDecision

    reg.register(AssessAIExposure, "assess_ai_exposure", key="policy")
    reg.register(UpdateExposure, "update_exposure", key="transition")
    reg.register(JobTransition, "job_transition", key="transition")
    reg.register(LLMRetrainingDecision, "llm_retraining_decision", key="policy")

    # ── Initialization helpers ────────────────────────────────────────────
    from .substeps.utils import load_population_attribute

    reg.register(load_population_attribute, "load_population_attribute", key="initialization")

    return reg


def get_runner(config, registry):
    runner = Runner(config, registry)
    return runner


# Module-level registry so Executor can find it via module.registry
registry = get_registry()
