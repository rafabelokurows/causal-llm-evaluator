from pydantic import BaseModel, Field
from typing import Optional


class PromptVariant(BaseModel):
    id: str
    system_prompt: str
    user_template: str  # use {input} as placeholder


class ExperimentRequest(BaseModel):
    variants: list[PromptVariant]
    test_inputs: list[str]
    n_samples: int = 30
    scorer: str = "llm_judge"       # "rouge" | "llm_judge"
    provider: str = "anthropic"     # "anthropic" | "groq"
    model: Optional[str] = None     # None → provider default
    temperature: float = 0.7
    judge_provider: Optional[str] = None   # None → same as provider
    judge_model: Optional[str] = None      # None → provider default judge


class EffectEstimate(BaseModel):
    variant_id: str
    ate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    cohens_d: float
    significant: bool


class ExperimentResult(BaseModel):
    experiment_id: str
    control_id: str
    effects: list[EffectEstimate]
    n_per_variant: int
    scorer_used: str
    winner: Optional[str] = None
    interpretation: str


class CreatorReport(BaseModel):
    """Structured output for Primetag creator analysis."""
    audience_fit: float = Field(..., ge=0.0, le=1.0, description="Brand-audience alignment score")
    content_quality: float = Field(..., ge=0.0, le=1.0, description="Content consistency and production value")
    brand_safety: float = Field(..., ge=0.0, le=1.0, description="Risk score — 1 = fully safe")
    growth_signal: float = Field(..., ge=0.0, le=1.0, description="Momentum: follower growth, engagement trend")
    summary: str = Field(..., description="2–3 sentence narrative synthesis")
