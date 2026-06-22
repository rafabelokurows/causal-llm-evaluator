import numpy as np
import pandas as pd
from scipy import stats
from app.schemas import EffectEstimate


def estimate_effects(records: list[dict], control_id: str) -> list[EffectEstimate]:
    df = pd.DataFrame(records)
    control_scores = df[df["variant_id"] == control_id]["score"].values
    effects = []

    for variant_id in df["variant_id"].unique():
        if variant_id == control_id:
            continue

        treatment_scores = df[df["variant_id"] == variant_id]["score"].values

        ate = treatment_scores.mean() - control_scores.mean()

        boot_ates = [
            np.random.choice(treatment_scores, size=len(treatment_scores), replace=True).mean()
            - np.random.choice(control_scores, size=len(control_scores), replace=True).mean()
            for _ in range(2000)
        ]
        ci_lower, ci_upper = np.percentile(boot_ates, [2.5, 97.5])

        _, p_value = stats.ttest_ind(treatment_scores, control_scores, equal_var=False)

        pooled_std = np.sqrt((treatment_scores.std() ** 2 + control_scores.std() ** 2) / 2)
        cohens_d = ate / pooled_std if pooled_std > 0 else 0.0

        effects.append(
            EffectEstimate(
                variant_id=variant_id,
                ate=round(float(ate), 4),
                ci_lower=round(float(ci_lower), 4),
                ci_upper=round(float(ci_upper), 4),
                p_value=round(float(p_value), 4),
                cohens_d=round(float(cohens_d), 4),
                significant=bool(p_value < 0.05),
            )
        )

    return effects


def interpret_results(effects: list[EffectEstimate], control_id: str) -> tuple[str | None, str]:
    significant = [e for e in effects if e.significant]

    if not significant:
        return None, (
            f"No statistically significant differences vs '{control_id}'. "
            "Variants perform similarly, or increase n_samples (>=50 recommended)."
        )

    winner = max(significant, key=lambda e: e.ate)
    magnitude = "Large" if abs(winner.cohens_d) > 0.8 else "Medium" if abs(winner.cohens_d) > 0.5 else "Small"
    return winner.variant_id, (
        f"'{winner.variant_id}' best variant "
        f"(ATE={winner.ate:+.3f}, Cohen's d={winner.cohens_d:.2f}, p={winner.p_value:.3f}). "
        f"95% CI: [{winner.ci_lower:.3f}, {winner.ci_upper:.3f}]. {magnitude} practical effect."
    )
