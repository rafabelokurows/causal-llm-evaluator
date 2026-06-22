import re
from app.providers import call_sync, default_judge_model

_JUDGE_SYSTEM = "You are an expert evaluator for creator analysis reports used by talent agencies."

_JUDGE_PROMPT = """Rate the quality of the following creator analysis on a scale of 1 to 10, where:
1-3 = Vague, generic, or missing key dimensions (audience, content, brand safety, growth)
4-6 = Partially covers dimensions but lacks specificity or actionable insight
7-9 = Covers all dimensions with specific, actionable recommendations
10 = Exceptional depth, precise metrics cited, clear investment recommendation

Input (creator profile):
{input}

Analysis to rate:
{output}

Reply with ONLY a number between 1 and 10. Nothing else."""


def score_rouge(output: str, reference: str) -> float:
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, output)["rougeL"].fmeasure


def score_llm_judge(
    output: str,
    input_text: str,
    provider: str = "anthropic",
    model: str | None = None,
) -> float:
    judge_model = model or default_judge_model(provider)
    raw = call_sync(
        provider=provider,
        model=judge_model,
        system=_JUDGE_SYSTEM,
        user=_JUDGE_PROMPT.format(input=input_text, output=output),
        max_tokens=10,
        temperature=0.0,
    )
    match = re.search(r"\d+\.?\d*", raw.strip())
    score = float(match.group()) if match else 5.0
    return min(max(score, 1.0), 10.0) / 10.0


def score_batch(
    records: list[dict],
    scorer: str,
    judge_provider: str = "anthropic",
    judge_model: str | None = None,
) -> list[dict]:
    for r in records:
        if scorer == "rouge":
            r["score"] = score_rouge(r["output"], r["input"])
        else:
            r["score"] = score_llm_judge(r["output"], r["input"], judge_provider, judge_model)
    return records
