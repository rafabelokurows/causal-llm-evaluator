import logging
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
from app.schemas import ExperimentRequest, ExperimentResult
from app.experiment import run_experiment
from app.scoring import score_batch
from app.causal import estimate_effects, interpret_results
from app.store import save_result, get_result, list_results

app = FastAPI(title="Causal LLM Evaluator", version="1.0")


@app.post("/experiment", response_model=ExperimentResult)
async def run(req: ExperimentRequest):
    exp_id = str(uuid.uuid4())[:8]
    control_id = req.variants[0].id
    logging.info(
        "starting experiment=%s provider=%s model=%s variants=%s n_inputs=%d n_reps_per_variant=%d",
        exp_id,
        req.provider,
        req.model or "default",
        [v.id for v in req.variants],
        len(req.test_inputs),
        req.n_reps_per_variant,
    )

    records = await run_experiment(req)
    judge_provider = req.judge_provider or req.provider
    scored = score_batch(records, req.scorer, judge_provider, req.judge_model)
    effects = estimate_effects(scored, control_id)
    winner, interpretation = interpret_results(effects, control_id)

    result = ExperimentResult(
        experiment_id=exp_id,
        control_id=control_id,
        effects=effects,
        n_per_variant=req.n_reps_per_variant,
        scorer_used=req.scorer,
        winner=winner,
        interpretation=interpretation,
    )
    save_result(exp_id, result)
    return result


@app.get("/results/{exp_id}", response_model=ExperimentResult)
def get_results(exp_id: str):
    return get_result(exp_id)


@app.get("/results", response_model=list[dict])
def all_results():
    return list_results()


@app.get("/health")
def health():
    return {"status": "ok"}
