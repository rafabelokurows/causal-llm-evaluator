import asyncio
import random
from app.schemas import ExperimentRequest, PromptVariant
from app.providers import call_async, default_model


async def call_llm(
    variant: PromptVariant,
    input_text: str,
    provider: str,
    model: str,
    temperature: float,
) -> dict:
    output = await call_async(
        provider=provider,
        model=model,
        system=variant.system_prompt,
        user=variant.user_template.format(input=input_text),
        temperature=temperature,
    )
    return {
        "variant_id": variant.id,
        "input": input_text,
        "output": output,
        "input_length": len(input_text),
        "output_length": len(output),
    }


async def run_experiment(req: ExperimentRequest) -> list[dict]:
    model = req.model or default_model(req.provider)
    tasks = []
    for input_text in req.test_inputs:
        # Balanced assignment: each variant gets equal reps per input
        reps_per_variant = max(1, req.n_reps_per_variant // len(req.variants))
        assignment = req.variants * reps_per_variant
        random.shuffle(assignment)
        for variant in assignment:
            tasks.append(call_llm(variant, input_text, req.provider, model, req.temperature))
    results = await asyncio.gather(*tasks)
    return list(results)
