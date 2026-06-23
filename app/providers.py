"""Thin normalization layer over Anthropic and Groq SDKs."""

_DEFAULT_JUDGE_MODEL = {
    "anthropic": "claude-haiku-4-5",
    "groq": "llama-3.1-8b-instant",
}

_DEFAULT_MODEL = {
    "anthropic": "claude-opus-4-8",
    "groq": "llama-3.3-70b-versatile",
}


def default_model(provider: str) -> str:
    return _DEFAULT_MODEL.get(provider, _DEFAULT_MODEL["anthropic"])


def default_judge_model(provider: str) -> str:
    return _DEFAULT_JUDGE_MODEL.get(provider, _DEFAULT_JUDGE_MODEL["anthropic"])


async def call_async(
    provider: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        resp = await AsyncAnthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    if provider == "groq":
        from groq import AsyncGroq
        resp = await AsyncGroq().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown provider '{provider}'. Supported: anthropic, groq")


def call_sync(
    provider: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 10,
    temperature: float = 0.0,
) -> str:
    if provider == "anthropic":
        from anthropic import Anthropic
        resp = Anthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            # temperature omitted — deprecated in current Anthropic SDK
        )
        return resp.content[0].text

    if provider == "groq":
        from groq import Groq
        resp = Groq().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown provider '{provider}'. Supported: anthropic, groq")
