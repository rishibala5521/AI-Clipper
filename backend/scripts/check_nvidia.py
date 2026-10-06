import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY", "").strip()
model_name = os.getenv("NVIDIA_MODEL", "").strip()
base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()

if not api_key:
    sys.exit("NVIDIA_API_KEY is missing. Add it to backend/.env")

client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0, max_retries=0)

# Part 1: which models can this key see?
KEYWORDS = ("instruct", "nemotron", "gpt-oss", "qwen", "mistral", "llama")
try:
    ids = sorted(model.id for model in client.models.list())
    print(f"{len(ids)} models listed. Ones with common text-model names:")
    for model_id in ids:
        if any(word in model_id.lower() for word in KEYWORDS):
            print("  ", model_id)
except Exception as exc:
    print(f"Could not list models: {type(exc).__name__}: {str(exc)[:200]}")
    if not model_name:
        sys.exit("Set NVIDIA_MODEL in .env (copy a model ID from build.nvidia.com) and run again.")

if not model_name:
    sys.exit(
        "\nSet NVIDIA_MODEL in .env to one ID from the list above "
        "(a plain 'instruct' model is best), then run this script again."
    )


# Part 2: does this model return valid JSON, and which request style works?
class Greeting(BaseModel):
    message: str = Field(description="A one-sentence friendly greeting.")
    word_count: int = Field(description="Number of words in the message.")


SCHEMA = Greeting.model_json_schema()
PROMPT = (
    "Write a one-sentence greeting for a new developer. "
    "Reply with JSON only, matching this JSON schema:\n" + json.dumps(SCHEMA)
)


def extract_json(text: str) -> str:
    """Drop any <think> block, then take everything from the first { to the last }."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return text[start : end + 1]


MODES = {
    "json_schema": {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Greeting", "schema": SCHEMA},
        }
    },
    "nvext_guided_json": {"extra_body": {"nvext": {"guided_json": SCHEMA}}},
    "guided_json": {"extra_body": {"guided_json": SCHEMA}},
    "prompt_only": {},
}

print(f"\nTesting model: {model_name}\n")
working = []
for mode, extra in MODES.items():
    started = time.monotonic()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.2,
            max_tokens=1000,
            **extra,
        )
        content = completion.choices[0].message.content or ""
        Greeting.model_validate_json(extract_json(content))
        try:
            json.loads(content)
            clean = "yes"
        except ValueError:
            clean = "no (needed cleanup)"
        usage = completion.usage
        seconds = time.monotonic() - started
        tokens = f"{usage.prompt_tokens}/{usage.completion_tokens}" if usage else "n/a"
        print(f"{mode:<18} OK    {seconds:5.1f}s  tokens in/out={tokens}  clean JSON: {clean}")
        working.append(mode)
    except Exception as exc:
        seconds = time.monotonic() - started
        status = getattr(exc, "status_code", None)
        print(f"{mode:<18} FAIL  {seconds:5.1f}s  {type(exc).__name__} {status or ''}: {str(exc)[:160]}")
        if status == 429:
            print("Rate limited. Stopping so we don't waste requests.")
            break

print("\nModes that worked:", ", ".join(working) if working else "none")