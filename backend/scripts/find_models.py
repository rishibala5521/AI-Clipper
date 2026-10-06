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
base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
if not api_key:
    sys.exit("NVIDIA_API_KEY is missing. Add it to backend/.env")

client = OpenAI(base_url=base_url, api_key=api_key, timeout=90.0, max_retries=0)

CANDIDATES = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-nano-3-30b-a3b",
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-large",
    "nv-mistralai/mistral-nemo-12b-instruct",
    "ai21labs/jamba-1.5-large-instruct",
    "databricks/dbrx-instruct",
    "microsoft/phi-3.5-moe-instruct",
    "ibm/granite-3.0-8b-instruct",
    "meta/llama-3.2-90b-vision-instruct",
]


class Greeting(BaseModel):
    message: str = Field(description="A one-sentence friendly greeting.")
    word_count: int = Field(description="Number of words in the message.")


PROMPT = (
    "Write a one-sentence greeting for a new developer. "
    "Reply with JSON only, matching this JSON schema:\n"
    + json.dumps(Greeting.model_json_schema())
)


def extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return text[start : end + 1]


good = []
for model in CANDIDATES:
    started = time.monotonic()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.2,
            max_tokens=1200,
        )
    except Exception as exc:
        seconds = time.monotonic() - started
        status = getattr(exc, "status_code", None)
        if status == 404:
            reason = "not available for your account (404)"
        else:
            reason = f"{type(exc).__name__} {status or ''}: {str(exc)[:100]}"
        print(f"{model:<44} FAIL  {seconds:5.1f}s  {reason}")
        if status == 429:
            print("Rate limited. Stopping; wait a minute and run again.")
            break
        time.sleep(1)
        continue

    seconds = time.monotonic() - started
    choice = completion.choices[0]
    content = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
    thinking = "yes" if ("<think>" in content or reasoning) else "no"
    try:
        Greeting.model_validate_json(extract_json(content))
    except ValueError:
        print(
            f"{model:<44} FAIL  {seconds:5.1f}s  reply was not valid JSON "
            f"(finish_reason={choice.finish_reason}) {content[:60]!r}"
        )
        time.sleep(1)
        continue

    try:
        json.loads(content)
        clean = "yes"
    except ValueError:
        clean = "no"
    usage = completion.usage
    tokens = f"{usage.prompt_tokens}/{usage.completion_tokens}" if usage else "n/a"
    print(
        f"{model:<44} OK    {seconds:5.1f}s  tokens in/out={tokens}  "
        f"clean JSON: {clean}  thinking text: {thinking}"
    )
    good.append((seconds, model))
    time.sleep(1)

print("\nModels that returned valid JSON, fastest first:")
for seconds, model in sorted(good):
    print(f"  {seconds:5.1f}s  {model}")
if not good:
    print("  none")