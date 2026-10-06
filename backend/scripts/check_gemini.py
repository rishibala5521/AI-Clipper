import os
import sys

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model_name = os.getenv("GEMINI_MODEL", "").strip()

if not api_key:
    sys.exit("GEMINI_API_KEY is missing. Add it to backend/.env")

client = genai.Client(api_key=api_key)

# Part 1: which models can this key use?
try:
    print("Models available to your key (names containing 'flash'):")
    for model in client.models.list():
        if "flash" in model.name.lower():
            print("  ", model.name)
except Exception as exc:
    sys.exit(f"Could not list models: {type(exc).__name__}: {exc}")

if not model_name:
    sys.exit(
        "\nSet GEMINI_MODEL in .env to one name from the list above "
        "(without the 'models/' prefix), then run this script again."
    )


# Part 2: one tiny structured-output call.
class Greeting(BaseModel):
    message: str = Field(description="A one-sentence friendly greeting.")
    word_count: int = Field(description="Number of words in the message.")


try:
    interaction = client.interactions.create(
        model=model_name,
        input="Write a one-sentence greeting for a new developer.",
        response_format=[
            {
                "type": "text",
                "mime_type": "application/json",
                "schema": Greeting.model_json_schema(),
            }
        ],
    )
except Exception as exc:
    sys.exit(f"Gemini call failed: {type(exc).__name__}: {exc}")

print("\nRaw output:", interaction.output_text)
parsed = Greeting.model_validate_json(interaction.output_text)
print("Parsed OK:", parsed)
print("Usage:", getattr(interaction, "usage", "(not available on this object)"))