# Checks that Gemini is reachable, returns clean JSON, and records the exact model string the API served, the same way gpt-5.2 was pinned.
# The current SDK is google-genai, which clashes with the old google-generativeai package, so only one of the two should be installed.

import json
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# either key name works, though the SDK prefers GOOGLE_API_KEY when both are set
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
assert GEMINI_API_KEY, "No Gemini key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY)."

JUDGE_MODEL_REQUESTED = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

# a small round trip that forces JSON only, so it also checks Gemini will follow that instruction when judging
test_system = "You reply with ONLY a JSON object and nothing else."
test_user = 'Return exactly: {"score": 0.1, "reasoning": "connectivity test"}'

resp = client.models.generate_content(
    model=JUDGE_MODEL_REQUESTED,
    contents=test_user,
    config=types.GenerateContentConfig(
        system_instruction=test_system,
        max_output_tokens=256,
        temperature=0.0,
    ),
)

raw = resp.text
print("Raw response text:")
print(raw)
print()

# the field name has moved between SDK versions, so read the attribute first and fall back to the dict
resolved = getattr(resp, "model_version", None)
if not resolved:
    try:
        d = resp.to_json_dict()
        resolved = d.get("model_version") or d.get("modelVersion")
    except Exception:
        resolved = None

print("Record for the methodology:")
print(f"  requested model string: {JUDGE_MODEL_REQUESTED}")
print(f"  resolved model_version: {resolved if resolved else 'not exposed by the SDK, cite the requested string and the run date'}")


def _strip_fences(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    return t


try:
    parsed = json.loads(_strip_fences(raw))
    print(f"  JSON parse ok: {parsed}")
    print("\nGemini is reachable, returns parseable JSON, and the model string resolves. Ready for step 2.")
except Exception as e:
    print(f"  JSON parse failed: {e}")
    print("\nGemini replied but the JSON was not clean. Step 2 falls back to a fence strip and a regex, so keep an eye on the parsing there.")
