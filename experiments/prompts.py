SYSTEM_PROMPT = "You are a scientific domain expert in extracting term-definition pairs from scientific text."

WITHOUT_GUIDELINE = """\
Extract all MACHINE LEARNING term-definition pairs from the text below.

## Output Format (STRICT)
- Return ONLY a valid JSON array.
- Each element must be an object with exactly two keys:
  - "term": the term or concept being defined
  - "definition": the full definition or explanation of that term
- Do NOT include any extra text, comments, or explanations.
- Do NOT include trailing commas.
- Ensure the JSON is valid and parseable.

## Example Output
[
  {{
    "term": "Photosynthesis",
    "definition": "The process by which green plants use sunlight to synthesize food from carbon dioxide and water."
  }}
]

If no term-definition pairs are found, return:
[]

Text:
\"\"\"
{text}
\"\"\"
"""

WITH_GUIDELINE = """\
Extract all MACHINE LEARNING term-definition pairs from the text below, following the guidelines provided.

## Guidelines
{guideline}

## Output Format (STRICT)
- Return ONLY a valid JSON array.
- Each element must be an object with exactly two keys:
  - "term": the term or concept being defined
  - "definition": the full definition or explanation of that term
- Do NOT include any extra text, comments, or explanations.
- Do NOT include trailing commas.
- Ensure the JSON is valid and parseable.

## Example Output
[
  {{
    "term": "Photosynthesis",
    "definition": "The process by which green plants use sunlight to synthesize food from carbon dioxide and water."
  }}
]

If no term-definition pairs are found, return:
[]

Text:
\"\"\"
{text}
\"\"\"
"""


def build_prompt(text: str, guideline: str | None, tokenizer=None) -> str:
    """
    Build the final prompt string.
    If a tokenizer with a chat template is provided, wraps in chat format.
    """
    body = (
        WITH_GUIDELINE.format(guideline=guideline, text=text)
        if guideline
        else WITHOUT_GUIDELINE.format(text=text)
    )

    if tokenizer is None:
        return body

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
