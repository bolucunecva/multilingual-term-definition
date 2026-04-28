"""
Prompts for Experiment 1 — Definition Extraction.

Task: Given a technical term and a source document, extract the EXACT
definition span from the document (no paraphrasing or generation).
"""

SYSTEM_PROMPT = (
    "You are an expert at locating and extracting exact definition spans "
    "from scientific text. You always quote the source verbatim."
)

# ------------------------------------------------------------------
# Without guideline
# ------------------------------------------------------------------
WITHOUT_GUIDELINE = """\
You are given a scientific document and a specific technical term.
Your task is to find and extract the EXACT definition of that term as it appears in the document.

## Output Format (STRICT)
- Return ONLY a valid JSON array.
- Each element must be an object with exactly two keys:
  - "term": the given term
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

Term: "{term}"

Text:
\"\"\"
{text}
\"\"\"
"""

# ------------------------------------------------------------------
# With guideline
# ------------------------------------------------------------------
WITH_GUIDELINE = """\
You are given a scientific document and a specific technical term.
Your task is to find and extract the EXACT definition of that term as it appears in the document,
following the annotation guidelines below.

## Guidelines
{guideline}

## Output Format (STRICT)
- Return ONLY a valid JSON array.
- Each element must be an object with exactly two keys:
  - "term": the given term
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

Term: "{term}"

Text:
\"\"\"
{text}
\"\"\"
"""



def build_prompt(term: str, text: str, guideline: str | None, tokenizer=None) -> str:
    """
    Build the final prompt for a single (term, document) input.
    Wraps in a chat template if a tokenizer is supplied.
    """
    body = (
        WITH_GUIDELINE.format(guideline=guideline, term=term, text=text)
        if guideline
        else WITHOUT_GUIDELINE.format(term=term, text=text)
    )

    if tokenizer is None:
        return body

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": body},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
