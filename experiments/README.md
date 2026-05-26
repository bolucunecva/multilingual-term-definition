Promps used in experiments:

EXP1 & EXP3:
```python
WITHOUT_GUIDELINE = """\
Extract all MACHINE LEARNING term-definition pairs from the text below.

A valid extraction MUST satisfy BOTH:
1. A clearly identifiable machine learning term/concept exists.
2. The text explicitly defines, explains, or describes that term.

Do NOT extract:
- isolated terms without definitions
- mentions of concepts without explanation
- generic statements
- examples without definitions
- partial or incomplete definitions
- inferred definitions
- background context that is not definitional

Only extract explicit term-definition relationships stated in the text.

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

If the text does NOT contain an explicit term-definition pair, return:
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

A valid extraction MUST satisfy BOTH:
1. A clearly identifiable machine learning term/concept exists.
2. The text explicitly defines, explains, or describes that term.

Do NOT extract:
- isolated terms without definitions
- mentions of concepts without explanation
- generic statements
- examples without definitions
- partial or incomplete definitions
- inferred definitions
- background context that is not definitional

Only extract explicit term-definition relationships stated in the text.

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

If the text does NOT contain an explicit term-definition pair, return:
[]

Text:
\"\"\"
{text}
\"\"\"
"""
```
