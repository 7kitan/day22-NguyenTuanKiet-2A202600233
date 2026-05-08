"""
Step 4 — Guardrails AI Validators
====================================
TASK:
  1. Build a PIIDetector validator that detects & redacts emails, phone
     numbers, SSNs, and credit card numbers
  2. Build a JSONFormatter validator that auto-repairs malformed JSON
  3. Wrap each with a Guard and test with sample inputs
  4. Run a full demo with 6 PII cases and 5 JSON cases

DELIVERABLE: All test cases pass (PII redacted, JSON repaired)

KEY CONCEPTS:
  - @register_validator — declares a custom validator class
  - Validator.validate() — implement the check + fix logic
  - OnFailAction.FIX — replace output instead of raising an error
  - Guard().use(MyValidator(on_fail=...)) — attach validator to guard
  - guard.validate(text) → ValidationOutcome
    .validation_passed — bool
    .validated_output   — the (possibly repaired) output string

⚠️  IMPORTANT: pass `on_fail` to the VALIDATOR constructor, NOT to Guard.use()
    WRONG: Guard().use(PIIDetector, on_fail=OnFailAction.FIX)  ← TypeError
    RIGHT: Guard().use(PIIDetector(on_fail=OnFailAction.FIX))  ← correct
"""

import re
import json
import sys
from pathlib import Path
from contextlib import contextmanager

# ── Logging utility for automatic evidence collection ─────────────────────────
class Tee:
    """Write to both console and file simultaneously."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

@contextmanager
def redirect_output(*file_paths):
    """Context manager to redirect stdout to multiple files."""
    files = [open(fp, "w") for fp in file_paths]
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, *files)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        for f in files:
            f.close()

Path("evidence").mkdir(exist_ok=True)

# ── 1. Imports ───────────────────────────────────────────────────────────────
from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
)
from guardrails.validator_base import OnFailAction


# ── 2. PII Detector Validator ─────────────────────────────────────────────────
@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).

    Patterns detected:
      - EMAIL: xxx@xxx.xxx
      - PHONE: (123) 456-7890 or 123-456-7890
      - SSN:   123-45-6789
      - CREDIT CARD: 1234 5678 9012 3456 (or dashes)
    """

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """
        Check value for PII; if found, redact and return FailResult
        with fix_value so the pipeline continues with redacted text.
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Redacted {len(found_pii)} PII items: {[p[0] for p in found_pii]}")
            return FailResult(error_message=f"PII detected: {[p[0] for p in found_pii]}", fix_value=redacted_text)
        return PassResult()


# ── 3. JSON Formatter Validator ───────────────────────────────────────────────
@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.

    Common repairs:
      - Strip markdown code fences (``` or ```json)
      - Replace single quotes with double quotes
      - Remove trailing commas before } or ]
      - Re-serialize with json.dumps for consistent formatting
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Attempt to repair a JSON string.
        """
        text = text.strip()

        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        text = text.replace("'", '"')

        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Try to parse value as JSON.
        If it fails, try _repair() then parse again.

        Return PassResult if already valid JSON.
        Return FailResult with fix_value if JSON needs repair.
        Return FailResult if JSON is unrecoverable.
        """
        try:
            parsed  = json.loads(value)
            repaired = json.dumps(parsed, indent=2)
            return PassResult()
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            repaired      = json.dumps(parsed, indent=2)
            print(f"  🔧 JSON repaired successfully")
            return FailResult(error_message="JSON repaired", fix_value=repaired)
        except json.JSONDecodeError as e:
            return FailResult(error_message=f"Invalid JSON after repair attempt: {e}")


# ── 4. PII Guard demo ────────────────────────────────────────────────────────
def demo_pii_guard():
    """
    Create a Guard with PIIDetector and test 6 sample texts:
      1. Text with an email address
      2. Text with a phone number
      3. Text with a Social Security Number
      4. Text with a credit card number
      5. Text with multiple PII types
      6. Clean text (no PII)
    """
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 5. JSON Guard demo ────────────────────────────────────────────────────────
def demo_json_guard():
    """
    Create a Guard with JSONFormatter and test 5 sample strings:
      1. Valid JSON (should pass as-is)
      2. JSON with markdown fences (should strip and pass)
      3. JSON with single quotes (should convert to double quotes)
      4. JSON with trailing comma (should remove and pass)
      5. Truly invalid JSON (should fail cleanly)
    """
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",        '{"name": "Alice", "age": 30}'),
        ("Markdown fences",   '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",     "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",    '{"key": "value",}'),
        ("Truly invalid",     "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:60]}")


# ── 6. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    print("\n📝 Running PII Detection Demo (saving to evidence/04_pii_demo_log.txt)...")
    with redirect_output("evidence/04_pii_demo_log.txt"):
        demo_pii_guard()
    print("✅ PII demo saved to evidence/04_pii_demo_log.txt")

    print("\n📝 Running JSON Formatting Demo (saving to evidence/04_json_demo_log.txt)...")
    with redirect_output("evidence/04_json_demo_log.txt"):
        demo_json_guard()
    print("✅ JSON demo saved to evidence/04_json_demo_log.txt")

    print("\n✅ Step 4 complete!")


if __name__ == "__main__":
    main()
