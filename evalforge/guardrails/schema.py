"""JSON and Pydantic schema validation guardrail with auto-repair capabilities."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional, Type
from pydantic import BaseModel, ValidationError

from evalforge.guardrails.base import (
    BaseGuardrail,
    GuardrailAction,
    GuardrailResult,
    GuardrailSeverity,
    GuardrailViolation,
)


class SchemaValidationGuardrail(BaseGuardrail):
    """Validates that LLM text output conforms to JSON structure and expected schema."""

    name: str = "schema_validation"
    description: str = "Ensures LLM output conforms to valid JSON and target Pydantic or dictionary schema."

    def __init__(
        self,
        schema_model: Optional[Type[BaseModel]] = None,
        required_keys: Optional[list[str]] = None,
        auto_repair: bool = True,
        action: GuardrailAction = GuardrailAction.BLOCK,
    ):
        self.schema_model = schema_model
        self.required_keys = required_keys or []
        self.auto_repair = auto_repair
        self.action = action

    def _attempt_json_repair(self, text: str) -> Optional[tuple[str, Any]]:
        """Attempt to extract and parse JSON embedded in markdown fences or conversational text."""
        # 1. Strip markdown code fences (```json ... ``` or ``` ...)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        candidates = []
        if fence_match:
            candidates.append(fence_match.group(1).strip())

        # 2. Extract outermost JSON object/array { ... } or [ ... ]
        obj_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if obj_match:
            candidates.append(obj_match.group(0).strip())

        candidates.append(text.strip())

        for cand in candidates:
            try:
                parsed = json.loads(cand)
                return cand, parsed
            except Exception:
                continue

        return None

    def check(self, text: str, context: Optional[dict[str, Any]] = None) -> GuardrailResult:
        start_t = time.perf_counter()
        violations: list[GuardrailViolation] = []
        sanitized_text = text
        parsed_obj: Any = None

        # Step 1: parse JSON
        try:
            parsed_obj = json.loads(text.strip())
            sanitized_text = text.strip()
        except json.JSONDecodeError as exc:
            if self.auto_repair:
                repair_result = self._attempt_json_repair(text)
                if repair_result:
                    sanitized_text, parsed_obj = repair_result
                else:
                    violations.append(
                        GuardrailViolation(
                            rule_name="schema.invalid_json",
                            severity=GuardrailSeverity.CRITICAL,
                            message=f"Output is not valid JSON: {str(exc)}",
                            details={"line": exc.lineno, "col": exc.colno},
                        )
                    )
            else:
                violations.append(
                    GuardrailViolation(
                        rule_name="schema.invalid_json",
                        severity=GuardrailSeverity.CRITICAL,
                        message=f"Output is not valid JSON: {str(exc)}",
                        details={"line": exc.lineno, "col": exc.colno},
                    )
                )

        # Step 2: Validate required keys if parsed is a dictionary
        if parsed_obj is not None and isinstance(parsed_obj, dict) and self.required_keys:
            missing_keys = [k for k in self.required_keys if k not in parsed_obj]
            if missing_keys:
                violations.append(
                    GuardrailViolation(
                        rule_name="schema.missing_required_keys",
                        severity=GuardrailSeverity.HIGH,
                        message=f"Missing required JSON keys: {missing_keys}",
                        details={"missing_keys": missing_keys},
                    )
                )

        # Step 3: Validate against Pydantic schema model if provided
        if parsed_obj is not None and self.schema_model is not None:
            try:
                if isinstance(parsed_obj, dict):
                    self.schema_model.model_validate(parsed_obj)
                elif isinstance(parsed_obj, list):
                    # If model is expecting a list or wrap in root
                    pass
            except ValidationError as val_err:
                for err in val_err.errors():
                    field_loc = " -> ".join(str(p) for p in err.get("loc", []))
                    violations.append(
                        GuardrailViolation(
                            rule_name="schema.type_validation_error",
                            severity=GuardrailSeverity.HIGH,
                            message=f"Field '{field_loc}': {err.get('msg')}",
                            details={"loc": err.get("loc"), "type": err.get("type")},
                        )
                    )

        latency_ms = (time.perf_counter() - start_t) * 1000.0
        passed = len(violations) == 0

        if passed:
            return GuardrailResult(
                guardrail_name=self.name,
                passed=True,
                action=GuardrailAction.PASS,
                score=1.0,
                violations=[],
                original_text=text,
                sanitized_text=sanitized_text,
                latency_ms=latency_ms,
                metadata={"parsed": True},
            )

        return GuardrailResult(
            guardrail_name=self.name,
            passed=False,
            action=self.action,
            score=0.0,
            violations=violations,
            original_text=text,
            sanitized_text=sanitized_text,
            latency_ms=latency_ms,
            metadata={"violations_count": len(violations)},
        )
