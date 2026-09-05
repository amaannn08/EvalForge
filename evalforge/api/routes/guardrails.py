"""Guardrails verification API routes."""

from fastapi import APIRouter
from evalforge.api.schemas import GuardrailCheckRequest, GuardrailCheckResponse
from evalforge.guardrails import (
    CostBudgetGuardrail,
    ExecutionMode,
    GuardrailPipeline,
    PIIGuardrail,
    PromptInjectionGuardrail,
    SchemaValidationGuardrail,
    ToxicityGuardrail,
)

router = APIRouter(prefix="/api/v1/guardrails", tags=["Guardrails"])


def _build_pipeline(guards_list: list[str], mode_str: str) -> GuardrailPipeline:
    mode = ExecutionMode.COLLECT_ALL
    if mode_str.upper() == "FAIL_FAST":
        mode = ExecutionMode.FAIL_FAST
    elif mode_str.upper() == "PARALLEL":
        mode = ExecutionMode.PARALLEL

    guards = []
    for g in guards_list:
        g_clean = g.lower().strip()
        if g_clean == "pii":
            guards.append(PIIGuardrail())
        elif g_clean in ("injection", "prompt_injection"):
            guards.append(PromptInjectionGuardrail())
        elif g_clean in ("toxicity", "toxic"):
            guards.append(ToxicityGuardrail())
        elif g_clean in ("schema", "json"):
            guards.append(SchemaValidationGuardrail())
        elif g_clean in ("budget", "sla"):
            guards.append(CostBudgetGuardrail())

    if not guards:
        guards = [PIIGuardrail(), PromptInjectionGuardrail(), ToxicityGuardrail()]

    return GuardrailPipeline(guards, mode=mode)


@router.post("/check", response_model=GuardrailCheckResponse)
def check_text(req: GuardrailCheckRequest):
    pipeline = _build_pipeline(req.guardrails, req.mode)
    res = pipeline.run(req.text, context=req.context)
    return GuardrailCheckResponse(
        passed=res.passed,
        final_action=res.final_action.value,
        original_text=res.original_text,
        sanitized_text=res.sanitized_text,
        overall_score=res.overall_score,
        total_latency_ms=res.total_latency_ms,
        violation_count=len(res.violations),
        violations=[v.to_dict() for v in res.violations],
        results=[r.to_dict() for r in res.results],
    )
