"""FastAPI surface for the GridWise preliminary.

Only two endpoints are exercised by the judge harness, and both names and
payload shapes are fixed by the Problem Statement.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

try:  # optional locally, absent in the container image
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from . import llm, service
from .schemas import OptimizeResponse, ScenarioRequest

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("gridwise")

app = FastAPI(
    title="GridWise LLM-Assisted Energy Optimizer",
    description="BUP CSE Fest 2026 preliminary: operator-note interpretation and 24-hour scheduling.",
    version="1.0.0",
    docs_url="/docs",
)


@app.exception_handler(RequestValidationError)
async def _malformed_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Section 6.1: structurally invalid requests answer 400, not FastAPI's default 422."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "invalid_request", "detail": "Request body does not match the required schema."},
    )


@app.exception_handler(Exception)
async def _controlled_failure(request: Request, exc: Exception) -> JSONResponse:
    """Section 6.1: controlled 500 with no stack trace and no secrets."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": "The service could not complete this request."},
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict:
    return {
        "service": "gridwise-llm",
        "endpoints": ["GET /health", "POST /optimize-energy"],
        "interpretation_providers": llm.configured_providers() or ["deterministic-fallback-only"],
    }


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(payload: ScenarioRequest) -> OptimizeResponse:
    response, diagnostics = await service.run(payload)
    log.info(
        "scenario=%s notes=%d source=%s cost=%.2f relaxations=%s violations=%d",
        payload.scenario_id,
        len(payload.operator_notes),
        diagnostics["interpretation_source"],
        response.total_cost_bdt,
        diagnostics["relaxations"] or "-",
        len(diagnostics["residual_violations"]),
    )
    return response
