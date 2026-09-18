"""Pydantic models for the GridWise request/response contract.

Canonical source: BUP CSE Fest 2026 Preliminary Problem Statement, sections 07 and 10.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DIRECTIVE_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

BATTERY_ACTIONS = ("charge", "discharge", "idle")

HORIZON = 24


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------
class HourEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float


class Battery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @model_validator(mode="after")
    def _coherent(self) -> "Battery":
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh exceeds capacity_kwh")
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh")
        return self


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    operator_notes: List[str]
    hours: List[HourEntry]
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def _notes(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("operator_notes must contain at least one note")
        # Spec says 1-3. We accept more rather than rejecting a scorable request,
        # but every note still gets exactly one interpretation entry.
        if len(v) > 20:
            raise ValueError("operator_notes contains too many entries")
        for n in v:
            if not isinstance(n, str) or not n.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: List[HourEntry]) -> List[HourEntry]:
        if len(v) != HORIZON:
            raise ValueError("hours must contain exactly 24 entries")
        seen = sorted(h.hour for h in v)
        if seen != list(range(HORIZON)):
            raise ValueError("hours must cover each hour 0..23 exactly once")
        return v

    def ordered_hours(self) -> List[HourEntry]:
        return sorted(self.hours, key=lambda h: h.hour)


# --------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------
class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: Literal[DIRECTIVE_TYPES]  # type: ignore[valid-type]
    structured_adjustment: Optional[Dict[str, Any]]
    explanation: str


class HourPlan(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal[BATTERY_ACTIONS]  # type: ignore[valid-type]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
