from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_MISSING = object()


def _get_field(container: Any, field_name: str) -> Any:
    if container is None:
        return _MISSING

    if hasattr(container, field_name):
        return getattr(container, field_name)

    if isinstance(container, Mapping) and field_name in container:
        return container[field_name]

    return _MISSING


def _render_receiver_payload(payload: Any) -> str:
    if payload is None:
        return ""

    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, Mapping):
        data = payload
    else:
        return str(payload)

    sections = {
        "Signals": data.get("signals", []),
        "Evidence": data.get("evidence", []),
        "Implications": data.get("implications", []),
    }
    uncertainty = data.get("uncertainty")
    if uncertainty:
        sections["Uncertainty"] = uncertainty

    lines: list[str] = []
    for title, items in sections.items():
        lines.append(f"### {title}")
        if items:
            lines.extend([f"- {item}" for item in items])
        else:
            lines.append("- None")
        lines.append("")

    return "\n".join(lines).strip()


def _save_questions_to_file(questions: list) -> None:
    """Speichert vom Modell generierte Fragen in eine Textdatei im Output-Ordner."""
    try:
        os.makedirs("output", exist_ok=True)
        questions_file = os.path.join("output", "open_questions.txt")

        formatted_questions = []
        for q in questions:
            if hasattr(q, "question"):
                formatted_questions.append(f"- {q.question}")
            elif isinstance(q, Mapping) and "question" in q:
                formatted_questions.append(f"- {q['question']}")
            else:
                formatted_questions.append(f"- {str(q)}")

        with open(questions_file, "a", encoding="utf-8") as f:
            f.write("--- Offene Fragen der KI ---\n")
            f.write("\n".join(formatted_questions) + "\n\n")

        logger.info("Offene Fragen wurden in '%s' gesichert.", questions_file)
    except Exception as e:
        logger.error("Fehler beim Speichern der offenen Fragen: %s", e)


def extract_expert_output(expert_output: Any, target_field: str) -> str:
    if expert_output is None:
        logger.warning("Expert output is None for field '%s'. Returning empty string.", target_field)
        return ""

    output_container: Any = _MISSING
    if hasattr(expert_output, "output"):
        output_container = expert_output.output
    elif isinstance(expert_output, Mapping):
        output_container = expert_output.get("output")

    if isinstance(output_container, list):
        logger.warning(
            "Expert output contains questions (List format) instead of direct analysis. "
            "Logging questions and continuing without HITL exception."
        )
        _save_questions_to_file(output_container)

        rendered_questions = []
        for q in output_container:
            if hasattr(q, "question"):
                rendered_questions.append(f"- {q.question}")
            elif isinstance(q, Mapping) and "question" in q:
                rendered_questions.append(f"- {q['question']}")
            else:
                rendered_questions.append(f"- {str(q)}")
        return "### Offene Punkte / Fragen aus der Analyse:\n" + "\n".join(rendered_questions)

    for candidate in (output_container, expert_output):
        payload = _get_field(candidate, target_field)
        if payload is not _MISSING:
            return _render_receiver_payload(payload)

    logger.warning(
        "Expert output missing '%s' field. Type: %s. Returning raw representation.",
        target_field,
        type(expert_output),
    )
    return str(output_container if output_container is not _MISSING else expert_output)


def extract_agent_content(value: Any) -> str:
    if not value:
        return ""

    if hasattr(value, "output"):
        output = value.output
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            logger.warning("AgentOutput contains questions (List format). Saving and continuing.")
            _save_questions_to_file(output)
            return "\n".join([str(item) for item in output])
        return str(output)

    if isinstance(value, dict):
        result = value.get("output") or value.get("content")
        if isinstance(result, str):
            return result
        return str(value)

    if isinstance(value, str):
        return value

    return str(value)


def extract_current_date(state: dict | None) -> str:
    """Extrahiert das Datum dynamisch aus dem State ohne hartcodierte Jahreszahlen."""
    if not state:
        return datetime.now().strftime("%Y-%m-%d")
    raw_date = state.get("current_date")
    if isinstance(raw_date, dict):
        return raw_date.get("date") or datetime.now().strftime("%Y-%m-%d")
    if isinstance(raw_date, str) and raw_date:
        return raw_date
    return datetime.now().strftime("%Y-%m-%d")


def calculate_precalculated_kpis(state_or_garmin: dict | None, current_date_str: str) -> dict:
    """Berechnet zusammenfassende KPIs aus der Historie mit flexiblen Fallbacks für verschachtelte Dicts."""
    garmin_data = {}
    if isinstance(state_or_garmin, dict):
        garmin_data = (
            state_or_garmin.get("garmin_data")
            or state_or_garmin.get("context", {}).get("garmin_data")
            or state_or_garmin
        )

    if not isinstance(garmin_data, dict):
        garmin_data = {}

    # VO2 Max (Aktuellster Wert)
    vo2_data = garmin_data.get("vo2_max_history") or garmin_data.get("vo2max") or garmin_data.get("vo2_max")
    latest_vo2 = "N/A"
    if isinstance(vo2_data, dict):
        latest_vo2 = str(vo2_data.get("latest_value") or vo2_data.get("vo2Max") or vo2_data.get("value") or "N/A")
    elif isinstance(vo2_data, (int, float, str)):
        latest_vo2 = str(vo2_data)

    # HRV Status
    hrv_data = garmin_data.get("hrv_summary") or garmin_data.get("hrv") or garmin_data.get("hrv_status")
    if isinstance(hrv_data, dict):
        hrv_status = hrv_data.get("status") or hrv_data.get("hrvSummary", {}).get("status", "Balanced")
        hrv_7d_avg = hrv_data.get("weekly_avg") or hrv_data.get("last7DaysAvg") or hrv_data.get("weeklyAvg", "N/A")
        hrv_str = f"{hrv_status} ({hrv_7d_avg} ms)" if hrv_7d_avg != "N/A" else str(hrv_status)
    elif isinstance(hrv_data, str):
        hrv_str = hrv_data
    else:
        hrv_str = "N/A"

    # Stress Level
    stress_data = garmin_data.get("stress_summary") or garmin_data.get("stress")
    avg_stress_7d = "N/A"
    if isinstance(stress_data, dict):
        avg_stress_7d = str(
            stress_data.get("avg_stress_7d")
            or stress_data.get("weekly_avg")
            or stress_data.get("avg")
            or "N/A"
        )
    elif isinstance(stress_data, (int, float, str)):
        avg_stress_7d = str(stress_data)

    stress_str = f"{avg_stress_7d} (7-day average)" if avg_stress_7d != "N/A" else "N/A"

    # Training Load Trend
    load_history = (
        garmin_data.get("training_load_history")
        or garmin_data.get("training_load")
        or garmin_data.get("load")
    )
    current_load = "N/A"
    if isinstance(load_history, list) and load_history:
        last_entry = load_history[-1]
        if isinstance(last_entry, dict):
            current_load = str(last_entry.get("load") or last_entry.get("value") or "N/A")
        else:
            current_load = str(last_entry)
    elif isinstance(load_history, (int, float, str)):
        current_load = str(load_history)

    return {
        "as_of_date": current_date_str,
        "vo2_max_current": latest_vo2,
        "hrv_status": hrv_str,
        "avg_stress_7d": stress_str,
        "current_training_load": current_load,
    }
