from __future__ import annotations

import logging
import os
from collections.abc import Mapping
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

    # Wenn der Output eine Liste ist, handelt es sich um Fragen aus der KI-Analyse
    if isinstance(output_container, list):
        logger.warning(
            "Expert output contains questions (List format) instead of direct analysis. "
            "Logging questions and continuing without HITL exception."
        )
        _save_questions_to_file(output_container)
        
        # Fragen als formatierten Text zurückgeben, damit die Synthese nicht leermeldend abstürzt
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

    logger.warning("Expert output missing '%target_field%' field. Type: %s. Returning raw representation.", target_field, type(expert_output))
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
