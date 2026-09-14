from typing import Any

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState

from .data_summarizer_node import create_data_summarizer_node


def _sanitize_hr_metrics(data: Any) -> Any:
    """Bereinigt Keys rekursiv: Trennt HR (bpm) strikt von HRV, um LLM-Missverständnisse zu verhindern."""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Schlüssel umbenennen, um Verwechselung mit HRV zu unterbinden
            if key in ("avg_hr", "hr", "heart_rate"):
                new_key = "heart_rate_bpm"
            elif key in ("max_hr", "max_heart_rate"):
                new_key = "max_heart_rate_bpm"
            elif key in ("min_hr", "min_heart_rate"):
                new_key = "min_heart_rate_bpm"
            else:
                new_key = key

            sanitized[new_key] = _sanitize_hr_metrics(value)
        return sanitized
    elif isinstance(data, list):
        return [_sanitize_hr_metrics(item) for item in data]

    return data


def extract_metrics_data(state: TrainingAnalysisState) -> dict:
    garmin_data = state.get("garmin_data", {})

    raw_metrics = {
        "training_load_history": garmin_data.get("training_load_history", []),
        "vo2_max_history": garmin_data.get("vo2_max_history", {}),
        "training_status": garmin_data.get("training_status", {}),
        "long_term_vo2_max_trend": garmin_data.get("long_term_vo2_max_trend", {}),
    }

    # Säubert alle HR-Bezeichnungen, bevor sie in den Prompt des Summarizers fließen
    return _sanitize_hr_metrics(raw_metrics)


metrics_summarizer_node = create_data_summarizer_node(
    node_name="Metrics Summarizer",
    agent_role=AgentRole.SUMMARIZER,
    data_extractor=extract_metrics_data,
    state_output_key="metrics_summary",
    agent_type="metrics_summarizer",
)
