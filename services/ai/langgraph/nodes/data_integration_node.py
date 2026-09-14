import logging
from datetime import datetime
from typing import Any

from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState

logger = logging.getLogger(__name__)


def _enrich_activity_laps(laps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reichert Runden-Daten mit Geländemarkierungen und eindeutigen Metrik-Bezeichnungen an."""
    enriched_laps = []

    for lap in laps:
        ele_gain = lap.get("elevation_gain_m", 0) or 0
        ele_loss = lap.get("elevation_loss_m", 0) or 0
        dist_m = lap.get("distance_m", 0) or (lap.get("distance_km", 1) * 1000)

        # Prozentuale Steigungsberechnung
        net_elevation = ele_gain - ele_loss
        grade_pct = (net_elevation / dist_m * 100.0) if dist_m > 0 else 0.0

        # Explizite Geländezuweisung für kleine LLMs (8b)
        if grade_pct >= 3.0:
            terrain_tag = f"Uphill (+{grade_pct:.1f}%)"
        elif grade_pct <= -3.0:
            terrain_tag = f"Downhill ({grade_pct:.1f}%)"
        else:
            terrain_tag = "Flat / Rolling"

        enriched_lap = {
            "lap_number": lap.get("lap_index") or lap.get("lap_number"),
            "distance_km": round(dist_m / 1000.0, 2) if dist_m else 0,
            "speed_kmh": lap.get("speed_kmh") or lap.get("pace_kmh"),
            
            # GELÄNDE-LABEL: Verhindert "Falsch-Pacing"-Diagnosen bergauf
            "terrain_category": terrain_tag,
            "elevation_gain_m": ele_gain,
            "elevation_loss_m": ele_loss,

            # KLARNAME: Verhindert HR / HRV Verwechselung
            "heart_rate_bpm": lap.get("avg_hr") or lap.get("heart_rate"),
            "max_heart_rate_bpm": lap.get("max_hr"),
            
            # Laufdynamik-Metriken
            "ground_contact_time_ms": lap.get("avg_gct"),
            "cadence_spm": lap.get("avg_cadence"),
            "stride_length_m": lap.get("avg_stride_length"),
        }
        enriched_laps.append(enriched_lap)

    return enriched_laps


async def data_integration_node(state: TrainingAnalysisState) -> dict[str, Any]:
    logger.info("Starting data integration node")

    try:
        agent_start_time = datetime.now()

        available_data_names = [
            name
            for name, key in [
                ("metrics analysis", "metrics_outputs"),
                ("activity analysis", "activity_outputs"),
                ("physiology analysis", "physiology_outputs"),
            ]
            if state.get(key)
        ]
        available_data_str = ", ".join(available_data_names) if available_data_names else "none"
        logger.info("Data integration: Available analysis data: %s", available_data_str)

        # --- Datenanreicherung für das LLM ---
        garmin_data = state.get("garmin_data") or {}
        raw_activities = garmin_data.get("activities", [])
        processed_activities = []

        for act in raw_activities:
            act_copy = dict(act)
            if "laps" in act_copy and isinstance(act_copy["laps"], list):
                act_copy["laps"] = _enrich_activity_laps(act_copy["laps"])
            processed_activities.append(act_copy)

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        logger.info("Data integration completed in %.2fs", execution_time)

        return {
            "season_plan_complete": True,
            "processed_activities": processed_activities,
            "costs": [
                {
                    "agent": "data_integration",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    except Exception as exc:
        logger.exception("Data integration node failed")
        return {"errors": [f"Data integration failed: {exc!s}"]}
