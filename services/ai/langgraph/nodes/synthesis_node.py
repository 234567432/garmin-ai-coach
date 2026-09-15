import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.output_helper import (
    calculate_precalculated_kpis,
    extract_current_date,
    extract_expert_output,
)
from services.ai.model_config import ModelSelector
from services.ai.tools.plotting import PlotStorage
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

logger = logging.getLogger(__name__)


SYNTHESIS_SYSTEM_PROMPT_BASE = """You are a performance integration specialist.
## Goal
Create comprehensive, actionable insights by synthesizing multiple data streams.
## Principles
- Integrate: Connect insights from metrics, activity, and physiology.
- Contextualize: Relate data to the athlete's history and goals.
- Simplify: Make complex relationships understandable."""

SYNTHESIS_PLOT_INSTRUCTIONS = """
## Plot Integration
- Include plot references as `[PLOT:plot_id]` in your text.
- These will become interactive charts."""

SYNTHESIS_USER_PROMPT_BASE = """Synthesize the expert analyses into a comprehensive athlete report for {athlete_name}.

## Inputs
### Pre-Calculated Key Metrics (STRICT SOURCE FOR KPI TABLE)
```json
{precomputed_kpis}
```

### Metrics
```markdown
{metrics_result}
```
### Activity
```markdown
{activity_result}
```
### Physiology
```markdown
{physiology_result}
```
### Direct Athlete Feedback / Answers
{user_answers_block}

### Context
- Competitions: ```json {competitions} ```
- Date: {current_date}
- Style: ```markdown {style_guide} ```

## Task
1. **Integrate**: Connect load (metrics), execution (activity), and response (physiology).
2. **Identify Patterns**: Spot trends in performance and adaptation.
3. **Synthesize**: Create a coherent story, not just a list of facts.

## STRICT KPI TABLE RULES
1. ONLY use the data provided under "Pre-Calculated Key Metrics" for the Key Performance Indicators table.
2. DO NOT use technical raw fields like "Last Night 5min High" or single-second max stress spikes (e.g. 97).
3. Always display HRV as overall weekly status (e.g., "Balanced / 55 ms") and Stress as weekly average (e.g., "24 (Low)").
4. Do NOT reference historical dates older than 28 days from the current date ({current_date}).

## Output Format & Formatting Rules
- **Executive Summary**: High-level status and key takeaways.
- **Key Performance Indicators**: Table format.
  - CRITICAL: Always insert an empty newline BEFORE starting any Markdown table.
- **Deep Dive**: Structured sections with clear headings.
- **Recommendations**: Brief and actionable.
- **Tone**: Professional, evidence-based, encouraging."""

SYNTHESIS_USER_PLOT_INSTRUCTIONS = """
## Plot References
- Include each unique `[PLOT:plot_id]` EXACTLY ONCE.
- Do not duplicate references."""


async def synthesis_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting synthesis node")

    try:
        plot_storage = PlotStorage(state["execution_id"])
        plotting_enabled = state.get("plotting_enabled", False)

        logger.info(
            "Synthesis node: Plotting %s - %s plot integration instructions",
            "enabled" if plotting_enabled else "disabled",
            "including" if plotting_enabled else "no",
        )

        agent_start_time = datetime.now()

        current_date_str = extract_current_date(state)
        garmin_data = state.get("garmin_data", {})
        precomputed_kpis = calculate_precalculated_kpis(garmin_data, current_date_str)

        raw_answers = state.get("user_answers") or state.get("context", {}).get("answers", "")
        if raw_answers:
            if isinstance(raw_answers, list):
                user_answers_block = "\n".join([f"- {item}" for item in raw_answers])
            else:
                user_answers_block = str(raw_answers)
        else:
            user_answers_block = "No direct feedback provided by athlete for this run."

        available_plots = state.get("plots", []) or state.get("available_plots", [])
        if plotting_enabled and available_plots:
            plot_lines = []
            for p in available_plots:
                if isinstance(p, dict):
                    p_id = p.get("plot_id", "")
                    desc = p.get("description", "")
                    plot_lines.append(f"- ID: {p_id} | Description: {desc}")
                else:
                    plot_lines.append(f"- ID: {p}")
            plot_list_str = "\n".join(plot_lines)
            user_plot_instructions = (
                f"\n\n## Available Plot References\n"
                f"You MUST embed ONLY the following generated plot IDs into your markdown using the exact syntax `[PLOT:plot_id]`:\n"
                f"{plot_list_str}\n"
                f"CRITICAL: Do NOT invent or alter any plot IDs. Include each unique tag EXACTLY ONCE."
            )
        else:
            user_plot_instructions = (
                "\n\n## Plot References\n"
                "No plots are available for this run. Do NOT insert any `[PLOT: ...]` tags anywhere in your output."
            )

        async def call_synthesis_analysis():
            llm = ModelSelector.get_llm(AgentRole.SYNTHESIS)
            system_content = SYNTHESIS_SYSTEM_PROMPT_BASE + (
                SYNTHESIS_PLOT_INSTRUCTIONS if plotting_enabled else ""
            )
            user_content = (
                SYNTHESIS_USER_PROMPT_BASE.format(
                    athlete_name=state.get("athlete_name", "Athlete"),
                    precomputed_kpis=json.dumps(precomputed_kpis, indent=2),
                    metrics_result=extract_expert_output(
                        state.get("metrics_outputs"), "for_synthesis"
                    ),
                    activity_result=extract_expert_output(
                        state.get("activity_outputs"), "for_synthesis"
                    ),
                    physiology_result=extract_expert_output(
                        state.get("physiology_outputs"), "for_synthesis"
                    ),
                    user_answers_block=user_answers_block,
                    competitions=json.dumps(state.get("competitions", []), indent=2),
                    current_date=current_date_str,
                    style_guide=state.get("style_guide", ""),
                )
                + user_plot_instructions
            )

            response = await llm.ainvoke([
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ])
            return response.content

        synthesis_result = await retry_with_backoff(
            call_synthesis_analysis, AI_ANALYSIS_CONFIG, "Synthesis Analysis with Tools"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        logger.info("Synthesis analysis completed in %.2fs", execution_time)

        return {
            "synthesis_result": synthesis_result,
            "synthesis_complete": True,
            "costs": [
                {
                    "agent": "synthesis",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "available_plots": plot_storage.list_available_plots(),
        }

    except Exception as exc:
        logger.exception("Synthesis node failed")
        return {"errors": [f"Synthesis analysis failed: {exc!s}"]}
