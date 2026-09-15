import json
import logging
import math
from datetime import datetime, timedelta

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import AgentOutput
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.output_helper import extract_expert_output
from services.ai.model_config import ModelSelector
from services.ai.utils.plan_storage import FilePlanStorage
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import (
    configure_node_tools,
    create_cost_entry,
    execute_node_with_error_handling,
    log_node_completion,
)
from .prompt_components import get_hitl_instructions, get_workflow_context
from .tool_calling_helper import handle_tool_calling_in_node

logger = logging.getLogger(__name__)

SEASON_PLANNER_SYSTEM_PROMPT = """You are a strategic season planner.
## Goal
Create strategic season plans for long-term athletic development.
## Principles
- Strategic: Focus on macro-cycles and phases.
- Adaptive: Use expert insights to tailor the plan.
- Systematic: Ensure logical progression towards goals."""

SEASON_PLANNER_USER_PROMPT = """Create a STRATEGIC, HIGH-LEVEL season plan.

## Inputs
- Athlete: {athlete_name}
- Date: ```json {current_date} ```
- Competitions: ```json {competitions} ```

STRICT TIMELINE CONSTRAINTS:
- Current Date: {current_date}
- Main Race Date: {competition_date}
- DAYS UNTIL RACE: {days_until_race} days (approx. {weeks_until_race} weeks)!
- CRITICAL: Do NOT create a generic 24-week plan! You ONLY have {weeks_until_race} weeks left until the main race. 
- Allocate your remaining weeks strictly into Peak/Specific Prep followed immediately by Tapering.

## Expert Insights
### Metrics
```markdown
{metrics_insights}
```
### Activity
```markdown
{activity_insights}
```
### Physiology
```markdown
{physiology_insights}
```

## Task
Create a macro-cycle framework.
- **Integrate**: Use expert insights as your north star.
- **Strategize**: Define phases, themes, and focus areas.
- **Respect Boundaries**: Do NOT prescribe daily workouts (Weekly Planner's job).

## Output Requirements
Format as structured markdown.
1. **Phases**: Define phases with goals and themes matching the remaining {weeks_until_race} weeks.
2. **Expert Rationale**: Explicitly reference how Metrics, Activity, and Physiology informed the plan.
3. **Constraints**: Qualitative constraints derived from experts.

**Stay high-level**. Design the **map of the season**, not the turn-by-turn navigation. **BE CONCISE**."""


def prepare_planning_context(current_date_str: str, competition_date_str: str) -> dict:
    curr_dt = datetime.strptime(current_date_str, "%Y-%m-%d")
    comp_dt = datetime.strptime(competition_date_str, "%Y-%m-%d")

    days_left = (comp_dt - curr_dt).days
    weeks_left = math.ceil(days_left / 7)

    dates_next_4_weeks = []
    for i in range(28):
        day_dt = curr_dt + timedelta(days=i)
        dates_next_4_weeks.append(day_dt.strftime("%a, %b %d (%Y-%m-%d)"))

    return {
        "days_until_race": days_left,
        "weeks_until_race": weeks_left,
        "exact_dates_28_days": dates_next_4_weeks,
    }


async def season_planner_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting season planner node")

    hitl_enabled = state.get("hitl_enabled", True)
    logger.info("Season planner node: HITL %s", "enabled" if hitl_enabled else "disabled")

    agent_start_time = datetime.now()

    # 1. Datum & Wettkämpfe extrahieren
    raw_date = state.get("current_date", "2026-09-14")
    current_date_str = (
        raw_date.get("date", "2026-09-14") if isinstance(raw_date, dict) else str(raw_date)
    )

    competitions = state.get("competitions", [])
    comp_date_str = (
        competitions[0].get("date", "2026-10-03") if competitions else "2026-10-03"
    )

    # 2. Zeitrahmen berechnen
    date_ctx = prepare_planning_context(current_date_str, comp_date_str)

    tools = configure_node_tools(
        agent_name="season_planner",
        plot_storage=None,
        plotting_enabled=False,
    )

    system_prompt = (
        SEASON_PLANNER_SYSTEM_PROMPT
        + get_workflow_context("season_planner")
        + (get_hitl_instructions("season_planner") if hitl_enabled else "")
    )

    qa_messages_raw = state.get("season_planner_messages", [])
    qa_messages = []
    for msg in qa_messages_raw:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
            qa_messages.append({"role": role, "content": msg.content})
        else:
            qa_messages.append(msg)

    existing_season_plan = ""
    try:
        storage = FilePlanStorage()
        loaded_plan = storage.load_plan(state["user_id"], "season_plan")
        if loaded_plan:
            existing_season_plan = loaded_plan
    except Exception as exc:
        logger.warning("Could not read existing season plan: %s", exc)

    base_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": SEASON_PLANNER_USER_PROMPT.format(
                athlete_name=state["athlete_name"],
                current_date=current_date_str,
                competition_date=comp_date_str,
                days_until_race=date_ctx["days_until_race"],
                weeks_until_race=date_ctx["weeks_until_race"],
                competitions=json.dumps(competitions, indent=2),
                metrics_insights=extract_expert_output(
                    state.get("metrics_outputs"), "for_season_planner"
                ),
                activity_insights=extract_expert_output(
                    state.get("activity_outputs"), "for_season_planner"
                ),
                physiology_insights=extract_expert_output(
                    state.get("physiology_outputs"), "for_season_planner"
                ),
            )
            + (
                f"\n\n## Existing Season Plan\nWe have an existing season plan. Do NOT start from scratch. Review this plan against the new expert insights. If the plan is still valid, maintain the phase structure and just refine the details. Only trigger a full replan if the new data suggests the old plan is dangerously off-track.\n\n```markdown\n{existing_season_plan}\n```"
                if existing_season_plan
                else ""
            ),
        },
    ]

    base_llm = ModelSelector.get_llm(AgentRole.SEASON_PLANNER)

    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(AgentOutput)

    async def call_season_planning():
        messages_with_qa = base_messages + qa_messages
        if tools:
            return await handle_tool_calling_in_node(
                llm_with_tools=llm_with_structure,
                messages=messages_with_qa,
                tools=tools,
                max_iterations=15,
            )
        else:
            response = await base_llm.ainvoke(messages_with_qa)
            content_text = response.content if hasattr(response, "content") else str(response)
            return AgentOutput(output=content_text, content=content_text)

    async def node_execution():
        agent_output = await retry_with_backoff(
            call_season_planning, AI_ANALYSIS_CONFIG, "Season Planning"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        log_node_completion("Season planning", execution_time)

        return {
            "season_plan": agent_output.model_dump(),
            "costs": [create_cost_entry("season_planner", execution_time)],
        }

    return await execute_node_with_error_handling(
        node_name="Season planner",
        node_function=node_execution,
        error_message_prefix="Season planning failed",
    )
