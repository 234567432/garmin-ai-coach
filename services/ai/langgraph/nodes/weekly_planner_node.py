import json
import logging
import math
from datetime import datetime, timedelta

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import AgentOutput
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.message_helper import normalize_langchain_messages
from services.ai.langgraph.utils.output_helper import extract_agent_content, extract_expert_output
from services.ai.model_config import ModelSelector
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

WEEKLY_PLANNER_SYSTEM_PROMPT = """## Goal
Create detailed, practical training plans that balance stress and recovery.
## Principles
- Adaptation: Progressive overload with adequate recovery.
- Specificity: Training must match the demands of the event.
- Individualization: Adapt to the athlete's current state and history."""

WEEKLY_PLANNER_USER_PROMPT = """## Task
Create a detailed 28-day (4-week) training plan.

## STRICT DATE MAPPING RULES (CRITICAL)
- You MUST use ONLY the following pre-calculated dates for the 4-week schedule:
  Week 1: {exact_dates_week_1}
  Week 2: {exact_dates_week_2}
  Week 3: {exact_dates_week_3}
  Week 4: {exact_dates_week_4}
- DO NOT invent dates in November or December! Use ONLY the exact dates provided above.

## Constraints
- **Honor the Phase**: Prioritize the Season Plan's phase intent.
- **Respect Readiness**: Adjust intensity based on Physiology/Metrics signals (e.g., pull back if recovery is low).
- **Integrate Signals**: Use Activity Expert advice for session structure.
- **Brevity**: Use standard notation (e.g., "4x(5' Z4, 2' r)") to keep the plan compact.

## Inputs
### Season Plan
```markdown
{season_plan}
```
### Athlete Context
- Name: {athlete_name}
- Current Date: {current_date_str}
- Days Until Race: {days_until_race} (approx {weeks_until_race} weeks)
- Competitions: ```json {competitions} ```
- **User Context**: ``` {planning_context} ```

### Expert Analysis
- Metrics: ``` {metrics_analysis} ```
- Activity: ``` {activity_analysis} ```
- Physiology: ``` {physiology_analysis} ```

## Output Requirements
1. **Zones Table**: Define intensity zones first.
2. **Structure**: Group by Week (1-4).
3. **Daily Format**:
   - **DAY & DATE**: e.g., "Mon, Sep 14"
   - **FOCUS**: 1-2 words (e.g., "Recovery", "VO2max")
   - **WORKOUT**: Concise structure string.
   - **PURPOSE**: One short sentence.
   - **ADAPTATION**: "If tired: ..."
"""

WEEKLY_PLANNER_FINAL_CHECKLIST = """
## Final Checklist
- Follow 28-day horizon and week grouping.
- Do not contradict expert constraints.
- Keep output compact and structured.
"""

def prepare_planning_context(current_date_str: str, competition_date_str: str) -> dict:
    curr_dt = datetime.strptime(current_date_str, "%Y-%m-%d")
    comp_dt = datetime.strptime(competition_date_str, "%Y-%m-%d")
    
    days_left = (comp_dt - curr_dt).days
    weeks_left = math.ceil(days_left / 7)
    
    dates_next_4_weeks = []
    for i in range(28):
        day_dt = curr_dt + timedelta(days=i)
        dates_next_4_weeks.append(day_dt.strftime("%a, %b %d"))
        
    return {
        "days_until_race": days_left,
        "weeks_until_race": weeks_left,
        "exact_dates_28_days": dates_next_4_weeks
    }

async def weekly_planner_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting weekly planner node")

    hitl_enabled = state.get("hitl_enabled", True)
    logger.info("Weekly planner node: HITL %s", "enabled" if hitl_enabled else "disabled")

    agent_start_time = datetime.now()

    raw_date = state.get("current_date", "2026-09-14")
    current_date_str = raw_date.get("date", "2026-09-14") if isinstance(raw_date, dict) else str(raw_date)

    competitions = state.get("competitions", [])
    comp_date_str = competitions[0].get("date", "2026-10-03") if competitions else "2026-10-03"

    date_ctx = prepare_planning_context(current_date_str, comp_date_str)
    dates = date_ctx["exact_dates_28_days"]

    tools = configure_node_tools(
        agent_name="weekly_planner",
        plot_storage=None,
        plotting_enabled=False,
    )

    system_prompt = (
        get_workflow_context("weekly_planner")
        + WEEKLY_PLANNER_SYSTEM_PROMPT
        + (get_hitl_instructions("weekly_planner") if hitl_enabled else "")
        + WEEKLY_PLANNER_FINAL_CHECKLIST
    )

    qa_messages = normalize_langchain_messages(state.get("weekly_planner_messages", []))
    
    user_message = {
        "role": "user",
        "content": WEEKLY_PLANNER_USER_PROMPT.format(
            exact_dates_week_1=", ".join(dates[0:7]),
            exact_dates_week_2=", ".join(dates),
            exact_dates_week_3=", ".join(dates),
            exact_dates_week_4=", ".join(dates),
            season_plan=extract_agent_content(state.get("season_plan")),
            athlete_name=state["athlete_name"],
            current_date_str=current_date_str,
            days_until_race=date_ctx["days_until_race"],
            weeks_until_race=date_ctx["weeks_until_race"],
            competitions=json.dumps(competitions, indent=2),
            planning_context=state["planning_context"],
            metrics_analysis=extract_expert_output(state.get("metrics_outputs"), "for_weekly_planner"),
            activity_analysis=extract_expert_output(state.get("activity_outputs"), "for_weekly_planner"),
            physiology_analysis=extract_expert_output(state.get("physiology_outputs"), "for_weekly_planner"),
        ),
    }
    base_messages = [{"role": "system", "content": system_prompt}, user_message]

    base_llm = ModelSelector.get_llm(AgentRole.WORKOUT)
    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(AgentOutput)

    async def call_weekly_planning():
        messages_with_qa = base_messages + qa_messages
        if tools:
            return await handle_tool_calling_in_node(
                llm_with_tools=llm_with_structure,
                messages=messages_with_qa,
                tools=tools,
                max_iterations=15,
            )
        
        response = await base_llm.ainvoke(messages_with_qa)
        content_text = response.content if hasattr(response, "content") else str(response)
        return AgentOutput(output=content_text, content=content_text)

    async def node_execution():
        agent_output = await retry_with_backoff(
            call_weekly_planning, AI_ANALYSIS_CONFIG, "Weekly Planning"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        log_node_completion("Weekly planning", execution_time)

        return {
            "weekly_plan": agent_output.model_dump(),
            "costs": [create_cost_entry("weekly_planner", execution_time)],
        }

    return await execute_node_with_error_handling(
        node_name="Weekly planner",
        node_function=node_execution,
        error_message_prefix="Weekly planning failed",
    )
