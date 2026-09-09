import logging
from datetime import datetime

try:
    import markdown
except ImportError:
    markdown = None

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .tool_calling_helper import extract_text_content

logger = logging.getLogger(__name__)

# Statisches HTML-Template mit CSS für kompaktes Layout, Tabellen und Checkboxen
STATIC_PLANNING_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Athletic Training Plan</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      color: #2c3e50;
      margin: 0;
      padding: 0;
      background-color: #f8f9fa;
    }
    .container {
      max-width: 1100px;
      margin: 20px auto;
      padding: 30px;
      background-color: #ffffff;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
      border-radius: 8px;
    }
    h1 {
      color: #1a252f;
      border-bottom: 2px solid #28a745;
      padding-bottom: 10px;
      margin-top: 0;
    }
    h2 {
      color: #2c3e50;
      margin-top: 25px;
      border-bottom: 1px solid #e9ecef;
      padding-bottom: 5px;
    }
    h3 {
      color: #34495e;
      margin-top: 15px;
    }
    ul {
      padding-left: 20px;
      list-style-type: none;
    }
    li {
      margin-bottom: 6px;
    }
    label {
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      cursor: pointer;
      accent-color: #28a745;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 15px 0;
      font-size: 14px;
    }
    th, td {
      border: 1px solid #dee2e6;
      padding: 8px 10px;
      text-align: left;
    }
    th {
      background-color: #f1f3f5;
      color: #1a252f;
    }
    tr:nth-child(even) {
      background-color: #f8f9fa;
    }
    .content-body {
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Athletic Training Plan</h1>
    <div class="content-body">
      <!--CONTENT_PLACEHOLDER-->
    </div>
  </div>
</body>
</html>
"""

PLAN_FORMATTER_SYSTEM_PROMPT = """You are a sports data assistant. Your sole task is to structure training plans into clean, structured Markdown text.
Use clear headings (##, ###), tables for weekly schedules, and task list items (- [ ]) for workouts and sub-tasks.
Do NOT generate HTML tags, CSS styles, or layout blocks."""

_BT = "```"
PLAN_FORMATTER_USER_PROMPT_BASE = (
    "Transform the following training plan inputs into readable Markdown.\n"
    "Structure into Section 1 (Season Plan Overview) and Section 2 (4-Week Plan).\n"
    "Use Markdown tables where applicable and task list items (- [ ]) for actionable workouts.\n"
    "Do not write HTML or CSS tags.\n\n"
    "## Season Plan\n"
    f"{_BT}markdown\n"
    "{season_plan}\n"
    f"{_BT}\n\n"
    "## 4-Week Plan\n"
    f"{_BT}markdown\n"
    "{weekly_plan}\n"
    f"{_BT}\n"
)


def _convert_markdown_to_html(md_text: str) -> str:
    """Konvertiert Markdown in HTML und erzeugt interaktive Checkboxen."""
    if markdown:
        html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    else:
        html_lines = []
        for line in md_text.splitlines():
            line_str = line.strip()
            if line_str.startswith("## "):
                html_lines.append(f"<h2>{line_str[3:]}</h2>")
            elif line_str.startswith("### "):
                html_lines.append(f"<h3>{line_str[4:]}</h3>")
            elif line_str.startswith("- ") or line_str.startswith("* "):
                html_lines.append(f"<li>{line_str[2:]}</li>")
            elif line_str:
                html_lines.append(f"<p>{line_str}</p>")
        html = "\n".join(html_lines)

    # Wandelt Markdown-Tasklisten (- [ ]) in echte HTML-Checkboxen um
    html = html.replace("[ ] ", '<label><input type="checkbox"> ')
    html = html.replace("[x] ", '<label><input type="checkbox" checked> ')
    return html


async def plan_formatter_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting plan formatter node")

    try:
        agent_start_time = datetime.now()

        def get_content(field):
            value = state.get(field, "")
            if hasattr(value, "output"):
                output = value.output
                if isinstance(output, str):
                    return output
                raise ValueError("AgentOutput contains questions, not content. HITL interaction required.")
            if isinstance(value, dict):
                return value.get("output", value.get("content", value))
            return value

        async def call_plan_formatting():
            season_plan = get_content("season_plan")
            weekly_plan = get_content("weekly_plan")

            response = await ModelSelector.get_llm(AgentRole.FORMATTER).ainvoke([
                {"role": "system", "content": PLAN_FORMATTER_SYSTEM_PROMPT},
                {"role": "user", "content": PLAN_FORMATTER_USER_PROMPT_BASE.format(
                    season_plan=season_plan,
                    weekly_plan=weekly_plan
                )},
            ])
            raw_markdown = extract_text_content(response)

            content_html = _convert_markdown_to_html(raw_markdown)
            return STATIC_PLANNING_HTML_TEMPLATE.replace("<!--CONTENT_PLACEHOLDER-->", content_html)

        planning_html = await retry_with_backoff(
            call_plan_formatting, AI_ANALYSIS_CONFIG, "Plan Formatter"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        logger.info("Plan formatting completed in %.2fs", execution_time)

        return {
            "planning_html": planning_html,
            "costs": [
                {
                    "agent": "plan_formatter",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    except Exception as exc:
        logger.exception("Plan formatter node failed")
        return {"errors": [f"Plan formatting failed: {exc!s}"]}
