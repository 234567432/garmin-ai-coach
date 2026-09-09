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

# Statisches HTML-Template mit isoliertem CSS-Block
STATIC_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Athletic Performance Analysis</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #2c3e50;
      margin: 0;
      padding: 0;
      background-color: #f8f9fa;
    }
    .container {
      max-width: 1000px;
      margin: 30px auto;
      padding: 40px;
      background-color: #ffffff;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
      border-radius: 8px;
    }
    h1 {
      color: #1a252f;
      border-bottom: 2px solid #007bff;
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
    }
    ul, ol {
      padding-left: 20px;
    }
    li {
      margin-bottom: 8px;
    }
    p {
      margin-bottom: 12px;
    }
    .content-body {
      font-size: 15px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Athletic Performance Analysis</h1>
    <div class="content-body">
      <!--CONTENT_PLACEHOLDER-->
    </div>
  </div>
</body>
</html>
"""

FORMATTER_SYSTEM_PROMPT = """You are a sports data assistant. Your sole task is to structure athletic performance insights into clean, structured Markdown text. Do NOT generate HTML tags, CSS styles, or layout blocks."""

# Sicherer String-Aufbau ohne Darstellungskonflikte im Chat
_BT = "```"
FORMATTER_USER_PROMPT_BASE = (
    "Summarize and structure the following synthesis result into readable Markdown "
    "with clear headings (##, ###), bullet points, and bold text. Do not write HTML or CSS tags.\n\n"
    "## Content\n"
    f"{_BT}markdown\n"
    "{synthesis_result}\n"
    f"{_BT}\n"
)

FORMATTER_PLOT_INSTRUCTIONS = """
## Plot Integration
- **Preserve**: Keep `[PLOT:plot_id]` references EXACTLY as written on a separate line.
"""


def _convert_markdown_to_html(md_text: str) -> str:
    """Konvertiert Markdown in echtes HTML."""
    if markdown:
        return markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    
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
    return "\n".join(html_lines)


async def formatter_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting HTML formatter node")

    try:
        plotting_enabled = state.get("plotting_enabled", False)
        logger.info(
            "Formatter node: Plotting %s - %s plot integration instructions",
            "enabled" if plotting_enabled else "disabled",
            "including" if plotting_enabled else "no",
        )

        agent_start_time = datetime.now()

        async def call_html_formatting():
            synthesis_result = extract_text_content(state.get("synthesis_result", ""))

            response = await ModelSelector.get_llm(AgentRole.FORMATTER).ainvoke([
                {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    FORMATTER_USER_PROMPT_BASE.format(synthesis_result=synthesis_result)
                    + (FORMATTER_PLOT_INSTRUCTIONS if plotting_enabled else "")
                )},
            ])
            raw_markdown = extract_text_content(response)
            
            content_html = _convert_markdown_to_html(raw_markdown)
            return STATIC_HTML_TEMPLATE.replace("<!--CONTENT_PLACEHOLDER-->", content_html)

        analysis_html = await retry_with_backoff(
            call_html_formatting, AI_ANALYSIS_CONFIG, "HTML Formatting"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        logger.info("HTML formatting completed in %.2fs", execution_time)

        return {
            "analysis_html": analysis_html,
            "costs": [
                {
                    "agent": "formatter",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    except Exception as exc:
        logger.exception("Formatter node failed")
        return {"errors": [f"HTML formatting failed: {exc!s}"]}
