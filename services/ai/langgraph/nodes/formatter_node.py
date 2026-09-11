import json
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
      margin-top: 15px;
    }
    blockquote {
      background-color: #eef6ff;
      border-left: 4px solid #007bff;
      margin: 15px 0;
      padding: 12px 18px;
      border-radius: 0 6px 6px 0;
    }
    ul, ol {
      padding-left: 20px;
    }
    li {
      margin-bottom: 8px;
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
      accent-color: #007bff;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 14px;
    }
    th, td {
      border: 1px solid #dee2e6;
      padding: 10px 12px;
      text-align: left;
    }
    th {
      background-color: #f1f3f5;
      color: #1a252f;
    }
    tr:nth-child(even) {
      background-color: #f8f9fa;
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

def parse_json_to_markdown(raw_input: str) -> str:
    """Parst JSON-Strukturen deterministisch in sauberes Markdown."""
    try:
        # Falls der String von Markdown-Code-Blocks umschlossen ist (```json ...)
        clean_input = raw_input.strip()
        if clean_input.startswith("```"):
            clean_input = clean_input.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(clean_input)
    except Exception:
        # Falls kein JSON vorliegt, unberührt zurückgeben
        return raw_input

    if not isinstance(data, dict):
        return raw_input

    if "output" in data and isinstance(data["output"], dict):
        data = data["output"]

    md_lines = []

    for key, value in data.items():
        section_title = key.replace("_", " ").title()

        if isinstance(value, dict):
            md_lines.append(f"## {section_title}\n")
            for sub_key, sub_val in value.items():
                sub_title = sub_key.replace("_", " ").title()

                if isinstance(sub_val, list):
                    md_lines.append(f"### {sub_title}")
                    if sub_val and isinstance(sub_val[0], dict):
                        # Tabelle für Liste von Dicts (z. B. Metrics/KPIs)
                        headers = list(sub_val[0].keys())
                        md_lines.append("| " + " | ".join([h.replace("_", " ").title() for h in headers]) + " |")
                        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        for item in sub_val:
                            row = [str(item.get(h, "")) for h in headers]
                            md_lines.append("| " + " | ".join(row) + " |")
                        md_lines.append("")
                    else:
                        for item in sub_val:
                            clean_item = str(item).lstrip("- ")
                            md_lines.append(f"- {clean_item}")
                        md_lines.append("")
                else:
                    md_lines.append(f"**{sub_title}:** {sub_val}\n")

        elif isinstance(value, list):
            md_lines.append(f"## {section_title}")
            if value and isinstance(value[0], dict):
                headers = list(value[0].keys())
                md_lines.append("| " + " | ".join([h.replace("_", " ").title() for h in headers]) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for item in value:
                    row = [str(item.get(h, "")) for h in headers]
                    md_lines.append("| " + " | ".join(row) + " |")
                md_lines.append("")
            else:
                for item in value:
                    clean_item = str(item).lstrip("- ")
                    md_lines.append(f"- {clean_item}")
                md_lines.append("")
        else:
            md_lines.append(f"## {section_title}\n{value}\n")

    return "\n".join(md_lines)


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
            elif line_str.startswith("> "):
                html_lines.append(f"<blockquote>{line_str[2:]}</blockquote>")
            elif line_str:
                html_lines.append(f"<p>{line_str}</p>")
        html = "\n".join(html_lines)

    html = html.replace("[ ] ", '<label><input type="checkbox"> ')
    html = html.replace("[x] ", '<label><input type="checkbox" checked> ')
    return html


async def formatter_node(state: TrainingAnalysisState) -> dict[str, list | str]:
    logger.info("Starting HTML formatter node")

    try:
        agent_start_time = datetime.now()

        async def call_html_formatting():
            synthesis_result = extract_text_content(state.get("synthesis_result", ""))

            # 1. JSON-Struktur deterministisch in Markdown auflösen
            markdown_content = parse_json_to_markdown(synthesis_result)

            # 2. Markdown in HTML konvertieren
            content_html = _convert_markdown_to_html(markdown_content)
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
