import json
import logging
from datetime import datetime

try:
    import markdown
except ImportError:
    markdown = None

from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState

logger = logging.getLogger(__name__)

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
    h4 {
      color: #495057;
      margin-top: 10px;
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

def parse_json_to_markdown(raw_input) -> str:
    """Parst JSON-Strukturen (Strings, Dicts oder Listen) deterministisch in sauberes Markdown."""
    if not raw_input:
        return ""

    data = raw_input
    if isinstance(raw_input, str):
        clean_input = raw_input.strip()
        if clean_input.startswith("```"):
            clean_input = clean_input.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(clean_input)
        except Exception:
            # Reiner Text (bereits Markdown)
            return raw_input

    if not isinstance(data, (dict, list)):
        return str(raw_input)

    if isinstance(data, dict) and "output" in data and isinstance(data["output"], (dict, list)):
        data = data["output"]

    md_lines = []

    if isinstance(data, dict):
        for key, value in data.items():
            section_title = key.replace("_", " ").title()

            if isinstance(value, dict):
                md_lines.append(f"### {section_title}\n")
                for sub_key, sub_val in value.items():
                    sub_title = sub_key.replace("_", " ").title()

                    if isinstance(sub_val, list):
                        md_lines.append(f"#### {sub_title}")
                        if sub_val and isinstance(sub_val[0], dict):
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
                                md_lines.append(f"- [ ] {clean_item}")
                            md_lines.append("")
                    else:
                        md_lines.append(f"**{sub_title}:** {sub_val}\n")

            elif isinstance(value, list):
                md_lines.append(f"### {section_title}")
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
                        md_lines.append(f"- [ ] {clean_item}")
                    md_lines.append("")
            else:
                md_lines.append(f"### {section_title}\n{value}\n")

    elif isinstance(data, list):
        if data and isinstance(data[0], dict):
            headers = list(data[0].keys())
            md_lines.append("| " + " | ".join([h.replace("_", " ").title() for h in headers]) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for item in data:
                row = [str(item.get(h, "")) for h in headers]
                md_lines.append("| " + " | ".join(row) + " |")
            md_lines.append("")
        else:
            for item in data:
                clean_item = str(item).lstrip("- ")
                md_lines.append(f"- [ ] {clean_item}")
            md_lines.append("")

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
            elif line_str.startswith("#### "):
                html_lines.append(f"<h4>{line_str[5:]}</h4>")
            elif line_str.startswith("- ") or line_str.startswith("* "):
                html_lines.append(f"<li>{line_str[2:]}</li>")
            elif line_str:
                html_lines.append(f"<p>{line_str}</p>")
        html = "\n".join(html_lines)

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

        season_plan_raw = get_content("season_plan")
        weekly_plan_raw = get_content("weekly_plan")

        # 1. Beide Pläne deterministisch aus JSON/Dict in Markdown wandeln
        season_md = parse_json_to_markdown(season_plan_raw)
        weekly_md = parse_json_to_markdown(weekly_plan_raw)

        # 2. Abschnitte zusammenbauen
        full_md = (
            "## Section 1: Season Plan Overview\n\n"
            f"{season_md}\n\n"
            "## Section 2: 4-Week Plan\n\n"
            f"{weekly_md}"
        )

        # 3. In HTML konvertieren
        content_html = _convert_markdown_to_html(full_md)
        planning_html = STATIC_PLANNING_HTML_TEMPLATE.replace("<!--CONTENT_PLACEHOLDER-->", content_html)

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
