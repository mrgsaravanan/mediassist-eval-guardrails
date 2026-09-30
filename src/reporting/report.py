"""Render the consolidated Markdown evaluation report."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.evals.heuristics import is_refusal
from src.evals.runner import EvalRun
from src.schemas import EvalRecord

TEMPLATES = Path(__file__).parent / "templates"


def render_report(run: EvalRun) -> str:
    """Build the report text from an ``EvalRun``."""
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                      lstrip_blocks=True, keep_trailing_newline=True)
    text = env.get_template("report.md.j2").render(
        run=run, failed=[c for c in run.checks if c.status != "pass"], gc=run.guardrail_counts,
        hs=run.heuristic_summary, perf=run.performance,
        cats=lambda d: ", ".join(f"{k}: {v}" for k, v in d.items()) or "—",
        f=lambda v: "n/a" if v is None else f"{v:.3f}",
        outcome=_outcome,
        failing_h=lambda rid: ", ".join(r.name for r in run.heuristic_results[rid] if not r.passed))
    return _tidy(text)


def _outcome(r: EvalRecord) -> str:
    if r.blocked:
        v = r.output_verdict if r.blocked_stage == "output_guard" else r.input_verdict
        return f"BLOCKED@{r.blocked_stage} ({v.category})"
    return "refused by target" if is_refusal(r.user_message, r.sources) else "answered"


def _tidy(text: str) -> str:
    """Ensure Markdown blocks (headings, tables) are separated by blank lines."""
    out: list[str] = []
    for line in text.splitlines():
        prev = out[-1] if out else ""
        if prev.startswith("#") and line.strip():
            out.append("")
        starts_block = line.startswith("#") or (line.startswith("|") and not prev.startswith("|"))
        if starts_block and prev.strip():
            out.append("")
        out.append(line)
    return "\n".join(out) + "\n"


def write_report(run: EvalRun, out_dir: Path) -> Path:
    """Write ``eval_report_<timestamp>.md`` (plus the raw run as JSON) into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_report_{run.timestamp}.md"
    path.write_text(render_report(run))
    (out_dir / f"eval_run_{run.timestamp}.json").write_text(run.model_dump_json(indent=2))
    return path
