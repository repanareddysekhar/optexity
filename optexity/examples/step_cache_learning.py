"""Iterative cache-learning demo for the Optexity take-home assignment.

Default seed workflow (``test_automation_iteration1.json``): Sauce Demo checkout —
login → add products → cart → multi-page checkout → order confirmation.

File roles:
- ``test_automation_iteration1.json`` — iteration 1 agentic seed (LLM explores)
- ``test_automation_cached.json`` — learned deterministic replay from cache
- ``test_automation.json`` — local ``/inference`` entrypoint (synced to cached output)

Workflow (``demo`` / ``iterate``):
1. **Iteration 1 (agentic)** — run ``test_automation_iteration1.json``; browser-use
   explores with the LLM and writes ``step_cache.json`` under task logs.
2. **Copy** — promote ``logs/step_*/cached_automation.json`` into the repo:
   ``cache_iterations/iteration_1_cached.json`` and ``test_automation_cached.json``.
3. **Iteration 2+ (cached)** — replay the copied deterministic automation (no LLM).

Usage (from repo root, with env activated):
    python -m optexity.examples.step_cache_learning demo
    python -m optexity.examples.step_cache_learning iterate --iterations 2
    python -m optexity.examples.step_cache_learning agentic
    python -m optexity.examples.step_cache_learning cached
    python -m optexity.examples.step_cache_learning build-cache /path/to/step_cache.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

_env_path = os.getenv("ENV_PATH")
if _env_path:
    load_dotenv(_env_path)
else:
    load_dotenv(REPO_ROOT / ".env")

from optexity.inference.core.run_automation import run_automation
from optexity.inference.infra.actual_browser import ActualBrowser
from optexity.learning.step_cache import build_optexity_automation
from optexity.schema.automation import Automation
from optexity.schema.task import Task
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LOCAL_INFERENCE_AUTOMATION = REPO_ROOT / "test_automation.json"
CACHED_AUTOMATION = REPO_ROOT / "test_automation_cached.json"
ITERATION1_AUTOMATION = REPO_ROOT / "test_automation_iteration1.json"
ITERATION_OUTPUT_DIR = REPO_ROOT / "cache_iterations"


def _require_google_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is required for agentic browser-use tasks. "
            "Export it in your shell or add it to "
            f"{REPO_ROOT / '.env'}."
        )


def _load_automation(path: Path) -> Automation:
    with open(path, encoding="utf-8") as f:
        return Automation.model_validate(json.load(f))


def _build_task(automation: Automation) -> Task:
    return Task(
        task_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        recording_id=str(uuid.uuid4()),
        endpoint_name="local_step_cache_learning",
        automation=automation,
        input_parameters=dict(automation.parameters.input_parameters),
        secure_parameters=dict(automation.parameters.secure_parameters),
        unique_parameter_names=[],
        created_at=datetime.now(timezone.utc),
        status="queued",
        api_key=settings.OPTEXITY_API_KEY,
        company_id="local-dev-company",
        local_test_override=True,
    )


async def _run_task(task: Task) -> float:
    unique_child_arn = f"step-cache-{task.task_id[:8]}"
    child_process_id = int(os.getenv("OPTEXITY_CHILD_PROCESS_ID", "0"))
    actual_browser = ActualBrowser(
        channel=task.automation.browser_channel,
        unique_child_arn=unique_child_arn,
        port=9222 + child_process_id,
        headless=os.getenv("OPTEXITY_HEADLESS", "false").lower() == "true",
        is_dedicated=False,
        use_proxy=task.use_proxy,
        os_emulation=task.automation.os_emulation,
        allow_cookies=task.automation.allow_cookies,
    )
    start = perf_counter()
    try:
        await actual_browser.start()
        if actual_browser.cdp_url is None:
            raise RuntimeError("Browser started but CDP URL is missing")
        await run_automation(
            task=task,
            unique_child_arn=unique_child_arn,
            child_process_id=child_process_id,
            cdp_url=actual_browser.cdp_url,
        )
    finally:
        await actual_browser.stop(graceful=True)
    return perf_counter() - start


def _latest_cached_automation_path(task: Task) -> Path | None:
    if not task.logs_directory.exists():
        return None
    step_dirs = sorted(task.logs_directory.glob("step_*"))
    for step_dir in reversed(step_dirs):
        candidate = step_dir / "cached_automation.json"
        if candidate.exists():
            return candidate
    return None


def _cached_node_count(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return len(json.load(f).get("nodes", []))


def _copy_cached_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    nodes = _cached_node_count(destination)
    logger.info(
        "Copied cached automation (%d nodes): %s -> %s",
        nodes,
        source,
        destination,
    )
    return destination


def _promote_agentic_cache(
    cache_in_logs: Path,
    output_dir: Path,
) -> Path:
    """Copy task-log cache into repo artifacts used for iteration 2 replay."""
    iteration_copy = output_dir / "iteration_1_cached.json"
    _copy_cached_file(cache_in_logs, iteration_copy)
    _copy_cached_file(iteration_copy, CACHED_AUTOMATION)
    _copy_cached_file(iteration_copy, LOCAL_INFERENCE_AUTOMATION)
    return CACHED_AUTOMATION


async def run_agentic() -> None:
    _require_google_api_key()
    logger.info("Iteration 1 (agentic/LLM): running %s", ITERATION1_AUTOMATION)
    task = _build_task(_load_automation(ITERATION1_AUTOMATION))
    elapsed = await _run_task(task)
    cache_in_logs = _latest_cached_automation_path(task)
    if cache_in_logs and _cached_node_count(cache_in_logs) > 0:
        _promote_agentic_cache(cache_in_logs, ITERATION_OUTPUT_DIR)
    else:
        logger.warning(
            "No cached automation in %s — check GOOGLE_API_KEY and task logs",
            task.logs_directory,
        )
    logger.info("Done in %.2fs. Logs: %s", elapsed, task.logs_directory)


async def run_cached() -> None:
    if not CACHED_AUTOMATION.exists():
        raise FileNotFoundError(
            f"{CACHED_AUTOMATION} not found. Run `demo` or `agentic` first to copy "
            "cached_automation.json from task logs."
        )
    logger.info("Cached replay (no LLM): running %s", CACHED_AUTOMATION)
    task = _build_task(_load_automation(CACHED_AUTOMATION))
    elapsed = await _run_task(task)
    logger.info("Done in %.2fs", elapsed)


def build_cached_automation(cache_path: Path, output_path: Path | None = None) -> Path:
    output = output_path or CACHED_AUTOMATION
    with open(cache_path, encoding="utf-8") as f:
        cache_data = json.load(f)
    automation = build_optexity_automation(cache_data)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(automation, f, indent=2)
    logger.info("Wrote %d node(s) to %s", len(automation.get("nodes", [])), output)
    return output


async def run_iterative_learning(
    iterations: int,
    seed_automation_path: Path,
    output_dir: Path,
) -> None:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    cached_automation_path: Path | None = None
    run_summary: list[dict[str, str | int | float | None]] = []

    logger.info(
        "Starting cache-learning demo: iterations=%d (1=agentic LLM, 2+=cached replay)",
        iterations,
    )

    for i in range(1, iterations + 1):
        mode: Literal["agentic", "cached"] = "agentic" if i == 1 else "cached"
        if mode == "agentic":
            _require_google_api_key()
            input_automation_path = seed_automation_path
        else:
            if cached_automation_path is None:
                raise RuntimeError(
                    "Iteration 1 did not produce a cached automation to replay. "
                    "Ensure GOOGLE_API_KEY is set and browser-use wrote step_cache.json."
                )
            input_automation_path = cached_automation_path

        logger.info(
            "Iteration %d/%d [%s]: running %s",
            i,
            iterations,
            mode,
            input_automation_path,
        )
        task = _build_task(_load_automation(input_automation_path))
        elapsed_seconds = await _run_task(task)

        copy_source: str | None = None
        copy_targets: list[str] = []
        nodes_count: int | None = None

        if mode == "agentic":
            cache_in_logs = _latest_cached_automation_path(task)
            if cache_in_logs is None:
                logger.warning(
                    "Iteration %d: no cached_automation.json under %s",
                    i,
                    task.logs_directory,
                )
            else:
                nodes_count = _cached_node_count(cache_in_logs)
                if nodes_count == 0:
                    logger.warning(
                        "Iteration %d: cached automation is empty (LLM likely failed)",
                        i,
                    )
                else:
                    copy_source = str(cache_in_logs)
                    cached_automation_path = _promote_agentic_cache(
                        cache_in_logs, output_dir
                    )
                    copy_targets = [
                        str(output_dir / "iteration_1_cached.json"),
                        str(CACHED_AUTOMATION),
                        str(LOCAL_INFERENCE_AUTOMATION),
                    ]

        run_summary.append(
            {
                "iteration": i,
                "mode": mode,
                "input_automation": str(input_automation_path),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "cached_nodes_count": nodes_count,
                "copy_source": copy_source,
                "copy_targets": copy_targets or None,
                "task_logs_directory": str(task.logs_directory),
            }
        )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    logger.info("Demo complete. Summary: %s", summary_path)


async def run_demo() -> None:
    await run_iterative_learning(
        iterations=2,
        seed_automation_path=ITERATION1_AUTOMATION,
        output_dir=ITERATION_OUTPUT_DIR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step cache learning demo")
    parser.add_argument(
        "command",
        choices=["demo", "agentic", "cached", "build-cache", "iterate"],
        help=(
            "demo=2-step LLM then cached replay; agentic=iteration 1 only; "
            "cached=replay test_automation_cached.json; iterate=custom loop"
        ),
    )
    parser.add_argument("cache_path", nargs="?", help="Path to step_cache.json for build-cache")
    parser.add_argument("--output", help="Output path for build-cache")
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Iterations for iterate (1=agentic LLM, 2+=cached replay). Default: 2.",
    )
    parser.add_argument(
        "--seed",
        default=str(ITERATION1_AUTOMATION),
        help="Agentic seed automation for iteration 1.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ITERATION_OUTPUT_DIR),
        help="Directory for iteration_1_cached.json and summary.json.",
    )
    args = parser.parse_args()

    if args.command == "demo":
        asyncio.run(run_demo())
    elif args.command == "agentic":
        asyncio.run(run_agentic())
    elif args.command == "cached":
        asyncio.run(run_cached())
    elif args.command == "build-cache":
        if not args.cache_path:
            parser.error("build-cache requires a path to step_cache.json")
        build_cached_automation(Path(args.cache_path), Path(args.output) if args.output else None)
    else:
        if args.iterations < 1:
            parser.error("--iterations must be >= 1")
        asyncio.run(
            run_iterative_learning(
                iterations=args.iterations,
                seed_automation_path=Path(args.seed),
                output_dir=Path(args.output_dir),
            )
        )


if __name__ == "__main__":
    main()
