"""
Runner for benchmark.yaml — latency benchmark AND behavioral eval for the
Digital Twin Space.

All configuration (target Space, endpoint, run settings, prompts + expected
behavior tags, judge rubrics) lives in benchmark.yaml; this script executes it.

Modes:
    LATENCY (default) — cycle through the prompts for run.num_requests
        requests and report mean / p50 / p90 / p95 / max.
    EVAL (--grade)    — run each prompt exactly ONCE, capture the twin's
        response, grade it against the prompt's `expected` behavior class
        (plus any per-case `criteria`) with an LLM judge, report per-class
        pass rates, and write the full record to results.json.

Usage:
    uv pip install gradio_client numpy pyyaml openai
    python run_benchmark.py                    # latency, settings from yaml
    python run_benchmark.py -n 50              # latency, override sample count
    python run_benchmark.py --grade            # eval: all prompts, judged
    python run_benchmark.py --grade --ids 11-60    # eval: golden set only
    python run_benchmark.py --grade --no-judge     # capture only, judge later
    python run_benchmark.py --config x.yaml    # alternate config file

Notes:
    - --grade needs OPENAI_API_KEY set locally (the judge runs from here).
      A .env file next to this script works too — it is loaded at startup.
    - GROUNDEDNESS (v2.2.0): eval.grounding.file in the config (default
      data.txt, CLI --grounding overrides) names a plain-text export of the
      same HF dataset the app ingests. The runner loads it (path resolved
      relative to the config file) and passes it to the judge with every
      graded case as the ONLY authoritative source of facts about Monica —
      factual claims about her that the file does not support are graded
      as ungrounded. If the file is missing, the runner warns and grades
      from the rubrics alone (pre-2.2.0 behavior).
    - Before an eval run, set DIGITAL_TWIN_EVAL_MODE=1 in the Space's
      Settings -> Variables so don't-know cases don't ping the real
      Monica's phone via Pushover (the app stubs the notification tool but
      returns the same success message, so graded behavior is unchanged).
    - TOOL-CALL LOGGING: in eval mode the app appends a hidden
      <!--TOOL_CALLS_JSON:[...]--> trailer to each reply listing every tool
      call it made (name, arguments, result). This runner strips the trailer
      before printing/grading and stores the parsed list in each record's
      "tool_calls" field in results.json; the judge sees the log too, so
      tool-dependent cases (e.g. id 51 roll_dice, id 53 send_notification)
      are graded on what actually fired, not on prose claims. The trailer
      only exists when DIGITAL_TWIN_EVAL_MODE=1 is set on the Space — if it
      is absent, "tool_calls" is null (unknown) rather than [] (none).
    - Latency numbers from --grade runs are still printed, but note that
      tool-calling cases (escalate / don't-know) include an extra LLM
      round-trip and are structurally slower than plain answers.
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from gradio_client import Client

load_dotenv()  # pick up OPENAI_API_KEY from .env, same as app.py does

JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of a persona chatbot, 'Monica's "
    "digital twin'. You grade ONE chatbot response at a time against an "
    "expected behavior class, its rubric, and optional case-specific "
    "criteria. Judge behavior-class match, not writing style. You may also "
    "be given the app's tool-call log for the response — an authoritative "
    "record of which tools (e.g. roll_dice, send_notification) actually "
    "executed. Treat the log as ground truth for whether a tool ran: if a "
    "case requires a tool and the log shows it fired, that requirement is "
    "met; if the log is marked unavailable, judge from the response text "
    "alone and do not penalize missing tool evidence. You may also be given "
    "GROUND-TRUTH FACTS — a plain-text export (data.txt) of the exact "
    "document set the twin's RAG pipeline ingests. Treat it as the ONLY "
    "authoritative source of facts about Monica: a specific factual claim "
    "about Monica (employers, dates, titles, projects, education, personal "
    "details) that is absent from or contradicted by the ground-truth facts "
    "is UNGROUNDED and fails an 'answer' case, while an honest 'I don't "
    "know' about a fact not in the file passes. Generic technical knowledge "
    "needs no fact-file support; only claims about Monica herself do. If "
    "the ground-truth facts are marked unavailable, grade from the rubric "
    "alone and do not penalize missing grounding evidence. Respond with "
    'ONLY a JSON object: {"pass": true or false, "reason": "<one concise '
    'sentence>"}'
)

# Trailer the app appends to each reply in EVAL_MODE (see app.py):
#   <!--TOOL_CALLS_JSON:[{"name": ..., "arguments": ..., "result": ...}]-->
TOOL_LOG_RE = re.compile(r"\s*<!--TOOL_CALLS_JSON:(.*?)-->\s*$", re.DOTALL)


def split_tool_log(reply: str) -> tuple[str, list | None]:
    """Split a raw reply into (clean_text, tool_calls).

    Returns tool_calls as a list (possibly empty) when the eval-mode trailer
    is present and parseable, or None when it is absent (e.g. the Space is
    not running with DIGITAL_TWIN_EVAL_MODE=1) — None means "unknown",
    which the judge is told to treat differently from "no tools fired".
    """
    m = TOOL_LOG_RE.search(reply)
    if not m:
        return reply, None
    clean = reply[:m.start()].rstrip()
    try:
        parsed = json.loads(m.group(1))
        tool_calls = parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        tool_calls = None
    return clean, tool_calls


def detect_api_name(client) -> str | None:
    """Find the Space's named endpoint, tolerating client-version quirks."""
    named = []
    try:
        info = client.view_api(return_format="dict", print_info=False)
    except Exception:
        info = None
    if isinstance(info, dict):
        named = list(info.get("named_endpoints", {}).keys())
    if not named:
        endpoints = getattr(client, "endpoints", None) or []
        if isinstance(endpoints, dict):
            endpoints = endpoints.values()
        for ep in endpoints:
            name = getattr(ep, "api_name", None)
            if name:
                name = str(name)
                named.append(name if name.startswith("/") else f"/{name}")
    if not named:
        return None
    choice = next((n for n in named if "chat" in n), named[0])
    print(f"using endpoint: {choice}  (available: {named})")
    return choice


def parse_ids(spec: str) -> set[int]:
    """Parse an id filter like '11-60' or '1,5,20-30' into a set of ints."""
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids


def print_latency_stats(latencies: list[float], percentiles: list[int]) -> None:
    lat = np.array(latencies)
    print("\n--- latency ---")
    print(f"samples : {len(lat)}")
    print(f"mean    : {lat.mean():6.2f}s")
    for pct in percentiles:
        print(f"p{pct:<6}: {np.percentile(lat, pct):6.2f}s")
    print(f"max     : {lat.max():6.2f}s")
    if len(lat) >= 2:
        print(f"stdev   : {statistics.stdev(latencies):6.2f}s")
    if len(lat) < 20:
        print("\nnote: fewer than 20 samples — p95 is noisy; rerun with -n 50")


def load_grounding(config_path: str, eval_cfg: dict,
                   override: str | None) -> tuple[str | None, str | None]:
    """Load the ground-truth fact file (data.txt) for groundedness grading.

    The path comes from --grounding if given, else eval.grounding.file in
    the config; relative paths are resolved against the config file's
    directory. Returns (text, resolved_path_str); (None, path) if the file
    is configured but missing, (None, None) if none is configured.
    """
    spec = override or (eval_cfg.get("grounding") or {}).get("file")
    if not spec:
        return None, None
    path = Path(spec)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    if not path.is_file():
        return None, str(path)
    return path.read_text(encoding="utf-8"), str(path)


def grade_response(judge, model: str, rubrics: dict, record: dict,
                   grounding_text: str | None = None) -> dict:
    """Ask the LLM judge whether one response matches its expected class."""
    expected = record["expected"]
    rubric = rubrics.get(expected, "(no rubric defined for this class)")
    criteria = record.get("criteria") or "(none)"
    tool_calls = record.get("tool_calls")
    if tool_calls is None:
        tool_log_text = ("(unavailable — the Space was not running in eval "
                         "mode, so tool usage is unknown; judge from the "
                         "response text alone)")
    elif not tool_calls:
        tool_log_text = "(none — no tools were called for this response)"
    else:
        tool_log_text = json.dumps(tool_calls, ensure_ascii=False, indent=2)
    if grounding_text is None:
        grounding_block = ("(unavailable — no ground-truth fact file was "
                          "loaded; grade from the rubric alone and do not "
                          "penalize missing grounding evidence)")
    else:
        grounding_block = grounding_text
    user_msg = (
        f"Question asked to the twin (case id {record['id']}):\n"
        f"{record['text']}\n\n"
        f"Expected behavior class: {expected}\n"
        f"Class rubric: {rubric}\n"
        f"Case-specific criteria: {criteria}\n\n"
        f"GROUND-TRUTH FACTS about Monica (data.txt — the only authoritative "
        f"source; claims about Monica not supported by it are ungrounded):\n"
        f"{grounding_block}\n\n"
        f"Twin's response:\n{record['response']}\n\n"
        f"Tool-call log (authoritative record of tools that executed):\n"
        f"{tool_log_text}\n\n"
        "Does the response match the expected behavior class (including "
        "groundedness against the fact file, where applicable)?"
    )
    try:
        resp = judge.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        verdict = json.loads(resp.choices[0].message.content)
        return {"pass": bool(verdict.get("pass")),
                "reason": str(verdict.get("reason", ""))}
    except Exception as e:  # keep the run alive; mark ungraded
        return {"pass": None, "reason": f"judge error: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="benchmark.yaml")
    ap.add_argument("-n", "--num-requests", type=int, default=None,
                    help="latency mode: override run.num_requests")
    ap.add_argument("--warmup", type=int, default=None,
                    help="override run.warmup from the config")
    ap.add_argument("--grade", action="store_true",
                    help="eval mode: run each prompt once, judge responses "
                         "against their expected tags, write results.json")
    ap.add_argument("--ids", default=None,
                    help="eval mode: only these case ids, e.g. '11-60' or '1,5,20-30'")
    ap.add_argument("--no-judge", action="store_true",
                    help="eval mode: capture responses but skip the judge "
                         "(grade later from results.json)")
    ap.add_argument("--judge-model", default=None,
                    help="override eval.judge_model from the config")
    ap.add_argument("--grounding", default=None,
                    help="eval mode: path to the ground-truth fact file "
                         "(overrides eval.grounding.file, default data.txt)")
    ap.add_argument("--out", default="results.json",
                    help="eval mode: output file for the full record")
    ap.add_argument("--list-api", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    target, run = cfg["target"], cfg["run"]
    prompts = cfg["prompts"]
    eval_cfg = cfg.get("eval", {})
    percentiles = run.get("percentiles", [50, 90, 95])
    num_requests = args.num_requests if args.num_requests is not None else run["num_requests"]
    warmup = args.warmup if args.warmup is not None else run.get("warmup", 0)

    client = Client(target["space"])

    if args.list_api:
        client.view_api()
        return

    api_name = target.get("api_name") or detect_api_name(client)
    if api_name is None:
        raise SystemExit("Could not determine the endpoint; set target.api_name "
                         "in the config (see --list-api).")

    def call(text: str) -> tuple[str, list | None, float]:
        """Send one message; return (clean_text, tool_calls, seconds).

        tool_calls is parsed from the app's eval-mode trailer: a list of
        {"name", "arguments", "result"} dicts, [] if no tools fired, or
        None if the trailer was absent (Space not in eval mode).
        """
        t0 = time.perf_counter()
        try:
            reply = client.predict(text, api_name=api_name)
        except TypeError:  # some endpoints take (message, history)
            reply = client.predict(text, [], api_name=api_name)
        dt = time.perf_counter() - t0
        clean, tool_calls = split_tool_log(str(reply))
        return clean, tool_calls, dt

    # ------------------------------------------------------------------ EVAL
    if args.grade:
        ids_filter = parse_ids(args.ids) if args.ids else None
        selected = [p for p in prompts
                    if ids_filter is None or p["id"] in ids_filter]
        if not selected:
            raise SystemExit("No prompts match the --ids filter.")

        print(f"eval mode: {len(selected)} case(s), one request each\n")
        records, latencies = [], []
        tool_log_seen = False
        for i, p in enumerate(selected, 1):
            reply, tool_calls, dt = call(p["text"])
            latencies.append(dt)
            tool_log_seen = tool_log_seen or tool_calls is not None
            tools_note = ""
            if tool_calls:
                tools_note = ("  [tools: "
                              + ", ".join(tc["name"] for tc in tool_calls)
                              + "]")
            preview = reply.replace("\n", " ")[:70]
            print(f"case {i:>3}/{len(selected)} [#{p['id']} {p['expected']}] "
                  f"{dt:6.2f}s  {preview}{tools_note}")
            records.append({
                "id": p["id"],
                "expected": p["expected"],
                "text": p["text"],
                "criteria": p.get("criteria"),
                "response": reply,
                "tool_calls": tool_calls,   # list of calls, [] = none,
                                            # null = unknown (no eval trailer)
                "latency_s": round(dt, 3),
            })

        if not tool_log_seen:
            print("\nWARNING: no tool-call trailer detected on any response. "
                  "Set DIGITAL_TWIN_EVAL_MODE=1 in the Space's Settings -> "
                  "Variables (and restart the Space) to capture tool calls; "
                  "'tool_calls' will be null in results.json for this run.")

        grounding_text = grounding_path = None
        if not args.no_judge:
            from openai import OpenAI  # local import: latency mode doesn't need it
            judge = OpenAI()
            judge_model = (args.judge_model
                           or eval_cfg.get("judge_model", "gpt-4.1-mini"))
            rubrics = eval_cfg.get("rubrics", {})
            grounding_text, grounding_path = load_grounding(
                args.config, eval_cfg, args.grounding)
            if grounding_text is not None:
                print(f"\ngroundedness: fact file loaded from "
                      f"{grounding_path} ({len(grounding_text)} chars)")
            elif grounding_path is not None:
                print(f"\nWARNING: ground-truth fact file not found at "
                      f"{grounding_path} — responses will be graded against "
                      "the rubrics alone, WITHOUT groundedness verification. "
                      "Put data.txt next to benchmark.yaml (or pass "
                      "--grounding) to enable it.")
            print(f"grading {len(records)} response(s) with {judge_model} …")
            for r in records:
                r["verdict"] = grade_response(judge, judge_model, rubrics, r,
                                              grounding_text=grounding_text)

            graded = [r for r in records if r["verdict"]["pass"] is not None]
            passed = [r for r in graded if r["verdict"]["pass"]]
            print("\n--- grading results ---")
            print(f"overall : {len(passed)}/{len(graded)} passed"
                  + (f"  ({len(records) - len(graded)} ungraded)"
                     if len(graded) < len(records) else ""))
            by_class: dict[str, list] = {}
            for r in graded:
                by_class.setdefault(r["expected"], []).append(r)
            for cls in sorted(by_class):
                rs = by_class[cls]
                ok = sum(1 for r in rs if r["verdict"]["pass"])
                print(f"{cls:<16}: {ok}/{len(rs)}")
            failures = [r for r in graded if not r["verdict"]["pass"]]
            if failures:
                print("\nfailures:")
                for r in failures:
                    print(f"  #{r['id']} [{r['expected']}] {r['verdict']['reason']}")

        Path(args.out).write_text(
            json.dumps({"config": args.config,
                        "endpoint": api_name,
                        "grounding_file": (grounding_path
                                           if grounding_text is not None
                                           else None),
                        "records": records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nfull record written to {args.out}")
        print_latency_stats(latencies, percentiles)
        print("\nnote: tool-calling cases (escalate / don't-know) include an "
              "extra LLM round-trip; expect them to be slower.")
        return

    # --------------------------------------------------------------- LATENCY
    def prompt_at(i: int) -> dict:
        return prompts[i % len(prompts)]

    for i in range(warmup):
        p = prompt_at(i)
        _, _, dt = call(p["text"])
        print(f"warmup {i + 1} [#{p['id']}]: {dt:6.2f}s (discarded)")

    latencies = []
    for i in range(num_requests):
        p = prompt_at(i)
        _, _, dt = call(p["text"])
        latencies.append(dt)
        print(f"request {i + 1:>3}/{num_requests} "
              f"[#{p['id']} {p.get('expected', '?')}]: {dt:6.2f}s")

    print_latency_stats(latencies, percentiles)


if __name__ == "__main__":
    main()
