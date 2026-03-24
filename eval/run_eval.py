"""
RAG Evaluation Runner for Jordanne Dyck AI Resume System

Usage:
  python eval/run_eval.py                    # Retrieval-only eval (fast, no LLM cost)
  python eval/run_eval.py --full             # Full eval including LLM response quality
  python eval/run_eval.py --full --verbose   # Full eval with detailed output
  python eval/run_eval.py --id ai-01         # Run a single test case
"""

import json
import re
import argparse
import requests
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

SEARCH_API = "http://localhost:5000/search"
CHAT_API = "http://localhost:3000/api/chat"

# ─── Colors for terminal output ───
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def load_eval_set():
    eval_path = Path(__file__).parent / "eval_set.json"
    with open(eval_path) as f:
        return json.load(f)


def test_retrieval(query: str, n_results: int = 5) -> dict:
    """Test the retrieval pipeline (search API only)."""
    try:
        resp = requests.post(SEARCH_API, json={"query": query, "n_results": n_results}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {
            "success": True,
            "results": data.get("results", []),
            "content": "\n".join(r["content"] for r in data.get("results", [])),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "results": [], "content": ""}


def test_chat(query: str) -> dict:
    """Test the full chat pipeline (retrieval + LLM response)."""
    try:
        resp = requests.post(
            CHAT_API,
            json={"messages": [{"role": "user", "content": query}]},
            timeout=60,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return {"success": True, "response": resp.text}
    except Exception as e:
        return {"success": False, "error": str(e), "response": ""}


def parse_sse_response(raw: str) -> str:
    """Parse SSE streaming tokens (0:"token" format) into plain text."""
    tokens = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Match pattern like 0:"text"
        match = re.match(r'^\d+:"(.*)"$', line)
        if match:
            tokens.append(match.group(1))
    return "".join(tokens) if tokens else raw


def eval_retrieval_criteria(case: dict, retrieval: dict) -> list:
    """Check retrieval_must_include against retrieved content."""
    results = []
    content_lower = retrieval["content"].lower()

    for term in case.get("retrieval_must_include", []):
        found = term.lower() in content_lower
        results.append({
            "type": "retrieval",
            "criterion": f"Retrieved content contains '{term}'",
            "passed": found,
        })

    return results


def eval_response_criteria(case: dict, response_text: str) -> list:
    """Check response_must_include and response_must_not_include."""
    results = []
    text_lower = response_text.lower()

    for term in case.get("response_must_include", []):
        found = term.lower() in text_lower
        results.append({
            "type": "response_include",
            "criterion": f"Response contains '{term}'",
            "passed": found,
        })

    for term in case.get("response_must_not_include", []):
        found = term.lower() in text_lower
        results.append({
            "type": "response_exclude",
            "criterion": f"Response does NOT contain '{term}'",
            "passed": not found,
        })

    return results


def eval_with_llm(case: dict, response_text: str) -> list:
    """Use GPT-4o-mini to score qualitative response_criteria."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    criteria = case.get("response_criteria", [])
    if not criteria:
        return []

    criteria_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))

    prompt = f"""You are evaluating an AI assistant's response about a job candidate named Jordanne Dyck.

QUERY: {case['query']}

RESPONSE:
{response_text}

CRITERIA TO EVALUATE:
{criteria_list}

For each criterion, respond with ONLY a JSON array of objects, each with:
- "criterion_index": (1-based number)
- "passed": true/false
- "reason": brief explanation (1 sentence max)

Return ONLY the JSON array, no other text."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=30,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        scores = json.loads(raw)

        results = []
        for score in scores:
            idx = score["criterion_index"] - 1
            results.append({
                "type": "llm_eval",
                "criterion": criteria[idx] if idx < len(criteria) else "unknown",
                "passed": score["passed"],
                "reason": score.get("reason", ""),
            })
        return results
    except Exception as e:
        return [{"type": "llm_eval", "criterion": "LLM evaluation", "passed": False, "reason": f"Eval error: {e}"}]


def safe_print(text):
    """Print text, replacing chars that can't encode on Windows cp1252."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def print_case_result(case: dict, checks: list, response_text: str = None, verbose: bool = False):
    """Print results for a single eval case."""
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    all_passed = passed == total

    status = f"{C.GREEN}PASS{C.END}" if all_passed else f"{C.RED}FAIL{C.END}"
    safe_print(f"\n{C.BOLD}[{case['id']}]{C.END} {status} ({passed}/{total}) -{C.DIM}{case['query']}{C.END}")

    for check in checks:
        icon = f"{C.GREEN}+{C.END}" if check["passed"] else f"{C.RED}x{C.END}"
        detail = f" -{check['reason']}" if check.get("reason") else ""
        safe_print(f"  {icon} {check['criterion']}{C.DIM}{detail}{C.END}")

    if verbose and response_text:
        safe_print(f"\n  {C.CYAN}Response preview:{C.END}")
        preview = response_text[:500].replace("\n", "\n  ")
        safe_print(f"  {C.DIM}{preview}...{C.END}")


def run_eval(args):
    eval_set = load_eval_set()
    cases = eval_set["eval_cases"]

    # Filter to single case if --id specified
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            print(f"{C.RED}No eval case found with id '{args.id}'{C.END}")
            return

    print(f"\n{C.BOLD}{'=' * 60}{C.END}")
    print(f"{C.BOLD}  RAG Evaluation - {len(cases)} test cases{C.END}")
    print(f"  Mode: {'Full (retrieval + LLM response + criteria)' if args.full else 'Retrieval only'}")
    print(f"{C.BOLD}{'=' * 60}{C.END}")

    all_checks = []
    category_stats = {}

    for case in cases:
        checks = []

        # 1. Test retrieval
        retrieval = test_retrieval(case["query"])
        if not retrieval["success"]:
            print(f"\n{C.RED}[{case['id']}] Retrieval FAILED: {retrieval.get('error')}{C.END}")
            checks.append({"type": "retrieval", "criterion": "Search API reachable", "passed": False})
        else:
            checks.extend(eval_retrieval_criteria(case, retrieval))

        response_text = None

        if args.full:
            # 2. Test full chat response
            chat = test_chat(case["query"])
            if not chat["success"]:
                print(f"\n{C.RED}[{case['id']}] Chat FAILED: {chat.get('error')}{C.END}")
                checks.append({"type": "chat", "criterion": "Chat API reachable", "passed": False})
            else:
                response_text = parse_sse_response(chat["response"])
                # 3. Check must_include / must_not_include
                checks.extend(eval_response_criteria(case, response_text))
                # 4. LLM-based qualitative eval
                checks.extend(eval_with_llm(case, response_text))

        print_case_result(case, checks, response_text, args.verbose)

        # Track stats
        cat = case["category"]
        if cat not in category_stats:
            category_stats[cat] = {"passed": 0, "total": 0}
        for c in checks:
            category_stats[cat]["total"] += 1
            if c["passed"]:
                category_stats[cat]["passed"] += 1

        all_checks.extend(checks)

    # Summary
    total_passed = sum(1 for c in all_checks if c["passed"])
    total_checks = len(all_checks)
    pct = (total_passed / total_checks * 100) if total_checks else 0

    print(f"\n{C.BOLD}{'=' * 60}{C.END}")
    print(f"{C.BOLD}  SUMMARY{C.END}")
    print(f"{'-' * 60}")
    print(f"  Overall: {total_passed}/{total_checks} checks passed ({pct:.0f}%)")
    print(f"\n  By category:")
    for cat, stats in sorted(category_stats.items()):
        cat_pct = (stats["passed"] / stats["total"] * 100) if stats["total"] else 0
        bar = f"{C.GREEN}{'#' * int(cat_pct // 10)}{C.DIM}{'.' * (10 - int(cat_pct // 10))}{C.END}"
        print(f"    {cat:<20} {bar} {stats['passed']}/{stats['total']} ({cat_pct:.0f}%)")
    print(f"{C.BOLD}{'=' * 60}{C.END}\n")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "full" if args.full else "retrieval"
    results_file = results_dir / f"eval_{mode}_{timestamp}.json"

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "summary": {"passed": total_passed, "total": total_checks, "percentage": round(pct, 1)},
        "category_stats": category_stats,
        "cases": [
            {
                "id": case["id"],
                "query": case["query"],
                "checks": [c for c in all_checks if True],  # all checks saved
            }
            for case in cases
        ],
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved to: {results_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--full", action="store_true", help="Run full eval (retrieval + LLM response)")
    parser.add_argument("--verbose", action="store_true", help="Show response previews")
    parser.add_argument("--id", type=str, help="Run a single test case by ID")
    args = parser.parse_args()
    run_eval(args)
