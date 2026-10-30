"""Eval coverage: cases per role theme against role demand, plus the Wealthsimple
R1-R15 view. Plan item A.5. Free and read-only: reads eval_set.json, calls nothing,
never writes to eval_set.json.

    python coverage_report.py            # print the tables
    python coverage_report.py --write    # also write COVERAGE.md (tables + Mermaid charts)

Role demand is `requirements.themes.<id>.roles_of_26` in eval_set.json, Jord's
approximate counts from 26 saved PM roles ("~15" is read as 15, "4+" as 4). No other
weight is applied. A case tagged with two themes counts once in each, so shares are
of theme tags, not of cases. Multi-turn cases (M01-M07) carry themes but no req, so
they appear in the theme table as their own column and not in the R1-R15 view.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_SET = HERE / "eval_set.json"
OUT_MD = HERE / "COVERAGE.md"
CASE_GROUPS = ("questions", "additive_cases", "theme_cases")


def load(path=EVAL_SET):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [c for g in CASE_GROUPS for c in spec.get(g, [])]
    multiturn = spec.get("multiturn", {}).get("cases", [])
    return spec, cases, multiturn


def parse_roles(label):
    """'~15' -> 15, '4+' -> 4, '3' -> 3. Raises on anything without a number."""
    m = re.search(r"\d+", str(label))
    if not m:
        raise ValueError(f"roles_of_26 has no number: {label!r}")
    return int(m.group())


def theme_rows(spec, cases, multiturn):
    themes = spec["requirements"]["themes"]
    ids = [t for t in themes if t != "note"]
    rows = []
    for t in ids:
        single = [c["id"] for c in cases if t in (c.get("themes") or [])]
        multi = [c["id"] for c in multiturn if t in (c.get("themes") or [])]
        rows.append({
            "theme": t,
            "name": themes[t]["name"],
            "roles_label": themes[t]["roles_of_26"],
            "roles": parse_roles(themes[t]["roles_of_26"]),
            "single": single,
            "multi": multi,
            "total": len(single) + len(multi),
        })
    role_sum = sum(r["roles"] for r in rows)
    tag_sum = sum(r["total"] for r in rows)
    for r in rows:
        r["demand_share"] = r["roles"] / role_sum
        r["case_share"] = r["total"] / tag_sum if tag_sum else 0.0
        r["per_role"] = r["total"] / r["roles"]
        r["gap_pts"] = (r["case_share"] - r["demand_share"]) * 100
    rows.sort(key=lambda r: (-r["roles"], r["theme"]))
    return rows, role_sum, tag_sum


def req_rows(spec, cases):
    reqs = {k: v for k, v in spec["requirements"]["wealthsimple"].items() if re.fullmatch(r"R\d+", k)}
    rows = []
    for rid in sorted(reqs, key=lambda k: int(k[1:])):
        hit = [c["id"] for c in cases if rid in (c.get("req") or [])]
        rows.append({"req": rid, "text": reqs[rid], "cases": hit})
    return rows


def untagged(cases, multiturn):
    return {
        "no_theme": [c["id"] for c in cases if not c.get("themes")],
        "no_theme_multiturn": [c["id"] for c in multiturn if not c.get("themes")],
        "no_req": [c["id"] for c in cases if not c.get("req")],
    }


def flag_theme(r):
    return "UNDER" if r["gap_pts"] < 0 else ""


def flag_req(r):
    return "THIN" if len(r["cases"]) <= 1 else ""


def short(text, n=58):
    return text if len(text) <= n else text[: n - 3] + "..."


def print_report(spec, cases, multiturn):
    rows, role_sum, tag_sum = theme_rows(spec, cases, multiturn)
    print(f"Eval coverage from {EVAL_SET.name}: {len(cases)} single-turn cases, {len(multiturn)} multi-turn cases.")
    print(f"Role demand: roles_of_26 (approximate). Sum of role counts {role_sum}; sum of theme tags {tag_sum}.")
    print("\n--- Cases per theme against role demand ---")
    hdr = f"{'theme':<7} {'name':<27} {'roles':>5} {'demand%':>7} {'cases':>5} {'M':>2} {'total':>5} {'cases%':>6} {'gap pts':>7} {'per role':>8}  flag"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['theme']:<7} {r['name']:<27} {r['roles_label']:>5} {r['demand_share']*100:>7.1f} "
              f"{len(r['single']):>5} {len(r['multi']):>2} {r['total']:>5} {r['case_share']*100:>6.1f} "
              f"{r['gap_pts']:>+7.1f} {r['per_role']:>8.2f}  {flag_theme(r)}")
    print("UNDER = share of cases below share of role demand.")
    for r in rows:
        ids = r["single"] + [f"{m}(M)" for m in r["multi"]]
        print(f"  {r['theme']:<7} {', '.join(ids) or '-'}")

    print("\n--- Wealthsimple R1-R15 ---")
    for r in req_rows(spec, cases):
        print(f"{r['req']:<4} {short(r['text']):<58} {len(r['cases']):>2}  {flag_req(r):<4}  {', '.join(r['cases']) or '-'}")
    print("THIN = one case or none. M01-M07 carry no req and are not counted here.")

    u = untagged(cases, multiturn)
    print("\n--- Untagged ---")
    print(f"No theme ({len(u['no_theme'])}): {', '.join(u['no_theme'])}")
    print(f"No theme, multi-turn ({len(u['no_theme_multiturn'])}): {', '.join(u['no_theme_multiturn'])}")
    print(f"No req ({len(u['no_req'])}): {', '.join(u['no_req'])}")


def markdown(spec, cases, multiturn):
    rows, _role_sum, tag_sum = theme_rows(spec, cases, multiturn)
    reqs = req_rows(spec, cases)
    u = untagged(cases, multiturn)
    out = [
        "# Eval coverage",
        "",
        "Generated by `python coverage_report.py --write` from `eval_set.json`. Do not edit by hand.",
        "",
        (f"{len(cases)} single-turn cases and {len(multiturn)} multi-turn cases. Two tracks: the role themes "
        "from 26 saved PM roles, and the Wealthsimple Staff PM requirements R1–R15. A case can carry both."),
        "",
        "## Cases per theme against role demand",
        "",
        ("Role demand is how many of the 26 saved roles ask for the theme. The counts are approximate "
        "(`roles_of_26` in `eval_set.json`). A case tagged with two themes counts in both, so shares are "
        f"of theme tags ({tag_sum}), not of cases. **Under** means the theme's share of cases is below its "
        "share of role demand."),
        "",
        "| Theme | Name | Roles of 26 | Demand share | Cases | Multi-turn | Total | Case share | Gap (pts) | Cases per role | |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['theme']} | {r['name']} | {r['roles_label']} | {r['demand_share']*100:.1f}% | {len(r['single'])} | "
            f"{len(r['multi'])} | {r['total']} | {r['case_share']*100:.1f}% | {r['gap_pts']:+.1f} | {r['per_role']:.2f} | "
            f"{'**Under**' if flag_theme(r) else ''} |"
        )
    labels = ", ".join(f'"{r["theme"]}"' for r in rows)
    demand = ", ".join(f"{r['demand_share']*100:.1f}" for r in rows)
    share = ", ".join(f"{r['case_share']*100:.1f}" for r in rows)
    top = max(max(r["demand_share"], r["case_share"]) for r in rows) * 100
    ymax = int(top // 5 + 1) * 5
    out += [
        "",
        ("Bars: share of role demand (%). Line: share of eval cases (%). Where the line sits below the bar, "
        "the theme has fewer cases than its demand."),
        "",
        "```mermaid",
        "xychart-beta",
        '    title "Role demand vs eval cases, by theme (%)"',
        f"    x-axis [{labels}]",
        f'    y-axis "Share (%)" 0 --> {ymax}',
        f"    bar [{demand}]",
        f"    line [{share}]",
        "```",
        "",
        "Cases per theme:",
        "",
    ]
    for r in rows:
        ids = r["single"] + [f"{m} (multi-turn)" for m in r["multi"]]
        out.append(f"- **{r['theme']}**: {', '.join(ids) or 'none'}")

    out += [
        "",
        "## Wealthsimple R1–R15",
        "",
        ("Single-turn cases per requirement. **Thin** means one case or none. Multi-turn cases carry no "
        "requirement tag and are not counted here."),
        "",
        "| Req | Requirement | Cases | Case ids | |",
        "|---|---|--:|---|---|",
    ]
    for r in reqs:
        out.append(f"| {r['req']} | {r['text']} | {len(r['cases'])} | {', '.join(r['cases']) or '–'} | "
                   f"{'**Thin**' if flag_req(r) else ''} |")
    rmax = max(len(r["cases"]) for r in reqs) + 1
    out += [
        "",
        "```mermaid",
        "xychart-beta",
        '    title "Eval cases per Wealthsimple requirement"',
        f"    x-axis [{', '.join(r['req'] for r in reqs)}]",
        f'    y-axis "Cases" 0 --> {rmax}',
        f"    bar [{', '.join(str(len(r['cases'])) for r in reqs)}]",
        "```",
        "",
        "## Untagged cases",
        "",
        ("These are not gaps by themselves. Adversarial, consistency and narrative cases test behaviour, "
        "not a theme or a requirement."),
        "",
        f"- No theme ({len(u['no_theme'])}): {', '.join(u['no_theme'])}",
        f"- No theme, multi-turn ({len(u['no_theme_multiturn'])}): {', '.join(u['no_theme_multiturn'])}",
        f"- No requirement ({len(u['no_req'])}): {', '.join(u['no_req'])}",
        "",
    ]
    return "\n".join(out)


def main(argv):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    spec, cases, multiturn = load()
    print_report(spec, cases, multiturn)
    if "--write" in argv:
        OUT_MD.write_text(markdown(spec, cases, multiturn), encoding="utf-8", newline="\n")
        print(f"\nWrote {OUT_MD.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
