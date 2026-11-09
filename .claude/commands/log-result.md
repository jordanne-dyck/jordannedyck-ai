---
description: Write a results-log entry for a run, probe, check or scoring round, marked unreviewed for Jord's batch review
argument-hint: [result file path(s) or short description] | --backlog
---

Log a result in the human results log: `nimbalyst-local/results-log.md` (local, not in git).
If that file does not exist, say so and stop.

Input: $ARGUMENTS

If the input is `--backlog`, run `py -3 .claude/hooks/unlogged_results.py --backlog`, print the
list it gives, and ask which items to log. Do nothing else until there is an answer.

Otherwise:

1. Read the "How to write an entry" section of the log and follow it exactly. It is the
   format; this command does not restate it. Every entry needs `summary`, `tags` and
   `hypothesis`. Write the question so a reader who was not in the session can follow it.
   Take the hypothesis from what the plan or session record says led to the test, and mark
   whether it was recorded before or inferred after. Never write one to fit the result.
2. Open every source file named in the input. If the input is a description with no file,
   find the file(s) it came from (`eval_runs/`, `nimbalyst-local/`, `eval_history/history.jsonl`)
   and confirm them with Jord before drafting. Never draft from a summary or from memory.
3. Read every number at its source and mark it **[read]**. A number found only in a plan,
   a summary or a STATE block is marked **[stated, not re-checked]**. Leave gaps as gaps.
   Compute ranges with the repo's own scripts (`bootstrap_ci.py`, Wilson on the run's counts),
   never by hand.
4. Pick the id: today's date, numbered after the last `R-YYYYMMDD-NN` with that date in the log.
   Never reuse or renumber an id.
5. `source`: every file the result was read from, by repo-relative path, prefixed `ai:` or
   `web:`. The capture hook matches files against these paths, so a missing path means the
   file keeps being flagged.
6. `status` is `proposed` unless EVALS.md already holds the result (promoted) or a plan
   Decided line or Jord closed the idea (declined, with the reason). Check the `decided`
   list before suggesting any next step, and do not re-propose a closed item.
7. Add `- reviewed: no` as the entry's last field and append it at the bottom of the log, in
   date order, without waiting for approval. One entry per result. Tell Jord the id and
   summary in one line; she reviews `reviewed: no` entries in batch (G.6) and flips them to
   `yes`. A file that is not a result goes under `## Not results` in the log as `ai:<path>`
   with a one-line reason, so the capture hook stops flagging it.

Free: this command makes no paid API calls. If logging a result seems to need a new paid run,
stop and ask.
