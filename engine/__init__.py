"""jordbot's answer engine: everything that decides what an answer says.

The fixed replies, the follow-up query rewrite, retrieval and its layout, the
system prompt and the answer model call. The eval runners import it. The site
still runs its own TypeScript copy (jordannedyck-ai-web lib/ and route.ts)
until it calls this engine; until then the drift checks in run_eval.py keep
the two in step.
"""
