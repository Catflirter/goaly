"""Effi-SQL's capability pack: the v0 baseline (spec §5.4).

- prompt.md: the system prompt.
- skills/*.md: the skill guides, inlined in prompt.md, which is the only way they reach the
  executor: pack code may open no file.
- config.yaml: sampling, budgets and the context window, which the harness factory gives core,
  and the pack's own section, which build_policy validates.
- tools.py: execute_sql and submit.
- policy.py: the policy, which runs core's agent loop with those tools and falls back to
  ``base_sql`` unchanged when a budget runs out.
"""
