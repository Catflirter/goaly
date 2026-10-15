"""The guards a candidate must pass before it is evaluated (spec §8.2 step 3, §8.5).

Each guard checks a candidate's commit, through git or a read-only snapshot of it
(goaly.rsi.workspace.Workspace.snapshot), never its live worktree, and returns a GuardResult.
"""
