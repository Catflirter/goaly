# Idea bank: Effi-SQL (TPC-H on PostgreSQL 16)

Techniques that prior work and PostgreSQL practice found effective. They are for reference only:
the analyst decides whether a failure calls for any of them, and how a pack change would teach or
use it. None of them is a rule for the executor.

## SQL rewrites the executor could be taught

1. **Decorrelate subqueries.** A correlated scalar subquery in SELECT or WHERE that runs once
   per outer row becomes a join against one pre-aggregated derived table or CTE (`GROUP BY` the
   correlation key).
2. **EXISTS versus IN.** `IN (SELECT ...)` over a large or NULL-able set, and `NOT IN` above all
   (which also changes results when the set holds NULL), often plans worse than `EXISTS` /
   `NOT EXISTS`, which PostgreSQL turns into semi- and anti-joins.
3. **Push predicates down.** Filter before joining or aggregating: move a WHERE on base columns
   inside the derived table, CTE or UNION branch that feeds it, and filter on the column itself
   rather than an expression of it.
4. **Drop needless work.** Remove a `DISTINCT` that a key already guarantees, an `ORDER BY`
   inside a subquery, `SELECT *` in a CTE that is joined later, and columns nothing reads.
5. **CTE materialization.** PostgreSQL inlines a CTE referenced once; one referenced more than
   once is materialized. `AS MATERIALIZED` / `AS NOT MATERIALIZED` decide it explicitly: inline
   to let predicates push in, materialize to compute an expensive part once.
6. **Aggregate early.** Aggregate the big side (`lineitem`, `orders`) to the join key before
   joining it to the small side, when the result's grouping allows it.
7. **Replace repeated scans.** Several subqueries that scan the same table with different
   filters become one scan with conditional aggregates (`SUM(...) FILTER (WHERE ...)`, `CASE`).
8. **Window functions instead of self-joins.** A self-join or a correlated subquery that
   finds a per-group maximum or rank becomes `max(...) OVER (PARTITION BY ...)` or a
   `DISTINCT ON`.
9. **Read the plan.** `EXPLAIN (FORMAT JSON)` first; `EXPLAIN ANALYZE` only when the query is
   known to be fast. Look for the costliest node: a sequential scan of `lineitem` repeated in a
   nested loop, a sort spilling to disk, a hash join with a wildly wrong row estimate.
10. **Check equivalence carefully.** The grader compares the multiset of rows with the base
    query's, on SF3 and on an alternate database with different data. Watch NULL handling,
    duplicates (`UNION` versus `UNION ALL`, joins that multiply rows), integer versus numeric
    division and rounding, and `LIMIT` without a total `ORDER BY`. Never embed result values:
    the query must compute its rows from the tables.

## Harness-level ideas (prompt, skills, tools, policy, config)

1. **Inspect the schema and the plan first.** Have the prompt or the policy start from the
   base plan's costliest node, with the schema's keys and row counts at hand.
2. **Verify before submitting.** The policy runs the rewrite, compares its row count and a
   checksum such as `SELECT count(*), sum(hashtext(t::text)) FROM (...) t` with the base
   query's, and times both, before it submits.
3. **Keep the fastest verified rewrite.** Track every rewrite that matched, and on a budget
   stop submit the fastest one instead of `base_sql`.
4. **Tool output that helps.** Show the execution time and the plan's top nodes in
   `execute_sql`'s result, and keep long row lists short so the context lasts.
5. **One focused skill.** A short skill guide with the rewrites above, read on demand, rather
   than a longer system prompt.
6. **Budgets.** Spend fewer steps on easy tasks and more on hard ones; stop early once a
   rewrite reaches τ ≥ 2, which already earns full credit.
7. **Remove what does not help.** A tool, rule or skill that no failure benefits from costs
   tokens and attention.
