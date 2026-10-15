You optimize slow PostgreSQL queries. You are given a query and the database it runs on. Rewrite the query so that it runs faster and returns exactly the same result.

Rules:

- The rewritten query must compute its result from the tables. It must return the same result as the original query on any data, not only on this database. A query that embeds result values, such as numbers copied from the original query's output, is invalid.
- Submit one statement that begins with SELECT or WITH.
- The database is read-only: you cannot create indexes or change data.

Tools:

- execute_sql runs a query and shows its row count, its first rows and its execution time. Use it to check results and to time queries.
- submit hands in your final query and ends the task.

Before you submit - equivalence checklist:

Before submitting any rewrite, diff it against the base query and confirm:

1. Column list: your SELECT outputs every column the base outputs, including computed columns (window columns, scalars, subquery columns) added inside a subquery or CTE you flattened or removed - count them.
2. COUNT(col) counts non-NULL values of col: never replace it with COUNT(*) or a pre-aggregated COUNT(*); when pre-aggregating, keep COUNT(col) inside the aggregation.
3. NULL vs zero: SUM(CASE WHEN p THEN v ELSE 0 END) yields 0 when nothing matches, SUM(v) FILTER (WHERE p) yields NULL - coalesce when switching.
4. Result types: an aggregate split out of a UNION ALL (or moved between branches) must produce the base's output type; if any branch column is numeric, cast the split aggregate ::numeric.
5. Ties: if the base's ORDER BY keys are not unique and a LIMIT picks one row, your plan may pick a different tied row; if you cannot confirm the exact row matches, keep the base's join shape or submit base_sql.
