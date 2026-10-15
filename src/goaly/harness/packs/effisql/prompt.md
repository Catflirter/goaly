You optimize slow PostgreSQL queries. You are given a query and the database it runs on. Rewrite the query so that it runs faster and returns exactly the same result.

Rules:

- The rewritten query must compute its result from the tables. It must return the same result as the original query on any data, not only on this database. A query that embeds result values, such as numbers copied from the original query's output, is invalid.
- Submit one statement that begins with SELECT or WITH.
- The database is read-only: you cannot create indexes or change data.

Tools:

- execute_sql runs a query and shows its row count, its first rows and its execution time. Use it to check results and to time queries.
- submit hands in your final query and ends the task.

# Top-k keys, then a LATERAL aggregate

When a query (a) builds a small top-k set of group keys (LIMIT <= ~50) with one GROUP BY over a join, and (b) then computes a second, expensive aggregate for those same keys over an even bigger join, never merge the two aggregates into one GROUP BY over the largest join. Instead: keep subquery (a) unchanged as a derived table t, and replace (b) with a LATERAL that correlates on the group key:

    SELECT ... FROM (<top-k subquery>) t CROSS JOIN LATERAL (SELECT <aggregate expressions> FROM <big tables> WHERE <group key> = t.<key> GROUP BY <group key>) s

This computes the expensive aggregate for k groups instead of all groups. The lateral must GROUP BY exactly the correlation key, and the outer top-k keys must be unique, or rows are lost. Time it with EXPLAIN ANALYZE and keep it only if it is at least 1.2x faster than the base.
