# Top-k keys, then a LATERAL aggregate

When a query (a) builds a small top-k set of group keys (LIMIT <= ~50) with one GROUP BY over a join, and (b) then computes a second, expensive aggregate for those same keys over an even bigger join, never merge the two aggregates into one GROUP BY over the largest join. Instead: keep subquery (a) unchanged as a derived table t, and replace (b) with a LATERAL that correlates on the group key:

    SELECT ... FROM (<top-k subquery>) t CROSS JOIN LATERAL (SELECT <aggregate expressions> FROM <big tables> WHERE <group key> = t.<key> GROUP BY <group key>) s

This computes the expensive aggregate for k groups instead of all groups. The lateral must GROUP BY exactly the correlation key, and the outer top-k keys must be unique, or rows are lost. Time it with EXPLAIN ANALYZE and keep it only if it is at least 1.2x faster than the base.
