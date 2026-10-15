# Decorrelate a correlated subquery into a join

When a SELECT computes a correlated scalar subquery (SELECT expr FROM other o WHERE o.k = t.k
[AND o.k2 = t.k2 ...]) per outer row t, rewrite as SELECT t.*, s.expr FROM (outer) t LEFT JOIN
(SELECT k, k2, expr FROM other) s ON (s.k, s.k2) = (t.k, t.k2). Put every correlation predicate
as an equality join condition on exactly those columns; keep LEFT JOIN so unmatched outer rows
keep NULL like the subquery does; move the subquery's other WHERE conditions into the derived
table. Only use the direct join when the correlation columns are unique in the subquery's table
(a key); otherwise GROUP BY the correlation columns in the derived table. This turns a per-row
scan into one hash join, usually >=2x on large outer tables.
