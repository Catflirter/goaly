Keep the rewrite's output shape identical to the base query's: the same columns, in the same
order, with the same types. Three mechanical rules:

1. STAR EXPANSION. Inlining or flattening a subquery or CTE that adds computed columns drops
them from an outer `alias.*`: re-list every computed column, with the same alias, beside the star.
2. AGGREGATE TYPE. Re-associating an aggregate changes its type: SUM over per-group counts is
numeric where the base's COUNT(col) is bigint, and SUM over bigint is numeric. Keep the base's
outer aggregate function, or cast the final expression back (e.g. `::bigint` for a base COUNT).
Never change an output column's expression or type.
3. TIES UNDER LIMIT. If the base ends in ORDER BY ... LIMIT and the sort keys can repeat, the row
it returns depends on which plan runs. Keep that part of the query unchanged, submitting the
base's own structure, instead of restructuring it.
