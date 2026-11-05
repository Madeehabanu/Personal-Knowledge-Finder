# Team meeting notes

Decided to migrate the payments service to the new database cluster next quarter.
Risks: downtime during cutover, stale caches. Action items: Priya drafts the rollback plan,
Sam benchmarks query latency. Postgres connection pooling needs tuning before launch.
