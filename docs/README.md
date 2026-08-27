# Documentation Map

Axiom-World separates two kinds of documents, and they age differently.

- **Records** describe *what was done under a specific protocol*. They are
  frozen once their protocol closes and are never edited in place; corrections
  are amendment entries. Records are stored under a protocol-version directory.
- **Specifications and guides** describe *how the machinery works right now*.
  They are living documents and are stored unversioned.

| Path | Kind | Contents |
|---|---|---|
| `docs/protocols/` | record | Pre-registration documents and amendment logs, by version. Start at [`protocols/README.md`](protocols/README.md). |
| `docs/reports/` | record | Technical reports, by protocol version. |
| `docs/experiments/` | record | Closure notes and post-mortems for individual experiment arms, by protocol version. |
| `docs/specs/` | living | Contracts the machinery must satisfy: verifier, adapter, metrics. |
| `docs/architecture.md` | living | Code layout, module responsibilities, extension points. |
| `docs/data-governance.md` | living | Dataset construction, splits, leakage gates, fingerprinting. |
| `docs/reproducibility.md` | living | Run artifacts, lineage enforcement, identity audits, reseeding. |
| `docs/roadmap.md` | living | Track sequencing and what each protocol version is for. |
| `CHANGELOG.md` | living | Repository-level changes, by tag. |

## Reading order for a newcomer

1. [`docs/roadmap.md`](roadmap.md) — what this project is trying to establish.
2. [`docs/protocols/README.md`](protocols/README.md) — which protocol produced which claims.
3. `docs/reports/v1/` — the results themselves.
4. [`docs/specs/verifier-contract.md`](specs/verifier-contract.md) — why the numbers are checkable.

## Stable paths

`docs/experimental-protocol.md` is a permanent stub preserving a path cited by
the published v1.0 report. Do not delete it. Any future restructure must leave
externally cited paths resolvable in the same way.
