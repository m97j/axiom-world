# Protocols — Index and Lineage

Every quantitative claim in this repository is attributable to exactly one
pre-registered protocol. A protocol is frozen in git *before* its first run; the
freezing commit is the protocol's anchor. Post-freeze changes are numbered
amendment entries, never in-place edits.

## Lineage

| Version            | Status | Frozen      | Question it settles                                                                                                                                                                         | Records       |
| ------------------ | ------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| **v1.4**     | closed | 2026-07-23  | Does staged post-training (general-reasoning SFT → task SFT) beat direct task tuning in a deterministically verifiable planning world? Where do DPO and verifier-rewarded RL help or fail? | [`v1/`](v1/) |
| **v2.0-CLB** | frozen | FREEZE_DATE | Under partial observability, does maintaining an explicit belief state improve closed-loop success and the model's transition knowledge, measured through counterfactual prediction error?  | [`v2/`](v2/) |

The current research program is organized under Protocol v2:

- **v2-CLB — Closed-Loop Belief:** observation, belief state, sequential interaction, and verifier-grounded online RL.
- **v2-AP — Agent Capability Scaling:** general-purpose and agent-capability improvements, including Phase-1 interventions.
- **v2-WE — World Extension:** controlled scaling of world complexity while capability is held fixed within each primary comparison.

The `v3.0-CLB` designation used in the earlier frozen documentation is retained
in git history for provenance. It was superseded before canonical experimental
execution by the current `v2.0-CLB` organization.

See [`../roadmap.md`](../roadmap.md) for the full research-program roadmap.

## What each version owns

Records are version-scoped. Machinery is not.

```
docs/protocols/vN/     pre-registration + amendment log
docs/reports/vN/       technical report
docs/experiments/vN/   arm closure notes and post-mortems
configs/protocols/vN/  experiment configurations
scripts/data/vN/       dataset construction for that protocol
scripts/diagnostics/vN/  diagnostics written for that protocol
data/vN/               frozen eval suites and manifests
hf_cards/vN/           model and dataset cards
notebooks/protocol_vN/ as-run notebooks, outputs included

src/                   shared machinery — NOT version-scoped
tests/                 contract tests — NOT version-scoped
scripts/common/        shared utilities — NOT version-scoped
scripts/audits/        always-on integrity audits — NOT version-scoped
```

A diagnostic that proves generally useful is promoted from `scripts/diagnostics/vN/`
to `scripts/audits/`; promotion is a machinery change and is recorded in
`CHANGELOG.md`, not in an amendment.

## Rules

1. **Freeze before run 1.** A protocol document committed after its first run is
   not a pre-registration and must not be described as one.
2. **Amend, never edit.** Every deviation gets a numbered entry with date,
   reason, and what it does and does not affect.
3. **Records are immutable after closure.** Only documentary path corrections are
   permitted, and only when a restructure would otherwise break a published
   citation.
4. **Claims name their protocol.** A number without a protocol version is not a
   result.
