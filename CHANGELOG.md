# Changelog

Repository-level changes. Scientific claims are not recorded here — those live in
protocol records under `docs/protocols/` and `docs/reports/`.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [SemVer](https://semver.org/) applied to the repository as an
artifact, and is independent of protocol version numbers.

## [Unreleased]

## [1.0.1] — 2026-08-28

Repository restructure. **No experimental result, artifact, hash, or number is
affected by this release.**

### Added
- Protocol-version layering for records: `docs/protocols/vN/`,
  `docs/reports/vN/`, `docs/experiments/vN/`, `configs/protocols/vN/`,
  `scripts/data/vN/`, `scripts/diagnostics/vN/`, `data/vN/`, `hf_cards/vN/`,
  `notebooks/protocol_vN/`.
- `src/axiom_world/worlds/` with a name-based `registry`, so a second world
  implementation can be added without touching the training or evaluation
  entry points.
- `docs/specs/` holding the contracts the machinery must satisfy
  (verifier, adapter, metric definitions).
- `docs/README.md`, `docs/protocols/README.md`, `docs/roadmap.md`,
  `docs/protocols/v1/amendments.md`, and this changelog.
- `scripts/audits/` for integrity checks that apply to every protocol.

### Changed
- `src/axiom_world/playworld/` → `src/axiom_world/worlds/playworld/`; imports
  updated. No behavioural change.
- `configs/experiments/` → `configs/protocols/v1/`.
- Protocol v1.4 document → `docs/protocols/v1/experiment_protocol_v1.md`.
- Arm closure notes → `docs/experiments/v1/`.
- Documentation references now cite the Zenodo **concept** DOI
  ([10.5281/zenodo.22052148](https://doi.org/10.5281/zenodo.22052148)), which
  always resolves to the latest version. The report itself continues to carry
  its version DOI, where citing a fixed version is correct.

### Preserved
- `docs/experimental-protocol.md` remains as a stub. The published v1.0 report
  cites that path; it must not 404.
- Tag `v1.0.0` is unmoved and continues to name the tree the v1.0 report
  describes.

### Verified
- All 54 CPU contract tests pass before and after the restructure.

## [1.0.0] — 2026-08-23

First public release. Protocol v1.4 complete: environment, frozen evaluation
suites, full experiment matrix, 3-seed confirmation, technical report,
curated champion adapter, and datasets.

[Unreleased]: https://github.com/m97j/axiom-world/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/m97j/axiom-world/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/m97j/axiom-world/releases/tag/v1.0.0
