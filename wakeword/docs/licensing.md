# Licensing strategy — fail closed

A model is **commercial/production** only if *every* input asset is on the allowlist.
This is enforced by code, not by human memory.

## Allowlist (commercial-safe)

| SPDX id | Attribution required? | Notes |
|---|---|---|
| `CC0-1.0` | No | Public-domain dedication. |
| `CC-BY-4.0` | Yes | Compile attribution into `ATTRIBUTION.generated.md`. |
| `MIT` | Yes (notice) | |
| `Apache-2.0` | Yes (notice) | |
| `BSD-2-Clause` / `BSD-3-Clause` | Yes (notice) | |
| `Public-Domain` | No | Use an explicit, defensible basis. |

`CC-BY-SA-4.0` is allowed for **configs/docs** only — avoid for *data* that could make
the trained model a share-alike derivative. Treat with care.

## Denylist (hard blockers)

`CC-BY-NC-*`, `CC-BY-ND-*`, `CC-*-NC-*`, "research only", "non-commercial",
"unknown", and any unlicensed asset. Presence of any of these in the production set
fails the gate.

## How the gate works

`scripts/verify_licenses.py`:

1. Walks every `*.license.json` under `datasets/`.
2. Validates each against the manifest schema (required fields present).
3. With `--require-commercial`, asserts each `license` is on the allowlist **and**
   `commercial_use == true`.
4. Exits non-zero on the first violation (CI-enforced; `train.py` calls it before
   training and refuses to start otherwise).

## Two-tier output

The pipeline can still produce a **personal-use** model from NC data for internal
experimentation, but the resulting `model_card.json` is tagged `tier: "personal"` and
can never be promoted to `recommended` in Qube. The commercial path
(`tier: "commercial"`) requires a green license gate. This mirrors the reference
notebook's own non-commercial caveat while keeping the shippable path clean.

## Output model license

A model trained exclusively on allowlisted data is distributable under terms
compatible with Qube's **MIT** license. Record the full provenance chain in the
model card so the claim is auditable.
