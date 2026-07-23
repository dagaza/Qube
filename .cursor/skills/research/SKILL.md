---
name: research
description: Repository investigation workflow for architecture discovery, dependency tracing, and implementation archaeology. Use when mapping how something is built before changing it.
---

# Research Skill

Use for repository archaeology, architecture mapping, dependency tracing, implementation
discovery, and risk identification. Read-only.

When investigating:
1. Map the existing implementation (entry points, modules, ownership).
2. Identify ownership boundaries (which package/layer owns what).
3. Trace the data flow (inputs -> transforms -> outputs -> persistence/UI).
4. Identify extension points (interfaces, registries, hooks).
5. Identify risks (duplication, leaks, coupling, missing tests).

Record concrete evidence (path + symbol / line range) so findings are attributable; feed
material findings into `.cursor/starfall/evidence-map.md`.

Output format:
```
## Current State
## Relevant Files      (path : symbol — one line each)
## Data Flow
## Existing Constraints
## Recommended Change
## Unknowns
```
