# Deploy GitHub Pages (docs site)

Publishes the [`docs/`](../docs/) folder as a static site (landing page at `docs/index.html`).

## Enable once

1. **GitHub → Settings → Pages**
2. **Source:** GitHub Actions (not “Deploy from branch” — this workflow owns deploy)

## What gets published

- Landing page: `https://dagaza.github.io/Qube/` → `docs/index.html`
- Linked assets under `assets/` (screenshots, logo, social preview) via relative paths

## Triggers

- Push to `main` when `docs/**` or `assets/screenshots/**` or `assets/logos/**` or `assets/social/**` changes
- Manual: **Actions → Deploy GitHub Pages → Run workflow**

## Before official launch

Re-run [launch documentation guidelines](../docs/launch_documentation_guidelines.md) Phase 4 and verify the live URL after deploy.
