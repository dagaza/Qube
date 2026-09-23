# Chocolatey — `qube-cuda` moderation recovery

`qube-cuda` is stuck when an older version sits in **Waiting for Maintainer**. Chocolatey blocks new pushes (HTTP **403**) until that version is resolved.

## Current situation (2026-09)

| Version | Status | Action |
|---------|--------|--------|
| **1.3.0** | Approved (exempted) | Keep — only live CUDA version today |
| **1.3.46** | Waiting for Maintainer | **Self-reject** (verification timed out downloading ~1 GiB installer) |
| **1.3.53** | Push failed (403) | Push after 1.3.46 is cleared |

CPU (`qube`) and Vulkan (`qube-vulkan`) **1.3.53** are already approved.

## Step 1 — Self-reject 1.3.46 (manual, ~2 minutes)

Self-reject is supported when verification/validation failed. See [Chocolatey FAQs — self-reject](https://docs.chocolatey.org/en-us/faqs/#how-do-i-self-reject-a-package).

1. Sign in at [community.chocolatey.org](https://community.chocolatey.org) as **Qubeapp** (package maintainer).
2. Open [qube-cuda 1.3.46](https://community.chocolatey.org/packages/qube-cuda/1.3.46).
3. In the **Review** section, choose **Reject** / **Self-reject** for this version (wording varies; maintainer-only while in moderation).
4. Confirm — status should become **Rejected** (not the same as obsolete; rejected versions are not installable).

There is no public API for self-reject; this must be done in the browser.

## Step 2 — Push 1.3.53

After 1.3.46 is rejected, push the current release package:

```bash
gh workflow run chocolatey-submit.yml \
  -f version=1.3.53 \
  -f packages=qube-cuda
```

Or locally (after rendering with `scripts/render_chocolatey_package.py` and `choco pack`):

```powershell
choco push chocolatey/out/1.3.53/qube-cuda/qube-cuda.1.3.53.nupkg `
  --source https://push.chocolatey.org/ `
  --api-key YOUR_API_KEY
```

Requires repository secret **`CHOCOLATEY_API_KEY`** and `CHOCOLATEY_AUTO_PUSH=true` for release workflow pushes.

## Step 3 — Moderator comment (paste on the new 1.3.53 package page)

Use **Add to Review Comments** on the package page (not Disqus):

```text
Superseding rejected 1.3.46 — please close that queue entry.

qube-cuda 1.3.53 verification note:
- The CUDA Setup.exe is ~1.1 GiB; Chocolatey verifier timed out on 1.3.46 during download (see https://gist.github.com/choco-bot/db1c08681e3329614c503e5cab0d3ca7).
- Release CI runs scripts/release/smoke_choco.ps1 (CPU package install/launch/uninstall) and builds all three variants from the same pipeline.
- WinGet dagaza.Qube.CUDA 1.3.53 passed full automated validation including Installation Validation (Defender) on the same GitHub Release binary.

Request: verification bypass or re-run when network allows — installer is silent Inno Setup (/VERYSILENT) and SHA256-pinned to the official GitHub Release asset.
```

## Package metadata fixes (in repo)

- **Icon:** `package.nuspec` uses PNG (`qube_logo_256.png`) per packaging guidelines (not `.ico`).
- **Push script:** `push_chocolatey_packages.ps1` fails the job if any `choco push` returns non-zero (1.3.53 CUDA 403 was previously masked as success).

## If 1.3.53 verification fails again

1. Read the new [choco-bot gist](https://gist.github.com/choco-bot) linked on the package page.
2. Replicate with [chocolatey-test-environment](https://github.com/chocolatey-community/chocolatey-test-environment).
3. Reply on the package review thread or request verifier bypass for oversized GitHub Release downloads.

See also [`chocolatey/README.md`](../chocolatey/README.md) and [`winget_cuda_defender_investigation.md`](winget_cuda_defender_investigation.md).
