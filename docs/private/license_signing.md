# Commercial license signing (maintainers)

This note covers **Pro / Team / Enterprise** license issuance. The **issuer tool stays in the public repo**; the **Ed25519 private key never does**.

## What is public vs secret

| Item | Location | Commit to GitHub? |
|------|----------|-------------------|
| Issuer CLI | `tools/issue_qube_license.py` | **Yes** — no secrets in the script |
| Verify + schema | `core/licensing/` | **Yes** |
| Embedded **public** keys | `assets/licensing/signing_keys.json` | **Yes** |
| Pack signing (same keys) | `tools/sign_qube_pack.py` | **Yes** |
| **Private** signing key (`.pem`) | Your machine / password manager | **Never** |
| Issued `.qube-license` / serial keys | Email to customers, order logs | **Never** (customer-specific) |
| Resend API keys, order CSVs | Fulfillment automation (Phase 3) | **Never** |

Publishing `issue_qube_license.py` does **not** let anyone forge licenses for **official** Qube builds without your **private** key. Forks can remove checks or use their own keys only for **their** builds.

The docstring “off-repo” on the issuer tools refers to the **private key path**, not the script itself.

## One-time production setup

1. Generate an Ed25519 keypair (keep PEM off-repo) and register the public key:

   ```bash
   python3 tools/generate_qube_signing_key.py \
     --key-id qube-prod-1 \
     --output ~/.qube-secrets/qube-prod-1.pem \
     --add-to-signing-keys
   ```

   Omit `--add-to-signing-keys` if you prefer to paste the printed JSON entry into `assets/licensing/signing_keys.json` manually.

2. Ship a release that includes the new public key before selling licenses signed with that `key_id`.

3. **Do not** use `qube-test-1` (RFC 8032 test vector in the repo) for paid customers.

## Issue a license (file + email serial)

```bash
python3 tools/issue_qube_license.py ~/orders/acme-001.qube-license \
  --tier pro \
  --private-key ~/.qube-secrets/qube-prod-1.pem \
  --key-id qube-prod-1 \
  --print-serial \
  --serial-out ~/orders/acme-001.key.txt
```

- **`--print-serial`** — QUBE1 key for the email body (no attachment).
- **`--serial-out`** — optional file copy for your order log.

Customer flow: **Settings → License → paste QUBE1 key → Activate license key** (or import the `.qube-license` file).

## Issue a batch (many serial keys)

```bash
python3 tools/issue_qube_license.py ~/orders/batch-20260804 \
  --tier pro \
  --count 500 \
  --private-key ~/.qube-secrets/qube-prod-1.pem \
  --key-id qube-prod-1 \
  --manifest-out ~/orders/batch-20260804/manifest.csv
```

Creates:

- `licenses/<prefix>-0001.qube-license` … one signed file per license
- `serials/<prefix>-0001.key.txt` … one QUBE1 key per license (email body)
- `manifest.csv` with columns `id,tier,serial,license_file,issued`

Each license gets a unique `issued` timestamp (base time plus microsecond offset) so serial keys do not collide. Use `--no-serial-files` if you only want the CSV manifest.

## Email (Resend) guidance

- Put the **serial key in the plain-text body**; avoid JSON attachments.
- Example subject: `Your Qube Pro license key`
- Keep a local order log (order id → tier → date); the license JSON does not include customer email.

## Security checklist

- [ ] Production `.pem` only under `~/.qube-secrets/` (gitignored via `*.pem` and `**/qube-secrets/`)
- [ ] Never `git add` `*.pem`, issued licenses, or API secrets
- [ ] Back up the private key securely (loss = cannot issue/re-sign with same key id)
- [ ] Key rotation: add new public key in app release; re-issue customers if retiring an old key

## Dev / CI

- Tests use ephemeral keys or `qube-test-1` with monkeypatched verify paths.
- CI does not need a production private key.

See also: [Releasing Qube](../releasing.md) (public release checklist).
