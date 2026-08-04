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

1. Generate an Ed25519 keypair (keep PEM off-repo):

   ```bash
   mkdir -p ~/.qube-secrets
   python3 -c "
   from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
   from cryptography.hazmat.primitives import serialization
   key = Ed25519PrivateKey.generate()
   pem = key.private_bytes(
       encoding=serialization.Encoding.PEM,
       format=serialization.PrivateFormat.PKCS8,
       encryption_algorithm=serialization.NoEncryption(),
   )
   path = '$HOME/.qube-secrets/qube-prod-1.pem'
   open(path, 'wb').write(pem)
   pub = key.public_key().public_bytes(
       encoding=serialization.Encoding.Raw,
       format=serialization.PublicFormat.Raw,
   )
   print('Wrote', path)
   print('public_key_hex:', pub.hex())
   "
   chmod 600 ~/.qube-secrets/qube-prod-1.pem
   ```

2. Add the **public** key to `assets/licensing/signing_keys.json` with a stable `key_id` (e.g. `qube-prod-1`). Ship in a release before selling licenses signed with that key.

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
