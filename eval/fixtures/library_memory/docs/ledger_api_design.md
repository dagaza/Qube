# Ledger API v2 — Design Document

Status: approved. Owners: platform team. This document describes the public HTTP interface of the Ledger service, which records account balances and transfers.

## Goals

- Give partner integrations a stable, versioned interface to read balances and create transfers.
- Make transfer creation safe to retry.
- Keep latency under 150 ms at the 95th percentile for reads.

Non-goals: real-time streaming of balance changes (a webhook design follows separately) and bulk export, which stays on the nightly file drop.

## Authentication

Clients authenticate with the OAuth 2.0 client-credentials grant. Access tokens are JWTs valid for 15 minutes; clients must refresh them before expiry rather than on a 401. Each token carries scopes: `ledger.read` for balances and history, `ledger.write` for transfers. Requests without the required scope get 403 with error code `insufficient_scope`.

Mutual TLS is required for production clients in addition to the token.

## Resources

- `GET /v2/accounts/{id}` — account metadata and current balance.
- `GET /v2/accounts/{id}/entries` — ledger entries, newest first.
- `POST /v2/transfers` — create a transfer between two accounts.
- `GET /v2/transfers/{id}` — transfer status: `pending`, `settled` or `rejected`.

Amounts are integers in minor units (pence, cents) with an ISO 4217 currency code. Floating-point amounts are rejected with 422.

## Pagination

List endpoints use cursor pagination. A response includes `next_cursor` when more results exist; pass it as `?cursor=` to get the next page. The default page size is 50 and the maximum is 200 (`?limit=`). Cursors expire after 24 hours. Offset pagination is not supported, because entries are inserted continuously and offsets would skip or repeat rows.

## Idempotency

`POST /v2/transfers` requires an `Idempotency-Key` header (a UUID chosen by the client). The server stores the key and the response for 24 hours. A retry with the same key and the same body returns the original response with header `Idempotent-Replayed: true`. A retry with the same key and a different body returns 409 `idempotency_key_reused`.

## Rate limits

Each client is limited to 600 requests per minute for reads and 60 per minute for transfer creation, using a token bucket. Responses carry `RateLimit-Remaining` and `RateLimit-Reset` headers. When the limit is exceeded the server answers 429 with a `Retry-After` header in seconds; clients should back off exponentially with jitter, starting at one second and capping at 32 seconds.

## Errors

Errors use the RFC 9457 problem-details format, with `type`, `title`, `status`, `detail` and a machine-readable `code`. Validation errors list each failing field under `errors`. Every response, success or failure, includes a `Request-Id` header to quote to support.

## Versioning and deprecation

The major version is in the path (`/v2/`). Additive changes — new fields, new optional parameters, new endpoints — are made without a version change, so clients must ignore unknown fields. Breaking changes require a new major version. A deprecated version is supported for at least 12 months after the successor is generally available; responses from a deprecated version carry a `Sunset` header with the shutdown date.

## Open questions

- Whether to offer a sandbox with synthetic balances (tracked separately).
- Whether transfer cancellation should be an endpoint or a status change.
