# API Integration Guide

## Retry behavior

**API retry behavior:** clients must use exponential backoff starting at 200 ms,
maximum five attempts, and honor `Retry-After` response headers on HTTP 429.

## Payload limits

**Max payload size:** 256 KB per request body for the public REST API.
Batch endpoints allow up to 1 MB only on the enterprise tier.

## Authentication

Use bearer tokens rotated every 90 days. Invalid tokens return HTTP 401 with problem+json body.
