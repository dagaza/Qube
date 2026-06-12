# Meeting Minutes — Router Eval Sync

**Date:** 2026-03-04

## Attendees

Platform, Memory, and QA leads.

## Summary

- Approved offline router evaluation corpus v1.
- Agreed to seed fixture library for automated RAG regression.
- Action: tune T4 chat margin before enabling hybrid on general knowledge prompts.

## Decisions

Ship eval harness behind `tools/evaluate_router.py` with isolated LanceDB directory.
