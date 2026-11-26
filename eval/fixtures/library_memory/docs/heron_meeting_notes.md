# Project Heron — Meeting Notes

Project Heron is the relaunch of the customer mobile app (iOS and Android) on the new design system and payments backend.

## Kick-off — 8 January 2026

Attendees: Priya (product), Tomasz (engineering lead), Ines (design), Kofi (QA), Lena (marketing).

- Scope agreed: new onboarding flow, saved payment methods, dark mode, accessibility to WCAG 2.2 AA.
- Out of scope for the relaunch: in-app chat and the loyalty programme; both move to phase 2.
- Target launch: 10 March 2026.
- Budget: £180,000 for the relaunch, including £25,000 for external accessibility testing.

Actions: Tomasz to estimate the payments migration; Ines to share the onboarding prototype by 16 January.

## Weekly sync — 22 January 2026

- The onboarding prototype tested well with eight users; two found the account-linking step confusing. Ines will simplify it to a single screen.
- Tomasz's estimate for the payments migration is six weeks, two weeks more than planned, because the old tokens must be re-vaulted.
- Risk raised: the March date is now tight. Decision deferred to the steering meeting.

Actions: Kofi to draft the regression test plan; Lena to brief the app-store screenshots.

## Steering meeting — 5 February 2026

Decision: the launch date moves from 10 March to 14 April 2026, to fit the payments migration and a full accessibility audit. The budget stays at £180,000; the extra engineering time is covered by dropping the planned tablet layout.

- The external accessibility audit is booked for the week of 16 March.
- Marketing will hold the press release until the new date is confirmed by QA.

## Weekly sync — 19 February 2026

- Payments migration is 60 % done. Re-vaulting tested on 2,000 staging accounts without failures.
- Dark mode is complete on iOS; Android is waiting on two components from the design system team.
- Kofi reported 41 open bugs, 6 of them high priority, mostly in the onboarding flow.

Actions: Tomasz to pair an engineer with the design-system team; Priya to triage the high-priority bugs daily.

## Weekly sync — 12 March 2026

- All high-priority bugs closed. 17 low-priority bugs remain and will not block launch.
- Accessibility audit found 11 issues, 3 of them serious (screen-reader labels on the payment form). Fixes due by 27 March.
- Beta release to 500 customers went out on 9 March; crash-free sessions at 99.6 %.

Actions: Lena to schedule the launch announcement for 14 April at 09:00.

## Go / no-go — 9 April 2026

Decision: go. Crash-free sessions in beta at 99.8 %, all serious accessibility issues fixed and re-tested. The phased rollout starts at 10 % of users on 14 April and reaches 100 % by 21 April if crash rates stay below 0.5 %.

Phase 2 (in-app chat, loyalty) planning starts in May.
