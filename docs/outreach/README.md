# Outreach drafts (T34)

**Drafts only. No message in this directory is sent by any tool in this repo.** Each file is a human-editable template; sending is a manual action by the author on the matching platform. The `.github/workflows/*.yml` files do not read anything under `docs/outreach/`.

## Files

| File | Length | Use when |
|---|---|---|
| [`email_6line.md`](email_6line.md) | 6 lines | Cold outreach to one named person (hiring manager, researcher, local LLM team). |
| [`linkedin.md`](linkedin.md) | ~250 words | LinkedIn post announcing v0.1.0 to Greek NLP / ML network. Paired with one headline figure. |
| [`hackernews.md`](hackernews.md) | 1 paragraph + link | Show HN submission. Keeps the title in the HN 80-char limit. |
| [`twitter_thread.md`](twitter_thread.md) | 5 tweets | Short thread with one visual per tweet, for X / Mastodon. |
| [`blog_post.md`](blog_post.md) | ~1500 words | Long-form post for a personal site or Dev.to — rewrites the technical report in conversational voice. |

All five point to the same canonical URLs (planner, release page, technical report) so a cross-poster does not have to rewrite the links per platform.

## Per-recipient tailoring checklist

Before sending **anything**, walk the list for the specific person or venue:

- [ ] **Role fit.** Does the recipient care about *capacity*, *quality*, or *infra* first? Rewrite the opening sentence so the first number is the one that matches their job.
- [ ] **Name, not "team"**. If you have a first name, use it. "Hi team" reads like a mail-merge; this is a cold reach-out, not a newsletter.
- [ ] **Link check.** Make sure the deployed planner URL (`georgebon5.github.io/greekkv/`) is up — if Pages is still waiting on the Actions billing unlock, pin the local-reproduce path (`cd planner && npm run preview`) instead of a dead link.
- [ ] **Number check.** Copy the measured headline numbers from [`docs/report.md` §4](../report.md) verbatim. Do not round away the CI — the CI is the point.
- [ ] **Scope honesty.** If the recipient works on serving infra, name that only Qwen2.5-0.5B is measured so far; do not imply Krikri-8B rows exist until they do.
- [ ] **Ask.** What specifically are you asking for? Review, hiring conversation, collaboration on the T4 re-run, redistribution to a mailing list? Spell it out in one line at the end.
- [ ] **Signature line.** Match the platform: short for Twitter, no signature in HN, full name + role in LinkedIn / email.

## What *not* to do

- No bcc / undisclosed-recipients blasts. If you need to reach five people, send five emails.
- No "I'd love your thoughts" without naming what you want thoughts on.
- No performance numbers from the planner's `estimated` rows — those lines are **not measured**, they exist only to show the arithmetic, and quoting them as results breaks the honesty contract the project is built around.
- No PDF attachment on cold emails — link to [the v0.1.0 release](https://github.com/georgebon5/greekkv/releases/tag/v0.1.0) instead, where the PDF lives alongside the hashes and the wheel.
