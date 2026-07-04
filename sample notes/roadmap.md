# Q3 Product Roadmap

## Priorities
1. Ship the mobile onboarding redesign — target: end of August.
2. Migrate the payments service to the new billing provider.
3. Pilot the AI support assistant with 50 beta users in Nairobi and Lagos.
4. Launch the referral program in September, contingent on onboarding shipping first.
5. Begin scoping a v2 of the analytics dashboard for Q4 — no committed date yet.

## Budget
Engineering budget for Q3 is $180,000, split roughly:
- 60% salaries ($108,000)
- 25% cloud infrastructure ($45,000)
- 15% tooling and vendor costs ($27,000)

Cloud spend has been trending about 8% over budget the last two months,
mostly driven by the payments migration's dual-running costs (old + new
provider running in parallel during the transition).

## Team
- Amina — Product lead, AI assistant pilot
- Brian — Frontend, onboarding redesign
- Faith — Backend, payments migration
- Kevin — Growth, referral program + localization

## Risks
- Payments migration depends on the vendor's sandbox environment, which was
  unstable in June (see meeting notes — this has since improved).
- Beta user recruitment for the AI assistant pilot is behind schedule; only
  18 of 50 target users had signed up as of the last count.
- Referral program launch is at risk of slipping into October if onboarding
  redesign isn't done by end of August, since the two share frontend resources.
- No dedicated QA resource for the payments migration — relying on Faith and
  one contractor for testing before cutover.

## Success metrics
- Onboarding redesign: reduce signup drop-off from 34% to under 20%.
- Payments migration: zero downtime during cutover, transaction fees reduced
  by at least 15% vs. current provider.
- AI assistant pilot: 70%+ of beta users rate first response "helpful" or
  better; target for full rollout decision by end of Q3.