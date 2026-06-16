# Scoped Sketch — Optional TradingAgents Deep-Research Lens

**Status:** proposal, not built. Decide before implementing.
**Author:** AI Portfolio Analyst session, 2026-06-15.

## One-line

Wire TradingAgents in as an **optional, opt-in, single-ticker** "deep research"
panel on the **Stock Search screen (Feature A)** — never in the portfolio report,
never directive. Our framework owns portfolio-level scoring/risk/advice;
TradingAgents is a per-name lens the user explicitly invokes and waits for.

## Why this shape (and not in the report)

- **Cost/latency is bounded to one name on demand.** TradingAgents fires ~10-20+
  LLM calls per ticker (4 analysts + bull/bear debate rounds + trader + 3-way
  risk team + PM). In the report that's 200-400 calls for a 20-holding book; on
  the search screen it's one ticker the user chose to wait on.
- **Epistemic risk is quarantined.** It is a *research framework* the authors
  explicitly say is "not financial, investment, or trading advice," with no
  validated out-of-sample edge in our universe. Behind an explicit opt-in + an
  "experimental" label on a single-name deep dive, that's honest. Sprayed across
  a portfolio report as per-holding verdicts, it is not.
- **No architectural takeover.** Run it out-of-process via its CLI/package, store
  the result, render it. We don't adopt LangGraph into our request path.

## Surface

Stock Search screen → after our score loads → a collapsed card:
"Run multi-agent deep research (experimental, ~1-3 min)". On click → async job →
poll → render the agents' reports (analyst takes, bull/bear debate transcript,
risk team view, PM rating) clearly labeled as experimental and non-directive.

## Architecture (minimal, isolated)

```
Frontend (Stock Search)
  └─ POST /api/deepresearch/run {ticker}            -> {job_id}
  └─ GET  /api/deepresearch/{job_id}                -> {status, result?}

Backend
  └─ run_deep_research_job(job_id, ticker)  (Celery, max_retries=0, own queue)
       └─ subprocess: `tradingagents analyze --ticker {T} --date {today} --json`
          (out-of-process; its deps/LangGraph never imported into our app)
       └─ parse stdout JSON -> persist DeepResearchJob.result_json
  └─ DeepResearchReport table:
       id, ticker, status(RunStatus), result_json (JSONB),
       provider='tradingagents', model, cost_estimate, created_at, completed_at
```

Config: pin `tradingagents==<version>` in a **separate** optional requirements
file (`requirements-deepresearch.txt`) installed only on the worker image, so a
break in their fast-moving repo can't take down the core API. Set
`config["llm_provider"]="anthropic"`, reuse `ANTHROPIC_API_KEY` +
`ANTHROPIC_MODEL`; `quick_think`/`deep_think` from env. `max_debate_rounds`
capped (1-2) to bound cost.

## Guardrails (non-negotiable, mirrors the rest of the product)

1. **Opt-in only.** Never auto-runs; never in the report path.
2. **Labeled experimental + non-directive.** Render the PM "BUY/SELL" as
   "experimental model output," not a recommendation. Same advisory voice and
   caveats as the Advisor's View.
3. **Cost ceiling.** Per-run LLM-call cap + a daily per-user budget; refuse past
   it with a clear message. Show an estimated cost before running.
4. **Cache.** Per ticker-per-day cache (their decision log already reflects this)
   so re-opening a name doesn't re-bill.
5. **Failure-isolated.** Job failure shows "deep research unavailable," never
   degrades the core score view. Subprocess timeout (~5 min) enforced.
6. **No automation.** Output never feeds the optimizer or trade export.

## Effort estimate

- Table + Celery job + 2 endpoints + subprocess wrapper/parser: ~0.5-1 day.
- Frontend card + poll + labeled render: ~0.5 day.
- Cost-cap + cache + timeout hardening: ~0.5 day.
- Worker image: add optional deps (heavier build). ~0.25 day.
- **Total ~2 days**, fully isolated from the core path.

## Open questions for the decision

1. **Is the per-name value worth ~2 days + ongoing LLM spend**, given we still
   have **zero validated IC** on our own scores? The honest prioritization
   argument: validate the existing signal first (mid-July forward window); a
   sophisticated unvalidated second opinion is easy to over-trust.
2. **Whose API key / whose budget** funds the agent calls — ours or the user's?
3. **Version pinning discipline** — who watches their 0.2.x cadence for breaks?

## Recommendation

Build it **only after** the first real IC read on our own scores, and only if
users actually ask for per-name depth. Until then, the native Bull/Bear case
(shipped) covers the structured-debate value at ~1% of the cost and risk. If we
do build it, the single-ticker opt-in surface above is the only shape that
respects the product's decision-support, non-directive, honest-about-uncertainty
posture.
