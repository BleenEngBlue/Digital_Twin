---
title: Monica's Digital Twin
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: true
python_version: '3.13'
---

# Monica's Digital Twin

A production RAG agent, not a chatbot demo: retrieval-augmented generation with agentic tool-calling, real-time intent routing, guardrails, and per-stage observability — built with Python, OpenAI, ChromaDB, and Gradio, and operated continuously in production since May 2026. Every reply is grounded in retrieved source documents, not model memory.

**Live:** https://huggingface.co/spaces/Monica-Wu/digital-twin · **Code:** https://github.com/BleenEngBlue/Digital_Twin

## Try it in 60 seconds

- Ask about the work: *"What production systems has Monica shipped?"* or *"What's her experience in regulated environments?"*
- Probe the grounding: ask something the corpus can't know — the agent says "I don't know" and escalates the gap to the real Monica instead of guessing.
- Trip the escalation: tell it you're interested in hiring or collaborating. The agent collects your details and the real Monica's phone buzzes within seconds — the same tool-calling loop documented below, firing live.

## Architecture — independently testable pipeline layers

```
Ingestion (once):   sources ─→ chunk ─→ embed ─→ store (ChromaDB)
Query (per msg):    embed query ─→ retrieve top-k ─→ assemble context ─→ generate ─→ tool calls ⟲
```

The pipeline is decomposed into five stages — chunking → embedding → retrieval → context assembly → generation — each with a single responsibility and a clean interface, so every layer is independently testable and independently swappable. That separation exists for one reason: when answer quality degrades, I want to localize the failure to a specific stage (a retrieval-quality problem vs. a generation problem) instead of debugging the system as a black box. It's also what makes component-level evaluation possible: each stage can be scored, regressed, and replaced on its own — e.g., swapping the vector store or the chunking strategy without touching generation.

Startup (runs once):

1. **Configure** — load environment variables, validate API keys, set model and ChromaDB constants.
2. **Load sources** — pull source documents from a private Hugging Face dataset.
3. **Define persona & guardrails** — build the system prompt: voice, behavioral constraints, and the fact sheet the model is restricted to.
4. **Init tools** — register the function-calling tools available to the model (see Intent routing & escalation).
5. **Chunk** — split each document into overlapping, boundary-aware text windows (paragraph → sentence → word, so no chunk cuts mid-thought). Chunking is fully deterministic, which keeps retrieval results reproducible run-to-run — a precondition for meaningful evals.
6. **Embed** — convert chunks to vectors via OpenAI Embeddings.
7. **Store** — persist vectors and metadata in a ChromaDB collection.
8. **Launch UI** — start the Gradio ChatInterface.

Per-message query flow:

9. **Embed query** — same embedding model as ingestion (mixing models silently breaks similarity search; the pipeline enforces one constant).
10. **Retrieve** — top-k semantic search over ChromaDB.
11. **Assemble context** — retrieved chunks are concatenated with rank, chunk index, and source attribution, so answers are traceable back to the exact source passages that produced them.
12. **Generate** — assembled context is injected into the system prompt and the LLM is called with full conversation history; the final answer streams token by token.
13. **Handle tool calls** — if the model requests a tool, execute it, feed the result back, and loop until a final text answer is returned.

## Evaluation

The architecture is eval-first: because each stage is isolated behind an interface, quality is measured per component rather than end-to-end only.

- **Chunking** — deterministic output with enforced size/overlap/boundary invariants, so chunk quality is inspectable and any change to the chunker is diffable against the previous chunk set. The semantic (boundary-aware) chunking strategy replaced a naive fixed-window baseline after side-by-side comparison of retrieval coherence.
- **Retrieval** — every query logs the retrieved chunks with source and chunk index, giving a per-query retrieval trace that supports manual relevance review and failure attribution today, and serves as the labeling substrate for a golden dataset.
- **Generation** — grounding is enforced structurally (the model may only state facts present in retrieved context, and must say "I don't know" otherwise), which turns hallucination into an observable, testable failure mode rather than a silent one.
- **End-to-end benchmark (run_benchmark.py)** — a 66-case golden question set replayed against the live Space with per-turn tool-call traces captured; every answer is graded by an LLM-as-judge (gpt-4.1-mini) against the source facts. Side effects are stubbed in eval mode so graded behavior is production-identical. Seven cases were added for the streaming and hardening pass (tool discipline, first-person refusal, notification spam) and pass 7/7 on the current build; benchmark-run latency is tracked per release (mean 3.86 s, p90 4.95 s with streaming on).
- **Attack suite (sec_tests.py)** — 12 tests that run against the real request handler, covering forged conversation history, notification spam, tool-loop and message bounds, malformed tool arguments, and private-data leakage including mid-stream.

Next: retrieval-level scoring (recall@k, MRR) over labeled query → chunk pairs, and wiring the judge scores into a CI regression gate.

## Guardrails — mapped to the OWASP LLM Top 10

The system prompt enforces boundaries the model can't reason its way around at inference time, targeting the risk categories from the OWASP Top 10 for LLM Applications:

- **Sensitive information disclosure (LLM02):** no disclosure of private contact information — phone, home address, private email — under any phrasing of the request.
- **Excessive agency (LLM06):** the agent cannot sign contracts, agree to terms, or make financial or professional commitments on Monica's behalf; its tool surface is deliberately minimal, and the only real-world side effect it can produce is a notification to a human.
- **Misinformation / hallucination (LLM09):** strict grounding — the only facts the model can state are those present in retrieved context; a knowledge gap triggers an explicit "I don't know" plus a human escalation (below) instead of a fabricated answer.
- **Prompt-injection resistance (LLM01):** persona and constraint instructions are isolated in the system prompt with delimited fact boundaries; adversarial, manipulative, or off-topic input triggers a tone shift and conversation redirection rather than compliance. Prompt text is not the only defense: conversation history is sanitized to user/assistant text only (last 20 turns) so a forged turn cannot replace the system prompt, and a deterministic redaction guard runs on every streamed chunk — with a hold-back so a half-generated phone number or email never reaches the screen. Twelve attack tests exercise these paths against the real handler.
- **Reputation and conduct:** professional neutrality about past employers and colleagues is enforced even when negative takes are requested directly; disrespectful or inappropriate input ends the witty persona and, if needed, the conversation.

## Intent routing & escalation

Two situations short-circuit the normal reply flow and route to a real-time push notification (Pushover), so a human decision never waits on someone reading chat logs:

- **Hire/collaborate intent:** when a visitor signals hiring or collaboration interest, the agent collects name and contact details and escalates them to the real Monica within seconds of the interaction.
- **Knowledge-gap escalation:** when retrieved context can't answer a question about Monica, the agent automatically notifies her with the unanswered question — no user action required — so gaps in the source corpus get surfaced and closed over time instead of hallucinated past.

Every tool invocation and its result is logged before being fed back to the model, so the agentic loop is auditable turn by turn.

## Observability & production operation

I own the observability of this system in production — latency, cost, quality drift, and error rates — with instrumentation currently scoped to what a small production system needs to catch failures and drift:

- **Retrieval tracing:** every query logs its retrieved chunks with source and chunk index, so a wrong or thin answer is attributable to retrieval quality vs. generation before any code is touched.
- **Tool-call tracing:** every tool call and result is logged, making the agent loop replayable when diagnosing an escalation that fired (or didn't).
- **Error containment:** the query-time path is wrapped so an exception — embedding failure, API error, malformed tool call — degrades to a clean retry message for the user while the underlying error is logged to the Space console for follow-up. One user's failure never crashes the session.
- **Cost control by design:** generation runs on a cost-efficient model tier (`gpt-4.1-mini`) with a compact retrieved context (top-k chunks, bounded chunk size), keeping per-query token spend small and predictable.
- **Abuse and cost bounds:** per-session and global rate limits on chat and on the notification tool, a 2,000-character message cap, a 3-round tool-loop cap, and a bounded request queue — all tunable by environment variable; a hard monthly spend limit and alert on the OpenAI project. Streaming is on by default.
- **Privacy-gated logging:** visitor text and retrieved corpus passages are logged only when verbose mode is explicitly enabled; exceptions log the error type, never the payload.
- **Roadmap:** per-request timing and token usage as queryable metrics (p95 latency, $/query, eval pass rate per release), and a full-suite regression run gating each deploy.

## Design notes — the same pattern at enterprise scale

The corpus here is one person's professional history, but the architecture is the general pattern for reconciling natural-language queries with messy, heterogeneous documents: deterministic ingestion, per-stage evals, retrieval traces with source attribution, structural grounding so an unmatched query fails loudly (human escalation) instead of silently (hallucination), and a feedback loop that surfaces corpus gaps so they get closed over time. Point the same pipeline at an enterprise corpus — legacy registers, drawings, PDFs, policy documents — and the stages you'd tune first are chunking (document structure varies far more) and retrieval (entity-heavy queries reward hybrid search and metadata filtering). That's exactly why those stages are swappable, and why every match is traceable to the source passages that produced it: at scale, a confidence you can't audit is a liability.

## Stack

Python · OpenAI (`gpt-4.1-mini`, `text-embedding-3-small`) · ChromaDB · Gradio · Hugging Face Spaces · Pushover· streaming

## Setup

The following secrets must be configured in the Space settings (not stored here):

- `OPENAI_API_KEY` — OpenAI API key for embeddings and generation
- `DIGITAL_TWIN_TOKEN` — Hugging Face token for the private source dataset (`Monica-Wu/digital-twin-data`)
- `PUSHOVER_USER` — Pushover user key for notifications
- `PUSHOVER_TOKEN` — Pushover app token for notifications