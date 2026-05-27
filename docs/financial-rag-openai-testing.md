# Financial RAG OpenAI API Testing

This note prepares the local filings platform for opt-in OpenAI answer testing.
Retrieval, citation validation, readiness checks, and evidence quality still run
locally first. The script only calls OpenAI when `--live` is supplied.

## Prerequisites

- Use `.\venv\Scripts\python.exe`.
- Install dependencies from `requirements.txt`; this workspace venv has
  `openai==2.38.0` installed.
- Set a real key in the shell or `.env`:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

- Optional: choose a model with `--model`; the default is `gpt-5.2`.
- Keep local chunks under `data/filings/chunks/` and vectors under
  `data/vector_cache/`.

The OpenAI quickstart documents that the SDK reads `OPENAI_API_KEY` from the
environment, and the current text-generation guide recommends the Responses API
for new text generation work.

## Dry Run

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_openai_answer_smoke.py
```

Dry run mode:

- retrieves local evidence,
- checks evidence quality,
- checks whether the key and SDK are ready,
- builds the citation-constrained prompt,
- writes `artifacts/rag_eval/openai_answer_smoke.json`,
- does not call OpenAI.

Add `--use-voyage` when `VOYAGE_API_KEY` is configured and you want the
retrieval step to use live Voyage query embeddings instead of the offline
constant query embedder.

## Live Smoke

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_openai_answer_smoke.py --live
```

For a stronger end-to-end answer test, use:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_openai_answer_smoke.py --use-voyage --live
```

Live mode refuses to run unless `OPENAI_API_KEY` is configured and the OpenAI SDK
is importable. The generated answer is validated after the call: citations are
accepted only when labels map to retrieved chunks, and hallucinated labels are
reported in `rejected_citations`.

## Guardrails

- The prompt instructs the model to answer only from provided evidence.
- Every factual sentence must cite retrieved labels like `[S1]`.
- The validator hydrates accepted labels with ticker, form type, filing date,
  accession, source URL, and chunk ID.
- No SEC refetch, Anthropic calls, paid transcript APIs, agents, Qdrant, or
  frontend framework work are involved.

## Deferred Work

- Streaming API responses.
- Answer-quality eval suites.
- UI answer display.
- Provider routing, cost tracing, and production observability.
