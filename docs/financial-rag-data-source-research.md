# Financial RAG Data Source Research

Researched: 2026-05-24

## Recommendation

Use a layered source strategy:

1. **SEC EDGAR as the filing backbone** for 10-K, 10-Q, 8-K, EX-99 exhibits, source URLs, accession numbers, and legally clean citations.
2. **Alpha Vantage as the first free transcript experiment** because it exposes an earnings-call transcript endpoint and claims 15+ years of coverage. Treat this as unproven until Phase 0 tests real tickers and quarters.
3. **EarningsAPI / earningscalls.dev as the first paid transcript upgrade** if Alpha Vantage coverage or rate limits are not enough. It is cheap, structured, API-first, and supports speaker roles.
4. **Quartr as the best long-term enterprise-grade option** if the project later needs highest-quality real-time transcripts, timestamps, speaker IDs, global coverage, and first-party IR data.
5. **sec-api.io as an optional paid parser accelerator** if our own SEC parsing takes too long, not as the default source.

Provider architecture for the RAG layer:

- **Voyage AI**: dense embeddings and optional reranking.
- **OpenAI**: query routing, decomposition, HyDE if used, answer synthesis, citation-aware generation, and answer evaluation.
- **Anthropic**: intentionally not used in this project, so a separate portfolio project can demonstrate Anthropic + Voyage instead.

The practical v1 plan should still work at $0/month: SEC filings plus 8-K exhibits plus whatever Alpha Vantage transcript coverage we can verify. If we want full Q&A transcripts quickly, budget $25-$40/month first, not $100+/month.

## Source Comparison

| Source | Data | Price | Quality / Fit | Concerns |
| --- | --- | --- | --- | --- |
| SEC EDGAR / data.sec.gov | 10-K, 10-Q, 8-K, exhibits, XBRL company facts, company submissions | Free | Best source of truth for filings and citations. No API key. Public filing content is free to access and reuse. | No true earnings-call transcript corpus. Need our own parsing, exhibit classification, rate limiting, and SEC user-agent compliance. |
| SEC Archives direct downloads | Filing HTML, complete submission text, EX-99 attachments, PDFs/images | Free | Essential for verifiable source links and raw-document retention. | HTML and SGML can be messy. Exhibits are uneven and company-specific. |
| Alpha Vantage | Earnings call transcript endpoint by ticker and fiscal quarter | Free tier available; premium starts around $49.99/month based on public pricing trackers, but official pricing page should be checked at signup | Best free transcript candidate. Docs say transcripts cover over 15 years and include LLM sentiment signals. | Need Phase 0 curl test. Free tier is only 25 requests/day, so 20 tickers x 8 quarters could take days. Speaker structure and coverage quality need verification. |
| EarningsAPI / earningscalls.dev | Earnings call transcripts, speaker segments, role tags, full-text search | Free reading and limited API previews; Pro $24.99/month; Ultra $39.99/month; Enterprise $299/month | Strong fit for a portfolio project. API-first, AI-ready, cheap, speaker-role metadata, US-listed coverage. | Newer/smaller provider. Need check transcript accuracy, coverage for our 20 tickers, and terms for storing/cache in local RAG. |
| API Ninjas Earnings Call Transcript API | Full transcripts, search/list endpoints, some enriched fields | Premium only. Developer shown as $39/month yearly-equivalent or $59/month; Business $99 yearly-equivalent or $149/month; Professional $199 yearly-equivalent or $299/month | Good API surface and broad company coverage claim. Search/list endpoints are convenient. | Developer tier does not allow data caching/storing, which is bad for RAG. Speaker split is only Business/Professional. Use only if terms fit local storage. |
| Financial Modeling Prep | SEC-like financial reports, transcript list, full earnings-call transcript API | Free Basic 250 calls/day for limited endpoints. Official pricing shows Ultimate at $149/month billed annually for earnings call transcripts | Mature API ecosystem. Good fallback if we already use FMP for calendars/profile/CIK. | Transcripts appear gated to Ultimate. More expensive than EarningsAPI for our needs. Terms mention display/redistribution licensing for FMP-sourced data. |
| Finnhub | Global filings, earnings transcripts, audio, participants | Free tier for some APIs; transcripts are Premium. Public pricing snippets show stock API plans around $49.99/$129.99/$199.99 per month and fundamentals plans around $50-$200/month/market, but transcript entitlement should be verified | Good breadth: 15+ years, 220K+ audio, participants, management/Q&A session metadata. | Pricing/entitlement is less straightforward. Need account-level confirmation before relying on it. |
| Quartr API | Live and historical transcripts, live audio, filings/reports, 10-K/10-Q/8-K including EX-99.1, slides, summaries | Contact sales / enterprise | Highest-quality strategic option. First-party IR data, 65 markets, structured transcripts, timestamps, speaker IDs, stated transcript accuracy. Best for production AI apps. | Not priced publicly; likely overkill for a resume project until later. Requires sales flow/sandbox. |
| Financial Datasets | SEC filings, press releases, earnings, fundamentals, guidance/KPIs | Developer $200/month, Pro $2,000/month, or pay-as-you-go credits: SEC filings $0.02/request, press releases $0.04/request, earnings $0.01/request | Clean developer API and pay-as-you-go option. Useful for metadata, filings index, press releases, and fundamentals. | Does not look like a full earnings-call transcript source. Expensive compared with direct SEC for filings. |
| EODHD | Market/fundamental/news/calendar data | Free 20 calls/day; paid packages from $19.99/month; all-in-one $99.99/month | Good for market/fundamental side projects, not core for filing/transcript RAG. | No clear earnings-call transcript API found. Not needed because this repo already has yfinance/options pipeline. |

## Best Free Path

Free v1 should use:

- SEC submissions API for company filing history.
- SEC Archives for filing documents and exhibits.
- SEC companyfacts API for future XBRL numerical verification.
- Alpha Vantage transcript endpoint as an experiment.
- Company investor-relations pages only as manual spot-check references, not automated scraping, unless the company provides stable downloadable PDFs/transcripts and terms allow it.

This gives us enough to build:

- 10-K risk-factor lookup.
- 10-Q MD&A and risk-factor comparisons.
- 8-K earnings-release and EX-99 exhibit ingestion.
- Uneven but honest earnings-call-adjacent coverage.
- Possible true call transcripts if Alpha Vantage works.

## Best Low-Cost Paid Path

If we decide to pay for transcripts, test in this order:

1. **EarningsAPI Pro at $24.99/month**
   - Best price-to-fit ratio.
   - Full text, speaker segments, roles, full-text search.
   - Good enough for a portfolio RAG corpus if coverage checks pass.
2. **API Ninjas Business**
   - Consider only if terms allow storing transcript text for RAG.
   - Developer tier should be avoided for this project because data caching/storing is not allowed.
3. **FMP Ultimate**
   - Use if FMP proves much better coverage than cheaper transcript providers.
   - Official pricing shows earnings transcripts on Ultimate at $149/month billed annually.

## Best High-Quality Production Path

If the goal shifts from resume project to serious production data quality, evaluate:

- Quartr API.
- Finnhub premium transcripts.
- sec-api.io for SEC parser acceleration.

The highest-quality production stack would probably be:

- Quartr for transcripts/audio/slides/IR events.
- SEC EDGAR or sec-api.io for raw SEC documents and exact source filings.
- Existing options pipeline for market context.

That is excellent, but it is not the right first move unless we get a free/sandbox Quartr developer arrangement.

## Data Quality Notes

Filings:

- SEC data is authoritative and citation-friendly.
- The engineering burden is parsing and chunking, not data trust.
- 8-K and EX-99 content is inconsistent by issuer.
- EX-99 exhibits can include press releases, CFO commentary, prepared remarks, slides, or other attachments.

Transcripts:

- Full Q&A transcripts are rarely available from SEC filings.
- Transcript vendors differ on speaker segmentation, timestamps, role tagging, and correction quality.
- Role-tagged speaker turns are extremely valuable for this project because speaker-specific retrieval is one of the target query types.
- Any provider used for transcript RAG must allow local caching/storage, otherwise we cannot build a durable vector index.

## Phase 0 Tests To Run Before Implementation

### SEC EDGAR

Test:

- Fetch `company_tickers.json`.
- Resolve CIKs for NVDA, MSFT, AAPL, AMD, JPM.
- Fetch `https://data.sec.gov/submissions/CIK##########.json`.
- Pull the latest 10-K, latest 10-Q, and last 8 quarters of 8-K filings.
- Download filing index pages and list EX-99 exhibits.

Record:

- Number of 10-K/10-Q/8-K documents per company.
- Number of 8-K filings with EX-99.
- Exhibit labels and filenames.
- Whether exhibits appear to be press releases, CFO commentary, slides, or prepared remarks.

### Alpha Vantage

Test:

- `function=EARNINGS_CALL_TRANSCRIPT`
- Tickers: NVDA, MSFT, AAPL, AMD, JPM.
- Quarters: latest 2 quarters plus one older quarter.

Record:

- Whether free key returns full transcript text.
- Whether responses include speakers/segments or one raw transcript blob.
- Coverage gaps by ticker/quarter.
- Rate-limit behavior.

### EarningsAPI

Test via free or one-month Pro if needed:

- Search latest transcripts for NVDA, MSFT, AAPL, AMD, JPM.
- Fetch one transcript.
- Inspect schema for speaker roles, Q&A markers, timestamps, and source metadata.
- Confirm local caching/storage terms.

### API Ninjas

Only test if we are willing to use Business tier or higher:

- Confirm storing/caching terms.
- Check `transcript_split`, participants, and Q&A flags.
- Compare one transcript against EarningsAPI and Alpha Vantage.

## Sources Checked

- SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- SEC Developer FAQ / fair access: https://www.sec.gov/about/webmaster-frequently-asked-questions
- FMP transcript docs: https://site.financialmodelingprep.com/developer/docs/stable/latest-transcripts
- FMP pricing: https://site.financialmodelingprep.com/pricing-plans
- Alpha Vantage docs: https://www.alphavantage.co/documentation/
- Alpha Vantage premium: https://www.alphavantage.co/premium/
- Finnhub docs: https://finnhub.io/docs/api
- Finnhub pricing: https://finnhub.io/pricing-stock-api-market-data
- EarningsAPI: https://earningscalls.dev/
- API Ninjas transcript API: https://api-ninjas.com/api/earningscalltranscript
- API Ninjas pricing: https://api-ninjas.com/pricing
- Quartr API: https://quartr.com/products/quartr-api
- Quartr pricing: https://quartr.com/pricing
- sec-api.io pricing: https://sec-api.io/pricing
- Financial Datasets pricing: https://www.financialdatasets.ai/pricing
- EODHD pricing: https://eodhd.com/pricing
