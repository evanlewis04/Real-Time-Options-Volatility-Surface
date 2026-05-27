# Financial RAG Phase 0 Findings

- Date run: 2026-05-24 21:17:12 UTC
- Tickers checked: NVDA, MSFT, AAPL, AMD, JPM
- Scope: SEC ticker metadata, SEC submissions, recent 8-K filing indexes, EX-99 exhibit coverage, and Alpha Vantage transcript endpoint availability.
- Out of scope: embeddings, vector databases, chunking, retrieval, LLM calls, answer synthesis, and downloaded filing corpus storage.

## SEC CIK Resolution

| Ticker | Company | CIK | Status |
| --- | --- | --- | --- |
| NVDA | NVIDIA CORP | 0001045810 | resolved |
| MSFT | MICROSOFT CORP | 0000789019 | resolved |
| AAPL | Apple Inc. | 0000320193 | resolved |
| AMD | ADVANCED MICRO DEVICES INC | 0000002488 | resolved |
| JPM | JPMORGAN CHASE & CO | 0000019617 | resolved |

## SEC Filing Coverage

| Ticker | Latest 10-K | Latest 10-Q | Recent 8-K count | Recent 8-K accessions |
| --- | --- | --- | ---: | --- |
| NVDA | [10-K 2026-02-25 (0001045810-26-000021)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm) | [10-Q 2026-05-20 (0001045810-26-000052)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000052/nvda-20260426.htm) | 5 | [8-K 2026-05-20 (0001045810-26-000051)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000051/nvda-20260520.htm), [8-K 2026-05-08 (0001045810-26-000028)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000028/nvda-20260507.htm), [8-K 2026-04-27 (0001045810-26-000026)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000026/nvda-20260424.htm), [8-K 2026-03-06 (0001045810-26-000024)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000024/nvda-20260302.htm), [8-K 2026-02-25 (0001045810-26-000019)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000019/nvda-20260225.htm) |
| MSFT | [10-K 2025-07-30 (0000950170-25-100235)](https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm) | [10-Q 2026-04-29 (0001193125-26-191507)](https://www.sec.gov/Archives/edgar/data/789019/000119312526191507/msft-20260331.htm) | 5 | [8-K 2026-05-14 (0001193125-26-224155)](https://www.sec.gov/Archives/edgar/data/789019/000119312526224155/d125909d8k.htm), [8-K 2026-04-29 (0001193125-26-191457)](https://www.sec.gov/Archives/edgar/data/789019/000119312526191457/msft-20260429.htm), [8-K 2026-01-28 (0001193125-26-027198)](https://www.sec.gov/Archives/edgar/data/789019/000119312526027198/msft-20260128.htm), [8-K 2025-12-08 (0001193125-25-311196)](https://www.sec.gov/Archives/edgar/data/789019/000119312525311196/d34077d8k.htm), [8-K 2025-10-29 (0001193125-25-256310)](https://www.sec.gov/Archives/edgar/data/789019/000119312525256310/msft-20251028.htm) |
| AAPL | [10-K 2025-10-31 (0000320193-25-000079)](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm) | [10-Q 2026-05-01 (0000320193-26-000013)](https://www.sec.gov/Archives/edgar/data/320193/000032019326000013/aapl-20260328.htm) | 5 | [8-K 2026-04-30 (0000320193-26-000011)](https://www.sec.gov/Archives/edgar/data/320193/000032019326000011/aapl-20260430.htm), [8-K 2026-04-20 (0001140361-26-015711)](https://www.sec.gov/Archives/edgar/data/320193/000114036126015711/ef20071035_8k.htm), [8-K 2026-02-24 (0001140361-26-006577)](https://www.sec.gov/Archives/edgar/data/320193/000114036126006577/ef20060722_8k.htm), [8-K 2026-01-29 (0000320193-26-000005)](https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/aapl-20260129.htm), [8-K 2026-01-02 (0001140361-26-000199)](https://www.sec.gov/Archives/edgar/data/320193/000114036126000199/ef20060722_8k.htm) |
| AMD | [10-K 2026-02-04 (0000002488-26-000018)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm) | [10-Q 2026-05-06 (0000002488-26-000076)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000076/amd-20260328.htm) | 5 | [8-K 2026-05-15 (0001193125-26-226746)](https://www.sec.gov/Archives/edgar/data/2488/000119312526226746/d118163d8k.htm), [8-K 2026-05-05 (0000002488-26-000072)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000072/amd-20260505.htm), [8-K 2026-02-24 (0000002488-26-000045)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000045/amd-20260223.htm), [8-K 2026-02-17 (0000002488-26-000029)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000029/amd-20260210.htm), [8-K 2026-02-03 (0000002488-26-000014)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000014/amd-20260203.htm) |
| JPM | [10-K 2026-02-13 (0001628280-26-008131)](https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/jpm-20251231.htm) | [10-Q 2026-05-01 (0001628280-26-029344)](https://www.sec.gov/Archives/edgar/data/19617/000162828026029344/jpm-20260331.htm) | 5 | [8-K 2026-05-21 (0000019617-26-000228)](https://www.sec.gov/Archives/edgar/data/19617/000001961726000228/jpm-20260519.htm), [8-K 2026-05-07 (0001193125-26-211978)](https://www.sec.gov/Archives/edgar/data/19617/000119312526211978/d903351d8k.htm), [8-K 2026-04-24 (0000019617-26-000119)](https://www.sec.gov/Archives/edgar/data/19617/000001961726000119/jpm-20260421.htm), [8-K 2026-04-23 (0001193125-26-173739)](https://www.sec.gov/Archives/edgar/data/19617/000119312526173739/d235028d8k.htm), [8-K 2026-04-14 (0001628280-26-025013)](https://www.sec.gov/Archives/edgar/data/19617/000162828026025013/jpm-20260414.htm) |

## EX-99 Exhibit Coverage From Recent 8-K Index Pages

| Ticker | 8-K accession | Filing date | EX-99 count | Exhibit filenames and descriptions |
| --- | --- | --- | ---: | --- |
| NVDA | [8-K 2026-05-20 (0001045810-26-000051)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000051/nvda-20260520.htm) | 2026-05-20 | 2 | [q1fy27pr.htm](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000051/q1fy27pr.htm) - EX-99.1; inferred: press release; [q1fy27cfocommentary.htm](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000051/q1fy27cfocommentary.htm) - EX-99.2; inferred: CFO commentary |
| NVDA | [8-K 2026-05-08 (0001045810-26-000028)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000028/nvda-20260507.htm) | 2026-05-08 | 0 | No EX-99 exhibit found |
| NVDA | [8-K 2026-04-27 (0001045810-26-000026)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000026/nvda-20260424.htm) | 2026-04-27 | 0 | No EX-99 exhibit found |
| NVDA | [8-K 2026-03-06 (0001045810-26-000024)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000024/nvda-20260302.htm) | 2026-03-06 | 0 | No EX-99 exhibit found |
| NVDA | [8-K 2026-02-25 (0001045810-26-000019)](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000019/nvda-20260225.htm) | 2026-02-25 | 2 | [q4fy26pr.htm](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000019/q4fy26pr.htm) - EX-99.1; inferred: press release; [q4fy26cfocommentary.htm](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000019/q4fy26cfocommentary.htm) - EX-99.2; inferred: CFO commentary |
| MSFT | [8-K 2026-05-14 (0001193125-26-224155)](https://www.sec.gov/Archives/edgar/data/789019/000119312526224155/d125909d8k.htm) | 2026-05-14 | 1 | [d125909dex991.htm](https://www.sec.gov/Archives/edgar/data/789019/000119312526224155/d125909dex991.htm) - EX-99.1 |
| MSFT | [8-K 2026-04-29 (0001193125-26-191457)](https://www.sec.gov/Archives/edgar/data/789019/000119312526191457/msft-20260429.htm) | 2026-04-29 | 1 | [msft-ex99_1.htm](https://www.sec.gov/Archives/edgar/data/789019/000119312526191457/msft-ex99_1.htm) - EX-99.1 |
| MSFT | [8-K 2026-01-28 (0001193125-26-027198)](https://www.sec.gov/Archives/edgar/data/789019/000119312526027198/msft-20260128.htm) | 2026-01-28 | 1 | [msft-ex99_1.htm](https://www.sec.gov/Archives/edgar/data/789019/000119312526027198/msft-ex99_1.htm) - EX-99.1 |
| MSFT | [8-K 2025-12-08 (0001193125-25-311196)](https://www.sec.gov/Archives/edgar/data/789019/000119312525311196/d34077d8k.htm) | 2025-12-08 | 0 | No EX-99 exhibit found |
| MSFT | [8-K 2025-10-29 (0001193125-25-256310)](https://www.sec.gov/Archives/edgar/data/789019/000119312525256310/msft-20251028.htm) | 2025-10-29 | 3 | [msft-ex99_1.htm](https://www.sec.gov/Archives/edgar/data/789019/000119312525256310/msft-ex99_1.htm) - EX-99.1; [msft-ex99_2.htm](https://www.sec.gov/Archives/edgar/data/789019/000119312525256310/msft-ex99_2.htm) - EX-99.2; [msft-ex99_3.htm](https://www.sec.gov/Archives/edgar/data/789019/000119312525256310/msft-ex99_3.htm) - EX-99.3 |
| AAPL | [8-K 2026-04-30 (0000320193-26-000011)](https://www.sec.gov/Archives/edgar/data/320193/000032019326000011/aapl-20260430.htm) | 2026-04-30 | 1 | [a8-kex991q2202603282026.htm](https://www.sec.gov/Archives/edgar/data/320193/000032019326000011/a8-kex991q2202603282026.htm) - EX-99.1 |
| AAPL | [8-K 2026-04-20 (0001140361-26-015711)](https://www.sec.gov/Archives/edgar/data/320193/000114036126015711/ef20071035_8k.htm) | 2026-04-20 | 0 | No EX-99 exhibit found |
| AAPL | [8-K 2026-02-24 (0001140361-26-006577)](https://www.sec.gov/Archives/edgar/data/320193/000114036126006577/ef20060722_8k.htm) | 2026-02-24 | 0 | No EX-99 exhibit found |
| AAPL | [8-K 2026-01-29 (0000320193-26-000005)](https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/aapl-20260129.htm) | 2026-01-29 | 1 | [a8-kex991q1202612272025.htm](https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/a8-kex991q1202612272025.htm) - EX-99.1 |
| AAPL | [8-K 2026-01-02 (0001140361-26-000199)](https://www.sec.gov/Archives/edgar/data/320193/000114036126000199/ef20060722_8k.htm) | 2026-01-02 | 0 | No EX-99 exhibit found |
| AMD | [8-K 2026-05-15 (0001193125-26-226746)](https://www.sec.gov/Archives/edgar/data/2488/000119312526226746/d118163d8k.htm) | 2026-05-15 | 0 | No EX-99 exhibit found |
| AMD | [8-K 2026-05-05 (0000002488-26-000072)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000072/amd-20260505.htm) | 2026-05-05 | 2 | [q12026991.htm](https://www.sec.gov/Archives/edgar/data/2488/000000248826000072/q12026991.htm) - EX-99.1; [amdq126earningsslidesfin.htm](https://www.sec.gov/Archives/edgar/data/2488/000000248826000072/amdq126earningsslidesfin.htm) - EX-99.2; inferred: presentation slides |
| AMD | [8-K 2026-02-24 (0000002488-26-000045)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000045/amd-20260223.htm) | 2026-02-24 | 1 | [pressreleasedatedfebruary2.htm](https://www.sec.gov/Archives/edgar/data/2488/000000248826000045/pressreleasedatedfebruary2.htm) - EX-99.1; inferred: press release |
| AMD | [8-K 2026-02-17 (0000002488-26-000029)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000029/amd-20260210.htm) | 2026-02-17 | 0 | No EX-99 exhibit found |
| AMD | [8-K 2026-02-03 (0000002488-26-000014)](https://www.sec.gov/Archives/edgar/data/2488/000000248826000014/amd-20260203.htm) | 2026-02-03 | 2 | [q42025991.htm](https://www.sec.gov/Archives/edgar/data/2488/000000248826000014/q42025991.htm) - EX-99.1; [amdq425earningsslidesfin.htm](https://www.sec.gov/Archives/edgar/data/2488/000000248826000014/amdq425earningsslidesfin.htm) - EX-99.2; inferred: presentation slides |
| JPM | [8-K 2026-05-21 (0000019617-26-000228)](https://www.sec.gov/Archives/edgar/data/19617/000001961726000228/jpm-20260519.htm) | 2026-05-21 | 0 | No EX-99 exhibit found |
| JPM | [8-K 2026-05-07 (0001193125-26-211978)](https://www.sec.gov/Archives/edgar/data/19617/000119312526211978/d903351d8k.htm) | 2026-05-07 | 0 | No EX-99 exhibit found |
| JPM | [8-K 2026-04-24 (0000019617-26-000119)](https://www.sec.gov/Archives/edgar/data/19617/000001961726000119/jpm-20260421.htm) | 2026-04-24 | 0 | No EX-99 exhibit found |
| JPM | [8-K 2026-04-23 (0001193125-26-173739)](https://www.sec.gov/Archives/edgar/data/19617/000119312526173739/d235028d8k.htm) | 2026-04-23 | 0 | No EX-99 exhibit found |
| JPM | [8-K 2026-04-14 (0001628280-26-025013)](https://www.sec.gov/Archives/edgar/data/19617/000162828026025013/jpm-20260414.htm) | 2026-04-14 | 1 | [a1q26_earningsxpresentat.htm](https://www.sec.gov/Archives/edgar/data/19617/000162828026025013/a1q26_earningsxpresentat.htm) - JPMORGAN CHASE & CO. EARNINGS PRESENTATION SLIDES - FINANCIAL RESULTS - 1Q26 |

## Alpha Vantage Transcript Probe

- Status: ok
- Tested symbol: NVDA
- Tested quarter: 2025Q3
- Result: ok: transcript list with 31 turns; sample keys: ['content', 'sentiment', 'speaker', 'title']

## Recommendation

SEC EDGAR is still the filing backbone, but recent EX-99 coverage is uneven enough that the app should surface coverage gaps prominently. Alpha Vantage returned a usable transcript-shaped response, so it remains a viable free transcript candidate pending broader coverage checks. A paid transcript source is not required to begin free-only filing RAG, but it is still likely needed for complete speaker-aware Q&A transcript workflows unless Alpha Vantage coverage proves broad and cache-friendly.
