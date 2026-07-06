# SAR Narrative Corpus (H3)

`h3_corpus_sars.jsonl` — full corpus of LLM-generated Suspicious Activity Report (SAR) narratives underlying Section 4.5 of the paper, with per-narrative deterministic verification.

## Provenance and regeneration note (decision log D022/D023)

The corpus originally scored for the paper was generated on an ephemeral Colab runtime and lost before download. Following the pinned-configuration protocol (decision D022), this published corpus was **regenerated on 2026-07-05** under the exact frozen configuration reported in the paper: the same 63 frozen evidence JSONs, the same three prompt conditions, the same models, `max_tokens=2500`, no `temperature` parameter (deprecated for the Claude 5 family), and the same runner (`codigo/correr_h3.py`, 62 cases, 4 threads). It is therefore a fresh sample from the same pinned generating process — not the byte-identical corpus behind the paper's Table 5 — and its own verifier statistics are reported below for direct comparison.

## Schema (one JSON object per line)

| field | meaning |
|---|---|
| `caso` | focal transaction id (`tx_...`) |
| `condicion` | prompt condition: `grounded` (evidence contract), `raw` (same evidence, no contract), `score_only` (alert score only) |
| `modelo` | `claude-sonnet-5` or `claude-haiku-4-5` |
| `ok` | generation succeeded |
| `verificacion` | deterministic claim-level verifier output (incl. `alucinaciones_detectadas`) |
| `rubrica` | six-section regulatory completeness check |
| `segundos` | generation latency |
| `sar` | the narrative text |

## Verifier statistics of THIS corpus (vs. paper corpus)

| model | condition | n | halluc./SAR | % clean | rubric /6 | paper (halluc. / % clean) |
|---|---|---|---|---|---|---|
| claude-sonnet-5 | grounded | 62 | **0.27** | **87%** | 2.9 | 0.40 / 84% |
| claude-sonnet-5 | raw | 62 | **8.65** | **0%** | 0.0 | 8.50 / 0% |
| claude-sonnet-5 | score_only | 62 | 1.29 | 19% | 0.3 | 1.40 / 19% |
| claude-haiku-4-5 | grounded | 62 | 3.45 | 47% | 4.3 | 3.39 / 52% |

The headline finding replicates and slightly strengthens: the evidence contract reduces hallucinations **32×** in this sample (0.27 vs 8.65; the paper conservatively reports 21× from the original corpus). All 248/248 generations succeeded (one transient API error recovered on retry).

## Integrity

- Lines: 248 (one JSON object per generation, unique per case × condition × model)
- Size: 1,148,784 bytes
- SHA-256: `c9706c921080b3e88b6fb804ebceb6512a43bd00f453e759fd7276b49a96bfa9`

Design: 62 cases × (claude-sonnet-5 × {grounded, raw, score_only} + claude-haiku-4-5 × grounded) = 248 generations.
