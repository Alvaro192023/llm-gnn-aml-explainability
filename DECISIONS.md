# Decision Log (D001–D023)

Frozen methodological decision log referenced in Sec. 3.7 of the paper. Entries are English condensations of the project's internal log; the log is **append-only** — when a later decision supersedes an earlier one, both are kept, so the evolution of the method remains auditable. Dates follow the project calendar (July 2026).

## Design and scope

**D001 — Dual-dataset strategy.** Elliptic (real Bitcoin graph) for comparability with the state of the art, plus IBM AML HI-Small / AMLworld (calibrated synthetic) for semantically meaningful fields and generator-labeled laundering typologies — the only setting where SAR narratives and explanations can be validated against ground truth.

**D002 — Compute split.** GPU training (end-to-end GNNs) on Google Colab (T4); EDA, graph construction, propagation-family training, and evaluation on CPU in the project workspace.

**D003 — LLM choice (initial).** Open-weights model as primary narrator with a commercial API as comparison. *Superseded by D016/D017:* pinned commercial API (Anthropic) became primary; the open-weights comparison was deferred to future work for schedule reasons.

**D004 — Temporal split frozen before modeling (initial cut).** Strictly temporal train/test separation fixed before any model was trained, to preempt temporal leakage — the leading methodological critique in this domain. Refined into the final frozen cuts by D009.

**D005 — Manuscript language.** English.

**D006 — Target venue.** IEEE Access (primary), aligned with the project timeline; preprint to arXiv at submission time.

## Data, splits, and graphs

**D007 — Execution environment.** All analysis code runs as versioned scripts inside a single reproducible workspace; no local IDE dependencies.

**D008 — Data ingestion.** Elliptic is read directly from the distributed zip (no 1 GB extraction); graph tensors are built with pandas/networkx and converted to PyG tensors only where training happens (per D002).

**D009 — Final frozen splits.** Elliptic: train = steps 1–29, validation = 30–34, test = 35–49 (test deliberately includes the step-43 dark-market shutdown as a distribution-shift stress test). AMLworld: daily volume collapses after Sep 10, so calendar cuts would leave a degenerate test set; cuts are placed at the 60th/80th volume quantiles (2022-09-06 13:36 and 2022-09-08 16:12), yielding 3,046,859 / 1,015,601 / 1,015,876 transactions. Explanation evaluation uses only laundering attempts fully contained in the test window.

**D010 — Outlier policy.** No observation is removed for being extreme (outliers frequently *are* the fraud signal). Only technical defects are corrected: 9 exact duplicate rows dropped; log1p on amount features; self-transfers and unknown categories kept with explicit handling.

**D011 — Graph design.** Elliptic: node classification, excluding the absolute time-step index from inputs (temporal-shortcut prevention). AMLworld: edge classification on a directed multigraph of entities keyed as bank‖account (515,088 entities; 8 raw account-ID collisions detected and disambiguated); 8 continuous + 3 categorical edge features; causal per-window node features. Hard rule: the pattern/attempt edge labels **never** enter training — they exist only for explanation evaluation.

**D012 — Leakage-safe message graphs.** Elliptic uses the full graph (time steps are isolated components; asserted programmatically). AMLworld uses *expanding* graphs: training passes messages only over training edges; validation over train+val; test over the full history. Future edges can never influence past predictions.

**D013 — Metrics protocol.** PR-AUC primary (test prevalences 6.5% and 0.177%); decision thresholds selected exclusively on validation (max-F1) and applied unchanged to test; minority-class F1/precision/recall reported; ROC-AUC secondary only — the logistic baseline's 0.933 ROC-AUC vs 0.026 PR-AUC is kept as the imbalance-inflation exhibit; seeds 42–44 with deterministic components reported as zero-variance.

## Modeling

**D014 — Two-phase detector plan.** Phase 1: precomputed-propagation family (SGC/SIGN-style; on AMLworld with incident-edge-feature aggregation before propagation), trainable on CPU and — because propagation is linear up to the head — admitting *exact* per-neighbor contribution decomposition, the property the evidence extractor exploits. Phase 2 (mandatory before submission): end-to-end GINe/GCN replication under the identical protocol to verify the pattern of results (executed in D021).

## Generative layer and explanation evaluation

**D015 — Evidence contract and verifier.** Evidence JSONs are strictly model-side (ground truth lives only in a separate evaluation manifest). SAR generation under a six-rule evidence contract with mandatory [tx_id] citations; three experimental conditions (grounded / raw / score-only) operationalize H3. The deterministic factuality verifier was validated by injection controls (3/3 planted hallucinations detected; 0 false positives).

**D016 — Explainability evaluation frame.** (a) Faithfulness and plausibility evaluated as independent axes (following Jacovi & Goldberg); (b) evidence ranking v2 = propagation weight × temporal kernel (τ = 3 days) adopted; the window-recency heuristic wins on this benchmark (0.602) but exploits the generator's post-collapse artifact — reported in the ablation, not adopted; (c) deletion fidelity measured in logits (probability deltas saturate); (d) structural limitation declared: BIPARTITE/STACK attempts are unreachable within 2 hops (reachability ceiling 0.727).

**D017 — Pinned LLM execution.** All SAR generation via pinned model versions (`claude-sonnet-5`, `claude-haiku-4-5`) through a self-contained runner with resumable JSONL checkpointing; the `temperature` parameter is intentionally omitted (deprecated in the Claude 5 family); budget: 62 cases × 4 model/condition arms = 248 generations.

**D018 — H3 execution fix.** Root cause of initial API failures diagnosed: sending `temperature` to Claude 5 family models returns HTTP 400 → parameter removed with a backward-compatible fallback for older models. Runner re-run end-to-end.

**D019 — H3 results adopted.** Verifier-scored corpus (244/248 successful): sonnet-5 grounded 0.40 hallucinations/SAR, 84% fully clean; raw 8.50 / 0%; score-only 1.40 / 19%; haiku-4-5 grounded 3.39 / 52%. Headline: the evidence contract reduces hallucinations **21×** at equal information.

## Execution, replication, and provenance

**D020 — Public repository and manuscript port.** Reproducibility package layout (code, light artifacts, figures) with datasets *not* redistributed (Kaggle terms) and secrets excluded by construction; manuscript ported to IEEEtran.

**D021 — End-to-end GINe replication (executed).** GINe (two GINEConv layers, entity embeddings, edge encoders, LinkNeighborLoader [15,10], 50:1 negatives) under the identical D012/D013 protocol, 3 seeds × 2 budgets. Extended budget (30 epochs, patience 6) reported: PR-AUC 0.229 ± 0.025 (standard budget 0.215 ± 0.061; doubling the budget adds only +0.014 while halving variance — the gap to linear propagation is not a training-budget artifact). Verdict: propagation v2 (0.536) leads by 2.3×; GINe nearly doubles the best tabular baseline. The paper's narrative — aggregation-before-propagation is the decisive ingredient — is confirmed and sharpened.

**D022 — Blinded-review sample regenerated under pinned configuration.** The original H3 corpus lived on an ephemeral compute runtime and was lost before download (verified exhaustively). Methodological decision: for the blinded expert review, the 10-narrative sample was **regenerated** under the exact pinned configuration of the paper (same 63 frozen evidence JSONs, same prompts, same pinned models, same generation settings). Validity argument: the review evaluates narratives *from the same pinned generating process* reported in the paper — a fresh sample of that process, not the byte-identical historical text. Blind design: strata 4 sonnet-grounded / 2 haiku-grounded / 2 raw / 2 score-only over 10 distinct cases (seed 42; no case repeated across conditions), shuffled presentation order, sealed de-anonymization manifest opened only after scoring.

**D023 — Full-corpus regeneration and open-data release.** To honor the paper's data-release commitment after the D022 loss, the **full 248-narrative corpus was regenerated** by re-running the original runner unmodified under the pinned configuration. Outcome: 248/248 successful; the regenerated corpus's own verifier statistics replicate the paper's (grounded 0.27 hallucinations/SAR / 87% clean vs raw 8.65 / 0% — a 32× contract effect in this sample vs the conservative 21× reported from the original corpus). Published at `corpus/h3_corpus_sars.jsonl` with per-narrative verifier output, SHA-256 integrity hash, and a provenance note (`corpus/README_corpus.md`). The blinded review (D022) was de-anonymized and integrated in Secs. 4.5/5.4; its central finding — the expert reviewer could not distinguish contracted from free-generation narratives despite the 21× factuality gap — motivates claim-level automated verification as a mandatory pipeline stage.
