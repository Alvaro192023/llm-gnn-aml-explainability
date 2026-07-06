# LLM-Enhanced GNNs for Explainable Financial Fraud Detection

Reproducibility package for the paper *"LLM-Enhanced Graph Neural Networks for Explainable Financial Fraud Detection: Generating Auditable Suspicious Activity Reports"* (Villanueva Kobayashi, 2026; under review).

A pipeline that (i) detects money-laundering transactions with a scalable propagation GNN, (ii) extracts model-faithful evidence with exact neighbor contributions, (iii) generates regulatory Suspicious Activity Report (SAR) narratives under an evidence contract, and (iv) evaluates the explanations on two axes — faithfulness (deletion fidelity) and plausibility (typology recovery) — plus SAR factuality.

## Headline results

- **Detection (IBM AML HI-Small):** propagation GNN 0.536 PR-AUC vs 0.125 best tabular baseline (**4.3×**) under a strictly temporal, leakage-controlled protocol. An end-to-end GINe under the same protocol reaches 0.229 ± 0.025 (2.3× below propagation): pre-propagation edge-feature aggregation, not learned message passing, is the decisive ingredient.
- **Explanation faithfulness:** deleting selected evidence drops the detector's margin 4.66 logits vs 0.10 for random controls (57/62 cases, p = 3.1×10⁻¹¹).
- **Explanation plausibility:** 46% typology-transaction recall (64% of the reachability ceiling); artifact-sensitive baselines reported.
- **SAR factuality:** the six-rule evidence contract cuts hallucinations **21×** (0.40 vs 8.50 per report over identical evidence; 84% of contracted narratives entirely clean).

## Datasets (not redistributed)

Download from Kaggle under their terms and place under `data/`:
- **Elliptic** — `ellipticco/elliptic-data-set`
- **IBM AML (AMLworld) HI-Small** — `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` (files `HI-Small_Trans.csv`, `HI-Small_Patterns.txt`)

Cite Weber et al. (2019) and Altman et al. (2023) respectively.

## Pipeline (order of execution)

```
codigo/verificar_datasets.py        # sanity-check the raw datasets
codigo/eda_elliptic.py --parte ...  # EDA + frozen temporal split
codigo/eda_ibm.py --parte ...
codigo/construir_grafo_elliptic.py  # build graph tensors -> data/procesado/
codigo/construir_grafo_ibm.py --parte cargar|grafo|patrones|nodos
codigo/baselines_elliptic.py --modelo lr|rf|xgb --variante af|lf
codigo/baselines_ibm.py --modelo construir|lr|rf|xgb
codigo/gnn_prop_elliptic.py --cabeza xgb|mlp --variante af|lf   # CPU detector
codigo/gnn_prop_ibm.py --enriquecido                            # CPU detector (v2)
codigo/gnn_elliptic.py / gnn_ibm.py                             # end-to-end PyG (GPU, Colab)
codigo/extraer_evidencia.py --auto                              # XAI -> evidence JSON
codigo/generar_sar.py --backend anthropic --modelo ...          # contracted SAR
codigo/correr_h3.py                                             # H3 batch + verifier
codigo/evaluar_explicabilidad.py --parte recall|fidelidad       # H2
```

Global seed 42. All decisions are logged in the project decision log (see paper Sec. 3.7). Deterministic factuality verification: `generar_sar.verificar_factualidad`.

## SAR corpus (open data)

`corpus/h3_corpus_sars.jsonl` — the full 248-narrative SAR corpus (62 cases × 4 model/condition arms) with per-narrative deterministic verification. Regenerated 2026-07-05 under the paper's pinned configuration after the original ephemeral-runtime corpus was lost (decision log D022/D023); its own verifier statistics replicate the paper's (contract vs raw: 0.27 vs 8.65 hallucinations/SAR, 32× in this sample vs 21× reported). See `corpus/README_corpus.md` for schema, statistics, and SHA-256.

## Reproducibility notes

- Detector experiments (propagation family) run on CPU; end-to-end GINe/GCN replication runs on a single T4 GPU (Colab).
- SAR generation uses pinned LLM API versions; `temperature` is intentionally omitted (deprecated in the Claude 5 family).
- `artefactos_ligeros/` holds small derived tables (metrics, encoders, pattern index) for inspection without rerunning; large tensors and datasets are excluded (see `.gitignore`).

## Citation

```bibtex
@article{villanuevakobayashi2026llmgnn,
  author  = {Villanueva Kobayashi, {\'A}lvaro Fabricio},
  title   = {LLM-Enhanced Graph Neural Networks for Explainable Financial Fraud Detection: Generating Auditable Suspicious Activity Reports},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Code released under the MIT License (see `LICENSE`). Datasets remain under their original Kaggle terms and are not included here.
