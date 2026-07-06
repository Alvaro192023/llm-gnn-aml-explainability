"""M4 - Baselines tabulares Elliptic. Uso: --modelo lr|rf|xgb --variante af|lf [--seed 42]
Protocolo D013: umbral en val, metricas en test. Resultados en resultados/*.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from metricas import evaluar, f1_por_step, umbral_optimo_f1

NPZ = Path("/sessions/intelligent-charming-einstein/procesado_cache/elliptic_grafo.npz")
if not NPZ.exists():
    NPZ = AQUI.parent / "datos" / "procesado" / "elliptic_grafo.npz"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["lr", "rf", "xgb"], required=True)
    ap.add_argument("--variante", choices=["af", "lf"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    D = np.load(NPZ)
    x, y, step = D["x"], D["y"], D["step"]
    x = x[:, :93] if a.variante == "lf" else x
    etiq = y >= 0
    mtr = D["mask_train"] & etiq; mva = D["mask_val"] & etiq; mte = D["mask_test"] & etiq
    Xtr, ytr = x[mtr], y[mtr]; Xva, yva = x[mva], y[mva]; Xte, yte = x[mte], y[mte]
    if a.modelo == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=a.seed))
    elif a.modelo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced", random_state=a.seed)
    else:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, tree_method="hist",
                          scale_pos_weight=float((ytr == 0).sum() / (ytr == 1).sum()),
                          eval_metric="aucpr", random_state=a.seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    pva = m.predict_proba(Xva)[:, 1]; pte = m.predict_proba(Xte)[:, 1]
    R = {"dataset": "elliptic", "modelo": a.modelo, "variante": a.variante, "seed": a.seed,
         "n_train": int(len(ytr)), **evaluar(yva, pva, yte, pte)}
    R["f1_por_step_test"] = f1_por_step(yte, pte, step[mte], umbral_optimo_f1(yva, pva), range(35, 50))
    out = AQUI / "resultados" / f"elliptic_{a.modelo}_{a.variante}_s{a.seed}.json"
    out.write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in R.items() if k != "f1_por_step_test"}, indent=2))

if __name__ == "__main__":
    main()
