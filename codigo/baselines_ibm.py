"""M4 - Baselines tabulares IBM AML (clasificacion de aristas).
Uso: --modelo construir|lr|rf|xgb [--seed 42] [--arboles N] [--continuar]
Features (53): ef_float(8) + one-hot monedas/formato(37) + nodo causal src/dst(8).
Nodo causal: filas train/val usan ventana hasta_c1; filas test usan hasta_c2 (D011).
Protocolo D013: umbral en val, metricas en test completo."""
import argparse, json, sys
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from metricas import evaluar

CACHE = Path("/sessions/intelligent-charming-einstein/procesado_cache")
if not CACHE.exists():
    CACHE = AQUI.parent / "datos" / "procesado"
FEAT = CACHE / "ibm_baseline_feats"

def construir():
    FEAT.mkdir(exist_ok=True)
    ef = np.load(CACHE / "ibm_ef_float.npy")
    cat = np.load(CACHE / "ibm_ef_cat.npy").astype(np.int16)
    y = np.load(CACHE / "ibm_y.npy")
    sp = np.load(CACHE / "ibm_split.npy")
    src = np.load(CACHE / "ibm_src.npy"); dst = np.load(CACHE / "ibm_dst.npy")
    nf1 = np.load(CACHE / "ibm_nodo_feats_hasta_c1.npy"); nf2 = np.load(CACHE / "ibm_nodo_feats_hasta_c2.npy")
    I15, I7 = np.eye(15, dtype=np.float32), np.eye(7, dtype=np.float32)
    oh = np.hstack([I15[cat[:, 0]], I15[cat[:, 1]], I7[cat[:, 2]]])
    for nombre, m, nf in [("tr", sp == 0, nf1), ("va", sp == 1, nf1), ("te", sp == 2, nf2)]:
        X = np.hstack([ef[m], oh[m], nf[src[m]], nf[dst[m]]]).astype(np.float32)
        np.save(FEAT / f"X{nombre}.npy", X)
        np.save(FEAT / f"y{nombre}.npy", y[m])
        print(nombre, X.shape, "positivos:", int(y[m].sum()))

def cargar():
    d = {}
    for n in ["tr", "va", "te"]:
        d[f"X{n}"] = np.load(FEAT / f"X{n}.npy", mmap_mode="r")
        d[f"y{n}"] = np.load(FEAT / f"y{n}.npy")
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["construir", "lr", "rf", "xgb"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arboles", type=int, default=100)
    ap.add_argument("--continuar", action="store_true")
    a = ap.parse_args()
    if a.modelo == "construir":
        construir(); return
    d = cargar()
    ytr = d["ytr"]
    if a.modelo == "lr":
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(np.asarray(d["Xtr"][::10]))  # ajuste en submuestra 10% (velocidad; documentado)
        m = SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=5, tol=None, random_state=a.seed)
        m.fit(sc.transform(np.asarray(d["Xtr"])), ytr)
        pva = m.predict_proba(sc.transform(np.asarray(d["Xva"])))[:, 1]
        pte = m.predict_proba(sc.transform(np.asarray(d["Xte"])))[:, 1]
        extra = {}
    elif a.modelo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(a.seed)
        pos = np.flatnonzero(ytr == 1); neg = np.flatnonzero(ytr == 0)
        neg_sub = rng.choice(neg, size=len(pos) * 100, replace=False)  # submuestreo 100:1 (documentado)
        idx = np.sort(np.concatenate([pos, neg_sub]))
        m = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=a.seed)
        m.fit(np.asarray(d["Xtr"][idx]), ytr[idx])
        pva = m.predict_proba(np.asarray(d["Xva"]))[:, 1]
        pte = m.predict_proba(np.asarray(d["Xte"]))[:, 1]
        extra = {"submuestreo_neg": "100:1"}
    else:
        import xgboost as xgb
        dtr = xgb.DMatrix(np.asarray(d["Xtr"]), label=ytr)
        params = {"max_depth": 8, "eta": 0.1, "tree_method": "hist",
                  "objective": "binary:logistic", "eval_metric": "aucpr",
                  "scale_pos_weight": float((ytr == 0).sum() / (ytr == 1).sum()), "seed": a.seed, "nthread": -1}
        modelo_path = FEAT / "xgb_ibm.json"
        booster = None
        if a.continuar and modelo_path.exists():
            booster = xgb.Booster(); booster.load_model(str(modelo_path))
        booster = xgb.train(params, dtr, num_boost_round=a.arboles, xgb_model=booster)
        booster.save_model(str(modelo_path))
        pva = booster.predict(xgb.DMatrix(np.asarray(d["Xva"])))
        pte = booster.predict(xgb.DMatrix(np.asarray(d["Xte"])))
        extra = {"arboles_totales": "acumulado en xgb_ibm.json"}
    R = {"dataset": "ibm", "modelo": a.modelo, "seed": a.seed, "n_train": int(len(ytr)), **extra,
         **evaluar(d["yva"], pva, d["yte"], pte)}
    (AQUI / "resultados" / f"ibm_{a.modelo}_s{a.seed}.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    np.save(FEAT / f"pred_te_{a.modelo}.npy", pte)
    print(json.dumps(R, indent=2))

if __name__ == "__main__":
    main()
