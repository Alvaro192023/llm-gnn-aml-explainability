"""M5 - GNN de propagacion para IBM AML (clasificacion de aristas) - CPU.
Propagacion direccional (in/out, 2 hops) sobre grafos de mensaje expandibles (D012):
train->aristas train, val->train+val, test->todas. Cabeza XGB sobre [ef, one-hot, h_src, h_dst].
Uso: [--seed 42] [--neg-ratio 50]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from metricas import evaluar

CACHE = Path("/sessions/intelligent-charming-einstein/procesado_cache")
if not CACHE.exists():
    CACHE = AQUI.parent / "datos" / "procesado"

def agrega_aristas_a_nodos(src, dst, mask, ef, n):
    """v2: media de features de arista entrantes y salientes por nodo (capa 0 estilo GINe)."""
    idx = np.flatnonzero(mask)
    acc_in = np.zeros((n, ef.shape[1]), np.float32); acc_out = np.zeros_like(acc_in)
    cnt_in = np.zeros(n, np.float32); cnt_out = np.zeros(n, np.float32)
    np.add.at(acc_in, dst[idx], ef[idx]); np.add.at(cnt_in, dst[idx], 1)
    np.add.at(acc_out, src[idx], ef[idx]); np.add.at(cnt_out, src[idx], 1)
    return np.hstack([acc_in / np.maximum(cnt_in, 1)[:, None], acc_out / np.maximum(cnt_out, 1)[:, None]])

def propaga_ventana(src, dst, mask, nf, ef=None, hops=2):
    n = len(nf)
    if ef is not None:  # v2 enriquecida
        nf = np.hstack([nf, agrega_aristas_a_nodos(src, dst, mask, ef, n)]).astype(np.float32)
    idx = np.flatnonzero(mask)
    unos = np.ones(len(idx), np.float32)
    Ain = coo_matrix((unos, (dst[idx], src[idx])), shape=(n, n)).tocsr()   # i recibe de j
    Aout = coo_matrix((unos, (src[idx], dst[idx])), shape=(n, n)).tocsr()  # i envia a j
    def rn(A):
        d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
        return diags((1.0 / d).astype(np.float32)) @ A
    Ain, Aout = rn(Ain), rn(Aout)
    h1i, h1o = Ain @ nf, Aout @ nf
    bloques = [nf, h1i, h1o]
    if hops >= 2:
        bloques += [Ain @ h1i, Aout @ h1o]
    return np.hstack(bloques).astype(np.float32)

def matriz(ef, oh, H, src, dst, idx):
    return np.hstack([ef[idx], oh[idx], H[src[idx]], H[dst[idx]]]).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-ratio", type=int, default=50)
    ap.add_argument("--enriquecido", action="store_true")
    ap.add_argument("--arboles", type=int, default=400)
    ap.add_argument("--hops", type=int, default=2)
    a = ap.parse_args()
    src = np.load(CACHE / "ibm_src.npy"); dst = np.load(CACHE / "ibm_dst.npy")
    ef = np.load(CACHE / "ibm_ef_float.npy"); cat = np.load(CACHE / "ibm_ef_cat.npy").astype(np.int16)
    y = np.load(CACHE / "ibm_y.npy"); sp = np.load(CACHE / "ibm_split.npy")
    nf1 = np.load(CACHE / "ibm_nodo_feats_hasta_c1.npy"); nf2 = np.load(CACHE / "ibm_nodo_feats_hasta_c2.npy")
    I15, I7 = np.eye(15, dtype=np.float32), np.eye(7, dtype=np.float32)
    oh = np.hstack([I15[cat[:, 0]], I15[cat[:, 1]], I7[cat[:, 2]]])
    efp = ef if a.enriquecido else None
    H_tr = propaga_ventana(src, dst, sp == 0, nf1, efp, a.hops)   # D012
    H_va = propaga_ventana(src, dst, sp <= 1, nf1, efp, a.hops)
    H_te = propaga_ventana(src, dst, np.ones_like(sp, bool), nf2, efp, a.hops)
    idx_tr = np.flatnonzero(sp == 0); idx_va = np.flatnonzero(sp == 1); idx_te = np.flatnonzero(sp == 2)
    rng = np.random.default_rng(a.seed)
    pos = idx_tr[y[idx_tr] == 1]; neg = idx_tr[y[idx_tr] == 0]
    sub = np.sort(np.concatenate([pos, rng.choice(neg, size=len(pos) * a.neg_ratio, replace=False)]))
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=a.arboles, max_depth=7, learning_rate=0.1, tree_method="hist",
                      scale_pos_weight=float(a.neg_ratio), eval_metric="aucpr",
                      random_state=a.seed, n_jobs=-1, subsample=0.9, colsample_bytree=0.9)
    m.fit(matriz(ef, oh, H_tr, src, dst, sub), y[sub])
    del H_tr

    def pred_chunks(H, idx, paso=150000):
        out = np.empty(len(idx), np.float32)
        for i in range(0, len(idx), paso):
            out[i:i + paso] = m.predict_proba(matriz(ef, oh, H, src, dst, idx[i:i + paso]))[:, 1]
        return out

    pva = pred_chunks(H_va, idx_va); del H_va
    pte = pred_chunks(H_te, idx_te); del H_te
    suf = "_v2" if a.enriquecido else ""
    R = {"dataset": "ibm", "modelo": "gnnprop_xgb" + suf, "seed": a.seed, "neg_ratio": a.neg_ratio,
         "hops": a.hops, "dims": "8ef+37oh+20src+20dst", **evaluar(y[idx_va], pva, y[idx_te], pte)}
    nombre_res = f"ibm_gnnprop_xgb{suf}_h{a.hops}_s{a.seed}.json" if a.hops != 2 else f"ibm_gnnprop_xgb{suf}_s{a.seed}.json"
    (AQUI / "resultados" / nombre_res).write_text(json.dumps(R, indent=2), encoding="utf-8")
    # higiene de artefactos: solo la configuracion CANONICA (hops=2, neg=50) usa el nombre canonico
    canon = (a.hops == 2 and a.neg_ratio == 50)
    variante = "" if canon else f"_h{a.hops}_nr{a.neg_ratio}"
    np.save(CACHE / f"ibm_gnnprop{suf}{variante}_probs_test_s{a.seed}.npy", pte)
    m.save_model(str(CACHE / f"ibm_gnnprop{suf}{variante}_modelo_s{a.seed}.json"))
    print(json.dumps(R, indent=2))

if __name__ == "__main__":
    main()
