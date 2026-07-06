"""M7 - Evaluacion de explicabilidad (H2).
--parte recall   : recuperacion de tipologias en los intentos 100% contenidos en test.
                   Evidencia top-20 (pesos de propagacion) vs baselines aleatorio (5 semillas)
                   y recencia, sobre el MISMO pool de candidatos; Wilcoxon pareado.
--parte fidelidad: test de delecion en los 4 casos demo (re-propagacion sin top-10 evidencia
                   vs sin 10 aleatorias del mismo pool).
Regla D011: ibm_arista_intento se usa SOLO como ground truth de evaluacion, jamas como insumo."""
import argparse, json
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
CACHE = Path("/sessions/intelligent-charming-einstein/procesado_cache")
if not CACHE.exists():
    CACHE = RAIZ / "datos" / "procesado"

def cargar():
    d = {n: np.load(CACHE / f"ibm_{n}.npy") for n in
         ["src", "dst", "ef_float", "y", "split", "arista_intento"]}
    d["probs"] = np.load(CACHE / "ibm_gnnprop_v2_probs_test_s42.npy")
    d["idx_te"] = np.flatnonzero(d["split"] == 2)
    return d

def indices(nodos, n_nodos):
    orden = np.argsort(nodos, kind="stable").astype(np.int64)
    lim = np.searchsorted(nodos[orden], np.arange(0, n_nodos + 1))
    return orden, lim

def inc(orden, lim, n):
    return orden[lim[n]:lim[n + 1]]

def vecindario(e_id, d, os_, ls_, od_, ld_, k=20, tau=None):
    """tau (dias): kernel temporal exp(-|dt|/tau) sobre el peso (v2). None = v1."""
    """Replica el ranking del extractor M6. Devuelve (top_k_ids, pool_ids)."""
    src, dst, ef = d["src"], d["dst"], d["ef_float"]
    u, v = int(src[e_id]), int(dst[e_id])
    pesos = {}
    def agrega(nodo, w_base=1.0):
        sal = inc(os_, ls_, nodo); ent = inc(od_, ld_, nodo)
        for ids, deg in [(ent, max(len(ent), 1)), (sal, max(len(sal), 1))]:
            for j in ids:
                j = int(j)
                if j == e_id: continue
                w = w_base * (1.0 / deg) * (1.0 + abs(float(ef[j, 0])))
                if j not in pesos or pesos[j] < w:
                    pesos[j] = w
    agrega(u); agrega(v)
    hop1 = sorted(pesos.items(), key=lambda kv: -kv[1])[:12]
    for j, w1 in hop1[:6]:
        contra = int(src[j]) if int(dst[j]) in (u, v) else int(dst[j])
        sal = inc(os_, ls_, contra); ent = inc(od_, ld_, contra)
        for ids, deg in [(ent, max(len(ent), 1)), (sal, max(len(sal), 1))]:
            for kk in ids[:200]:
                kk = int(kk)
                if kk == e_id or kk in pesos: continue
                pesos[kk] = w1 * (1.0 / deg) * (1.0 + abs(float(ef[kk, 0])))
    if tau is not None:
        t0 = float(d["ef_float"][e_id, 5])
        pesos = {j: w * float(np.exp(-abs(float(d["ef_float"][j, 5]) - t0) / tau)) for j, w in pesos.items()}
    orden_p = sorted(pesos.items(), key=lambda kv: -kv[1])
    return [j for j, _ in orden_p[:k]], np.array([j for j, _ in orden_p], dtype=np.int64)

def parte_recall():
    d = cargar()
    n_nodos = int(max(d["src"].max(), d["dst"].max())) + 1
    os_, ls_ = indices(d["src"], n_nodos); od_, ld_ = indices(d["dst"], n_nodos)
    ai, sp, probs, idx_te = d["arista_intento"], d["split"], d["probs"], d["idx_te"]
    rng = np.random.default_rng(42)
    filas = []
    import pandas as pd
    I = pd.read_csv(CACHE / "ibm_intentos.csv").set_index("attempt_id")
    for aid in np.unique(ai[ai >= 0]):
        m = np.flatnonzero(ai == aid)
        if not (sp[m] == 2).all() or len(m) < 3:
            continue
        p = probs[np.searchsorted(idx_te, m)]
        focal = int(m[np.argmax(p)])
        gt = set(int(x) for x in m if x != focal)
        top, pool = vecindario(focal, d, os_, ls_, od_, ld_, k=20)
        rec_ev = len(gt & set(top)) / len(gt)
        top2, _ = vecindario(focal, d, os_, ls_, od_, ld_, k=20, tau=3.0)
        rec_ev2 = len(gt & set(top2)) / len(gt)
        t_focal = float(d["ef_float"][focal, 5])
        prox = pool[np.argsort(np.abs(d["ef_float"][pool, 5] - t_focal))][:20]
        rec_ev3 = len(gt & set(prox.tolist())) / len(gt)
        rec_pool = len(gt & set(pool.tolist())) / len(gt)
        rec_rnd = float(np.mean([len(gt & set(rng.choice(pool, size=min(20, len(pool)), replace=False).tolist())) / len(gt)
                                 for _ in range(5)])) if len(pool) else 0.0
        recientes = pool[np.argsort(-d["ef_float"][pool, 5])][:20]
        rec_rec = len(gt & set(recientes.tolist())) / len(gt)
        filas.append({"attempt_id": int(aid), "familia": I.loc[int(aid), "familia"], "n_tx": len(m),
                      "pool": int(len(pool)), "recall_evidencia": round(rec_ev, 4),
                      "recall_evidencia_v2_temporal": round(rec_ev2, 4),
                      "recall_evidencia_v3_proximidad": round(rec_ev3, 4),
                      "recall_pool_cota_superior": round(rec_pool, 4),
                      "recall_aleatorio": round(rec_rnd, 4), "recall_recencia": round(rec_rec, 4)})
    F = pd.DataFrame(filas)
    F.to_csv(AQUI / "h2_recall_por_intento.csv", index=False)
    from scipy.stats import wilcoxon
    w_rnd = wilcoxon(F["recall_evidencia_v3_proximidad"], F["recall_aleatorio"], alternative="greater")
    w_rec = wilcoxon(F["recall_evidencia_v3_proximidad"], F["recall_recencia"], alternative="greater", zero_method="zsplit")
    R = {"n_intentos_evaluados": int(len(F)),
         "recall_evidencia_media": round(float(F["recall_evidencia"].mean()), 4),
         "recall_evidencia_v2_media": round(float(F["recall_evidencia_v2_temporal"].mean()), 4),
         "recall_evidencia_v3_media": round(float(F["recall_evidencia_v3_proximidad"].mean()), 4),
         "recall_pool_cota_superior_media": round(float(F["recall_pool_cota_superior"].mean()), 4),
         "recall_aleatorio_media": round(float(F["recall_aleatorio"].mean()), 4),
         "recall_recencia_media": round(float(F["recall_recencia"].mean()), 4),
         "wilcoxon_vs_aleatorio_p": float(w_rnd.pvalue), "wilcoxon_vs_recencia_p": float(w_rec.pvalue),
         "por_familia": {f: {"n": int(len(g)), "recall_ev_v1": round(float(g["recall_evidencia"].mean()), 4),
                             "recall_ev_v2": round(float(g["recall_evidencia_v2_temporal"].mean()), 4),
                             "recall_ev_v3": round(float(g["recall_evidencia_v3_proximidad"].mean()), 4),
                             "recall_recencia": round(float(g["recall_recencia"].mean()), 4),
                             "cota_pool": round(float(g["recall_pool_cota_superior"].mean()), 4)}
                         for f, g in F.groupby("familia")}}
    (AQUI / "h2_recall_resumen.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps(R, indent=2))

def parte_fidelidad(casos):
    import importlib.util, sys, xgboost as xgb
    spec = importlib.util.spec_from_file_location("gpi", RAIZ / "M04_Modelos" / "gnn_prop_ibm.py")
    gpi = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(gpi)
    d = cargar()
    n_nodos = int(max(d["src"].max(), d["dst"].max())) + 1
    os_, ls_ = indices(d["src"], n_nodos); od_, ld_ = indices(d["dst"], n_nodos)
    nf2 = np.load(CACHE / "ibm_nodo_feats_hasta_c2.npy")
    cat = np.load(CACHE / "ibm_ef_cat.npy").astype(np.int16)
    I15, I7 = np.eye(15, dtype=np.float32), np.eye(7, dtype=np.float32)
    oh = np.hstack([I15[cat[:, 0]], I15[cat[:, 1]], I7[cat[:, 2]]])
    booster = xgb.Booster(); booster.load_model(str(CACHE / "ibm_gnnprop_v2_modelo_s42.json"))
    rng = np.random.default_rng(42)
    H_CACHE = CACHE / "H_te_cache.npy"
    def score(e_id, quitar=()):
        if not quitar and H_CACHE.exists():
            H = np.load(H_CACHE, mmap_mode="r")
        else:
            mask = np.ones(len(d["src"]), bool)
            for q in quitar: mask[q] = False
            H = gpi.propaga_ventana(d["src"], d["dst"], mask, nf2, d["ef_float"])
            if not quitar:
                np.save(H_CACHE, H)
        fila = np.hstack([d["ef_float"][e_id], oh[e_id], H[int(d["src"][e_id])], H[int(d["dst"][e_id])]])[None, :].astype(np.float32)
        m = float(booster.predict(xgb.DMatrix(fila), output_margin=True)[0])
        return 1.0 / (1.0 + np.exp(-m)), m
    resultados = []
    for e_id in casos:
        top, pool = vecindario(e_id, d, os_, ls_, od_, ld_, k=10, tau=3.0)
        s0, m0 = score(e_id)
        s_ev, m_ev = score(e_id, top)
        s_rn, m_rn = score(e_id, rng.choice(pool, size=min(10, len(pool)), replace=False).tolist())
        resultados.append({"arista": int(e_id), "score_original": round(s0, 4),
                           "score_sin_top10_evidencia": round(s_ev, 4),
                           "score_sin_10_aleatorias": round(s_rn, 4),
                           "fidelidad_delta_prob": round(s0 - s_ev, 4), "control_delta_prob": round(s0 - s_rn, 4),
                           "logit_original": round(m0, 3),
                           "fidelidad_delta_logit": round(m0 - m_ev, 3), "control_delta_logit": round(m0 - m_rn, 3)})
        print(json.dumps(resultados[-1]))
    ruta = AQUI / "h2_fidelidad.json"
    previos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []
    previos += resultados
    ruta.write_text(json.dumps(previos, indent=2), encoding="utf-8")

def casos_de_intentos(desde, hasta):
    """Aristas focales (max score) de los intentos 100% en test con >=3 tx, orden por attempt_id."""
    d = cargar()
    ai, sp, probs, idx_te = d["arista_intento"], d["split"], d["probs"], d["idx_te"]
    focales = []
    for aid in np.unique(ai[ai >= 0]):
        m = np.flatnonzero(ai == aid)
        if not (sp[m] == 2).all() or len(m) < 3:
            continue
        focales.append(int(m[np.argmax(probs[np.searchsorted(idx_te, m)])]))
    return focales[desde:hasta]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", choices=["recall", "fidelidad", "fidelidad_lote"], required=True)
    ap.add_argument("--casos", default="")
    ap.add_argument("--desde", type=int, default=0)
    ap.add_argument("--hasta", type=int, default=3)
    a = ap.parse_args()
    if a.parte == "recall":
        parte_recall()
    elif a.parte == "fidelidad_lote":
        parte_fidelidad(casos_de_intentos(a.desde, a.hasta))
    else:
        parte_fidelidad([int(x) for x in a.casos.split(",") if x])
