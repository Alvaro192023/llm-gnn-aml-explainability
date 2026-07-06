"""M6 - Extractor de evidencia XAI para casos marcados por la GNN (IBM AML).
Produce JSON de evidencia ESTRICTAMENTE del lado del modelo (sin ground truth de patrones):
transaccion focal, factores TreeSHAP agrupados, vecindario 2-hops con pesos de propagacion
exactos, agregados por entidad, y transacciones crudas trazables por tx_id (= fila del CSV).
La seleccion de casos usa el ground truth SOLO para el manifiesto de evaluacion (M7), que se
guarda aparte y nunca entra al prompt. Uso: --auto | --aristas id1,id2,..."""
import argparse, json
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
CACHE = Path("/sessions/intelligent-charming-einstein/procesado_cache")
if not CACHE.exists():
    CACHE = RAIZ / "datos" / "procesado"
TRANS = RAIZ / "datos" / "ibm_aml" / "HI-Small_Trans.csv"
EF_NOMBRES = ["log_monto_recibido", "log_monto_pagado", "misma_moneda", "self_transfer",
              "mismo_banco", "dias_desde_inicio", "hora_sin", "hora_cos"]

def cargar_todo():
    d = {n: np.load(CACHE / f"ibm_{n}.npy") for n in
         ["src", "dst", "ef_float", "ef_cat", "y", "split", "row_id", "arista_intento"]}
    d["enc"] = json.loads((CACHE / "ibm_encoders.json").read_text(encoding="utf-8"))
    d["probs"] = np.load(CACHE / "ibm_gnnprop_v2_probs_test_s42.npy")
    d["idx_te"] = np.flatnonzero(d["split"] == 2)
    return d

def indice_incidencia(nodos, n_nodos):
    orden = np.argsort(nodos, kind="stable")
    limites = np.searchsorted(nodos[orden], np.arange(0, n_nodos + 1))
    return orden, limites

def incidentes(orden, limites, nodo):
    return orden[limites[nodo]:limites[nodo + 1]]

def fila_cruda(row_ids):
    objetivo = set(int(r) for r in row_ids)
    out = {}
    with open(TRANS, encoding="utf-8") as f:
        cab = f.readline().rstrip("\n").split(",")
        for i, linea in enumerate(f):
            if i in objetivo:
                out[i] = dict(zip(["timestamp", "banco_origen", "cuenta_origen", "banco_destino",
                                   "cuenta_destino", "monto_recibido", "moneda_recibida",
                                   "monto_pagado", "moneda_pagada", "formato", "_flag"], linea.rstrip("\n").split(",")))
                out[i].pop("_flag")
                if len(out) == len(objetivo):
                    break
    return out

def grupos_shap(contribs):
    """Agrupa las 245 contribuciones en factores legibles."""
    g = {}
    g["atributos de la transaccion (monto/moneda/formato/horario)"] = float(np.sum(contribs[0:45]))
    seg = ["historial propio", "recibido de contrapartes (1 salto)", "enviado a contrapartes (1 salto)",
           "actividad entrante a 2 saltos", "actividad saliente a 2 saltos"]
    for b, base in [(45, "cuenta ORIGEN"), (145, "cuenta DESTINO")]:
        for k, nombre in enumerate(seg):
            g[f"{base}: {nombre}"] = float(np.sum(contribs[b + k * 20: b + (k + 1) * 20]))
    return sorted(g.items(), key=lambda kv: -abs(kv[1]))

def evidencia_de(e_id, d, ord_src, lim_src, ord_dst, lim_dst, booster, H_te, oh, umbral, crudas_global=None):
    src, dst, ef = d["src"], d["dst"], d["ef_float"]
    u, v = int(src[e_id]), int(dst[e_id])
    # --- vecindario con pesos de propagacion exactos (1/grado por salto) ---
    vecinos = {}
    t_focal = float(ef[e_id, 5])
    TAU = 3.0  # kernel temporal (dias) — ranking v2, D016
    def agrega(nodo, rol):
        sal = incidentes(ord_src, lim_src, nodo); ent = incidentes(ord_dst, lim_dst, nodo)
        for ids, direccion, deg in [(ent, "entrante", max(len(ent), 1)), (sal, "saliente", max(len(sal), 1))]:
            for j in ids:
                if j == e_id: continue
                w = (1.0 / deg) * (1.0 + abs(float(ef[j, 0]))) * float(np.exp(-abs(float(ef[j, 5]) - t_focal) / TAU))
                if j not in vecinos or vecinos[j][0] < w:
                    vecinos[j] = (w, 1, rol, direccion)
    agrega(u, "origen"); agrega(v, "destino")
    hop1 = sorted(vecinos.items(), key=lambda kv: -kv[1][0])[:12]
    for j, (w1, _, rol, _) in list(hop1)[:6]:  # 2 saltos desde las contrapartes top
        contra = int(src[j]) if int(dst[j]) in (u, v) else int(dst[j])
        sal = incidentes(ord_src, lim_src, contra); ent = incidentes(ord_dst, lim_dst, contra)
        for ids, direccion, deg in [(ent, "entrante", max(len(ent), 1)), (sal, "saliente", max(len(sal), 1))]:
            for k in ids[:200]:
                if k == e_id or k in vecinos: continue
                w = w1 * (1.0 / deg) * (1.0 + abs(float(ef[k, 0]))) * float(np.exp(-abs(float(ef[k, 5]) - t_focal) / TAU))
                vecinos[k] = (w, 2, f"contraparte de {rol}", direccion)
    top = sorted(vecinos.items(), key=lambda kv: (-kv[1][0]))[:20]
    ids_necesarias = [e_id] + [int(j) for j, _ in top]
    if crudas_global is None:
        crudas = fila_cruda(d["row_id"][ids_necesarias])
    else:
        crudas = {int(d["row_id"][j]): crudas_global[int(d["row_id"][j])] for j in ids_necesarias}
    # --- factores del modelo (TreeSHAP exacto) ---
    import xgboost as xgb
    fila = np.hstack([ef[e_id], oh[e_id], H_te[u], H_te[v]]).astype(np.float32)[None, :]
    contribs = booster.predict(xgb.DMatrix(fila), pred_contribs=True)[0][:-1]
    factores = [{"factor": n, "contribucion": round(c, 4), "direccion": "eleva sospecha" if c > 0 else "reduce sospecha"}
                for n, c in grupos_shap(contribs)[:6]]
    # --- agregados por entidad (ventana completa D012-test) ---
    def resumen(nodo):
        ent = incidentes(ord_dst, lim_dst, nodo); sal = incidentes(ord_src, lim_src, nodo)
        return {"tx_entrantes": int(len(ent)), "tx_salientes": int(len(sal)),
                "monto_entrante_log1p_suma": round(float(ef[ent, 0].sum()), 2),
                "monto_saliente_log1p_suma": round(float(ef[sal, 0].sum()), 2),
                "contrapartes_distintas_entrada": int(len(np.unique(src[ent]))),
                "contrapartes_distintas_salida": int(len(np.unique(dst[sal])))}
    score = float(d["probs"][np.searchsorted(d["idx_te"], e_id)])
    return {"caso": f"tx_{int(d['row_id'][e_id])}",
            "score_modelo": round(score, 4), "umbral_operativo": umbral,
            "decision": "ALERTA" if score >= umbral else "sin alerta",
            "transaccion_focal": {"tx_id": int(d["row_id"][e_id]), **crudas.get(int(d["row_id"][e_id]), {})},
            "factores_del_modelo": factores,
            "entidad_origen_resumen": resumen(u), "entidad_destino_resumen": resumen(v),
            "vecindario_relevante": [
                {"tx_id": int(d["row_id"][j]), "salto": h, "relacion": f"{rol} ({direccion})",
                 "peso_relevancia": round(w, 5), **crudas.get(int(d["row_id"][j]), {})}
                for j, (w, h, rol, direccion) in top],
            "ventana_de_evidencia": "todas las transacciones historicas disponibles al momento de test (D012)",
            "nota_contrato": "Toda afirmacion del SAR debe citar tx_id presentes en este JSON."}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--aristas", default="")
    ap.add_argument("--umbral", type=float, default=0.5)
    a = ap.parse_args()
    d = cargar_todo()
    n_nodos = int(max(d["src"].max(), d["dst"].max())) + 1
    ord_src, lim_src = indice_incidencia(d["src"], n_nodos)
    ord_dst, lim_dst = indice_incidencia(d["dst"], n_nodos)
    # reconstruir H_te y oh (determinista) + booster
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("gpi", RAIZ / "M04_Modelos" / "gnn_prop_ibm.py")
    gpi = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(gpi)
    nf2 = np.load(CACHE / "ibm_nodo_feats_hasta_c2.npy")
    H_te = gpi.propaga_ventana(d["src"], d["dst"], np.ones_like(d["split"], bool), nf2, d["ef_float"])
    I15, I7 = np.eye(15, dtype=np.float32), np.eye(7, dtype=np.float32)
    cat = d["ef_cat"].astype(np.int16)
    class OHPorFila:  # one-hot bajo demanda (evita materializar 5M x 37 = 750 MB)
        def __getitem__(self, i):
            return np.concatenate([I15[cat[i, 0]], I15[cat[i, 1]], I7[cat[i, 2]]])
    oh = OHPorFila()
    import xgboost as xgb
    booster = xgb.Booster(); booster.load_model(str(CACHE / "ibm_gnnprop_v2_modelo_s42.json"))

    if a.auto:
        y, sp, ai, probs, idx_te = d["y"], d["split"], d["arista_intento"], d["probs"], d["idx_te"]
        # intentos 100% contenidos en test
        contenido = {}
        for aid in np.unique(ai[ai >= 0]):
            m = ai == aid
            contenido[int(aid)] = bool((sp[m] == 2).all())
        import pandas as pd
        I = pd.read_csv(CACHE / "ibm_intentos.csv")
        casos, manifiesto = [], []
        for familia in ["CYCLE", "SCATTER-GATHER", "FAN-IN"]:
            aids = [int(r.attempt_id) for r in I.itertuples() if r.familia == familia and contenido.get(int(r.attempt_id), False)]
            cand = [(float(probs[np.searchsorted(idx_te, e)]), int(e)) for aid in aids for e in np.flatnonzero(ai == aid) if sp[e] == 2]
            if not cand: continue
            p, e = max(cand)
            casos.append(e); manifiesto.append({"arista": e, "tipo": "TP", "familia_gt": familia,
                                                "attempt_id_gt": int(ai[e]), "score": round(p, 4)})
        fp_cand = idx_te[(y[idx_te] == 0)]
        p_fp = probs[np.searchsorted(idx_te, fp_cand)]
        e_fp = int(fp_cand[np.argmax(p_fp)])
        casos.append(e_fp); manifiesto.append({"arista": e_fp, "tipo": "FP", "familia_gt": None,
                                               "attempt_id_gt": None, "score": round(float(p_fp.max()), 4)})
        (AQUI / "evidencias" / "_manifiesto_evaluacion_M7.json").write_text(
            json.dumps(manifiesto, indent=2), encoding="utf-8")
        lista = casos
    else:
        lista = [int(x) for x in a.aristas.split(",") if x]
    # fase 1: recolectar row_ids de todos los casos (sin tocar el CSV)
    class _Sonda(dict):
        def __missing__(self, k): return {}
    necesarias = set()
    pre = {}
    for e_id in lista:
        ev = evidencia_de(e_id, d, ord_src, lim_src, ord_dst, lim_dst, booster, H_te, oh, a.umbral, crudas_global=_Sonda())
        pre[e_id] = ev
        necesarias.add(ev["transaccion_focal"]["tx_id"])
        necesarias |= {v["tx_id"] for v in ev["vecindario_relevante"]}
    # fase 2: UNA pasada del CSV para todas las filas
    crudas_global = fila_cruda(sorted(necesarias))
    # fase 3: re-emitir con las filas reales
    for e_id in lista:
        ev = evidencia_de(e_id, d, ord_src, lim_src, ord_dst, lim_dst, booster, H_te, oh, a.umbral, crudas_global=crudas_global)
        ruta = AQUI / "evidencias" / f"evidencia_{ev['caso']}.json"
        ruta.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(lista)} evidencias emitidas")

if __name__ == "__main__":
    main()
