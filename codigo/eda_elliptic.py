"""M2 - EDA Elliptic. Uso: python eda_elliptic.py --parte perfil|grafo|figuras
Reproducibilidad: si no existe el cache local, extrae del zip en datos/elliptic."""
import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
CACHE = Path("/sessions/intelligent-charming-einstein/elliptic_cache/elliptic_bitcoin_dataset")
if not CACHE.exists():  # fallback reproducible fuera del sandbox
    CACHE = RAIZ / "datos" / "elliptic" / "elliptic_bitcoin_dataset"
    if not CACHE.exists():
        z = zipfile.ZipFile(next((RAIZ / "datos" / "elliptic").glob("*.zip")))
        z.extractall(RAIZ / "datos" / "elliptic")

def cargar_features(solo_id_step=False):
    cols = [0, 1] if solo_id_step else None
    dt = {0: np.int64, 1: np.int16}
    if not solo_id_step:
        dt.update({i: np.float32 for i in range(2, 167)})
    df = pd.read_csv(CACHE / "elliptic_txs_features.csv", header=None, usecols=cols,
                     dtype=dt, engine="pyarrow")
    df.columns = ["txId", "step"] + [f"f{i}" for i in range(2, df.shape[1])]
    df["txId"] = df["txId"].astype(np.int64)
    df["step"] = df["step"].astype(np.int16)
    return df

def perfil():
    R = {}
    df = cargar_features()
    cl = pd.read_csv(CACHE / "elliptic_txs_classes.csv")
    R["filas"], R["columnas"] = df.shape
    R["nan_total"] = int(df.isna().sum().sum())
    R["txId_duplicados"] = int(df["txId"].duplicated().sum())
    R["steps"] = [int(df["step"].min()), int(df["step"].max())]
    m = df[["txId", "step"]].merge(cl, on="txId", how="left")
    m["clase"] = m["class"].map({"1": "ilicita", "2": "licita"}).fillna("desconocida")
    tabla = m.pivot_table(index="step", columns="clase", values="txId", aggfunc="count").fillna(0).astype(int)
    tabla["total"] = tabla.sum(axis=1)
    tabla["tasa_ilicita_etiquetadas"] = (tabla["ilicita"] / (tabla["ilicita"] + tabla["licita"])).round(4)
    tabla.to_csv(AQUI / "elliptic_clases_por_step.csv")
    R["tasa_ilicita_steps_1_42"] = round(float(tabla.loc[:42, "ilicita"].sum() / (tabla.loc[:42, "ilicita"] + tabla.loc[:42, "licita"]).sum()), 4)
    R["tasa_ilicita_steps_43_49"] = round(float(tabla.loc[43:, "ilicita"].sum() / (tabla.loc[43:, "ilicita"] + tabla.loc[43:, "licita"]).sum()), 4)
    # redundancia de features (para reduccion en M3)
    X = df.iloc[:, 2:].to_numpy(dtype=np.float32)
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    R["pares_features_|r|>0.95"] = int((np.abs(C[iu]) > 0.95).sum())
    R["pares_features_|r|>0.99"] = int((np.abs(C[iu]) > 0.99).sum())
    R["features_constantes"] = int((np.nanstd(X, axis=0) == 0).sum())
    (AQUI / "elliptic_perfil.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps(R, indent=2))

def grafo():
    R = {}
    ids = cargar_features(solo_id_step=True)
    ed = pd.read_csv(CACHE / "elliptic_txs_edgelist.csv")
    R["aristas"] = len(ed)
    paso = dict(zip(ids["txId"].to_numpy(), ids["step"].to_numpy()))
    s1 = ed["txId1"].map(paso); s2 = ed["txId2"].map(paso)
    R["aristas_endpoint_desconocido"] = int(s1.isna().sum() + s2.isna().sum())
    R["aristas_inter_step"] = int((s1 != s2).sum())
    R["self_loops"] = int((ed["txId1"] == ed["txId2"]).sum())
    R["aristas_duplicadas"] = int(ed.duplicated().sum())
    # grados
    gout = ed["txId1"].value_counts(); gin = ed["txId2"].value_counts()
    R["grado_out"] = {"max": int(gout.max()), "media": round(float(R["aristas"] / len(ids)), 3), "p99": int(gout.quantile(0.99))}
    R["grado_in"] = {"max": int(gin.max()), "p99": int(gin.quantile(0.99))}
    R["nodos_aislados"] = int(len(ids) - pd.concat([ed["txId1"], ed["txId2"]]).nunique())
    gout.to_frame("g").to_csv(AQUI / "elliptic_grado_out.csv")
    gin.to_frame("g").to_csv(AQUI / "elliptic_grado_in.csv")
    # componentes conexas (scipy, no dirigido)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = {t: i for i, t in enumerate(ids["txId"].to_numpy())}
    r = ed["txId1"].map(idx).to_numpy(); c = ed["txId2"].map(idx).to_numpy()
    n = len(idx)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    ncomp, etiq = connected_components(A, directed=False)
    tam = np.bincount(etiq)
    R["n_componentes"] = int(ncomp)
    R["mayores_componentes"] = sorted(tam.tolist(), reverse=True)[:5]
    (AQUI / "elliptic_grafo.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps(R, indent=2))

def figuras():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tabla = pd.read_csv(AQUI / "elliptic_clases_por_step.csv", index_col=0)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].bar(tabla.index, tabla["desconocida"], label="Unknown", color="#c9c9c9")
    ax[0].bar(tabla.index, tabla["licita"], bottom=tabla["desconocida"], label="Licit", color="#4878a8")
    ax[0].bar(tabla.index, tabla["ilicita"], bottom=tabla["desconocida"] + tabla["licita"], label="Illicit", color="#c44e52")
    ax[0].set_ylabel("Transactions"); ax[0].legend(frameon=False)
    ax[0].set_title("Elliptic: transactions per time step by class")
    ax[1].plot(tabla.index, 100 * tabla["tasa_ilicita_etiquetadas"], "o-", color="#c44e52", ms=3.5)
    ax[1].set_ylabel("Illicit rate (%)\n(labelled only)"); ax[1].set_xlabel("Time step")
    for a in ax:
        a.axvline(34.5, color="k", ls="--", lw=1)
        a.axvline(42.5, color="#8172b2", ls=":", lw=1.2)
    ax[1].annotate("train | test split", xy=(34.5, ax[1].get_ylim()[1] * 0.9), fontsize=8, ha="right", rotation=90)
    ax[1].annotate("dark market shutdown", xy=(42.5, ax[1].get_ylim()[1] * 0.9), fontsize=8, ha="right", rotation=90, color="#8172b2")
    fig.tight_layout(); fig.savefig(AQUI / "figuras" / "F1_elliptic_temporal.png", dpi=150); plt.close(fig)
    # F2: grados log-log
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, archivo, titulo in [(ax[0], "elliptic_grado_out.csv", "Out-degree"), (ax[1], "elliptic_grado_in.csv", "In-degree")]:
        g = pd.read_csv(AQUI / archivo)["g"]
        vc = g.value_counts().sort_index()
        a.loglog(vc.index, vc.values, "o", ms=3, alpha=0.6, color="#4878a8")
        a.set_xlabel(f"{titulo} k"); a.set_ylabel("# nodes"); a.set_title(f"Elliptic: {titulo.lower()} distribution")
    fig.tight_layout(); fig.savefig(AQUI / "figuras" / "F2_elliptic_grados.png", dpi=150); plt.close(fig)
    print("Figuras F1, F2 guardadas")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parte", choices=["perfil", "grafo", "figuras"], required=True)
    {"perfil": perfil, "grafo": grafo, "figuras": figuras}[ap.parse_args().parte]()
