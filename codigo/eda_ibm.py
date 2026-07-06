"""M2 - EDA IBM AML HI-Small. Uso: python eda_ibm.py --parte perfil|patrones|figuras"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "datos" / "ibm_aml"
TRANS = DATOS / "HI-Small_Trans.csv"
PATRONES = DATOS / "HI-Small_Patterns.txt"

def cargar(usecols=None):
    df = pd.read_csv(TRANS, engine="pyarrow", usecols=usecols)
    return df

def perfil():
    R = {}
    df = cargar()
    df.columns = ["ts", "banco_o", "cta_o", "banco_d", "cta_d", "monto_rec", "mon_rec",
                  "monto_pag", "mon_pag", "formato", "lavado"]
    R["filas"], R["columnas"] = int(df.shape[0]), int(df.shape[1])
    R["nulos_por_col"] = {c: int(n) for c, n in df.isna().sum().items() if n > 0} or "ninguno"
    R["duplicados_exactos"] = int(df.duplicated().sum())
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    R["rango_fechas"] = [str(df["ts"].min()), str(df["ts"].max())]
    R["dias"] = int((df["ts"].max().normalize() - df["ts"].min().normalize()).days) + 1
    R["cuentas_unicas"] = int(pd.concat([df["cta_o"], df["cta_d"]]).nunique())
    R["bancos_unicos"] = int(pd.concat([df["banco_o"], df["banco_d"]]).nunique())
    R["self_transfers_misma_cuenta"] = int((df["cta_o"] == df["cta_d"]).sum())
    R["monedas"] = df["mon_rec"].value_counts().head(8).to_dict()
    R["formatos"] = df["formato"].value_counts().to_dict()
    R["mismatch_monto_pag_rec"] = int((df["monto_pag"] != df["monto_rec"]).sum())
    R["mismatch_moneda"] = int((df["mon_pag"] != df["mon_rec"]).sum())
    m = df["monto_rec"].astype(float)
    R["monto_rec"] = {"min": float(m.min()), "p50": float(m.median()), "p99": float(m.quantile(0.99)),
                      "max": float(m.max()), "ceros": int((m == 0).sum()), "negativos": int((m < 0).sum())}
    # series diarias
    d = df.set_index("ts").resample("D").agg(vol=("lavado", "size"), lav=("lavado", "sum"))
    d["tasa"] = d["lav"] / d["vol"]
    d.to_csv(AQUI / "ibm_diario.csv")
    # split temporal 60/20/20 por CUANTILES DE VOLUMEN (D009): el volumen diario es
    # muy desigual (colapsa tras los primeros dias), un corte por dias calendario
    # dejaria train=99.98%. El corte por cuantiles mantiene el orden temporal estricto.
    c1, c2 = df["ts"].quantile(0.6), df["ts"].quantile(0.8)
    R["split"] = {"corte_train_val": str(c1), "corte_val_test": str(c2)}
    for nombre, mask in [("train", df["ts"] < c1), ("val", (df["ts"] >= c1) & (df["ts"] < c2)), ("test", df["ts"] >= c2)]:
        sub = df.loc[mask, "lavado"]
        R["split"][nombre] = {"tx": int(mask.sum()), "lavado": int(sub.sum()), "tasa": round(float(sub.mean()), 5)}
    (AQUI / "ibm_perfil.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    print(json.dumps(R, indent=2, default=str))

def patrones():
    R = {"intentos": 0}
    filas, actual, tipo = [], None, None
    for linea in PATRONES.read_text(encoding="utf-8", errors="ignore").splitlines():
        if linea.startswith("BEGIN LAUNDERING ATTEMPT"):
            tipo = linea.split("-", 1)[1].strip().upper()
            actual = []
        elif linea.startswith("END LAUNDERING ATTEMPT"):
            if actual is not None:
                ts = pd.to_datetime([t.split(",")[0] for t in actual], format="%Y/%m/%d %H:%M")
                filas.append({"tipo": tipo, "n_tx": len(actual),
                              "inicio": ts.min(), "fin": ts.max(),
                              "duracion_h": round((ts.max() - ts.min()).total_seconds() / 3600, 1)})
            actual, tipo = None, None
        elif actual is not None and linea.strip():
            actual.append(linea)
    P = pd.DataFrame(filas)
    P.to_csv(AQUI / "ibm_patrones.csv", index=False)
    R["intentos"] = len(P)
    R["tx_totales_en_patrones"] = int(P["n_tx"].sum())
    R["tx_por_intento"] = {"min": int(P["n_tx"].min()), "p50": float(P["n_tx"].median()), "max": int(P["n_tx"].max())}
    R["duracion_horas"] = {"p50": float(P["duracion_h"].median()), "max": float(P["duracion_h"].max())}
    R["por_tipologia"] = P.groupby("tipo")["n_tx"].agg(["count", "median", "sum"]).astype(float).to_dict("index")
    R["rango_temporal_patrones"] = [str(P["inicio"].min()), str(P["fin"].max())]
    (AQUI / "ibm_patrones.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    print(json.dumps(R, indent=2, default=str))

def figuras():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(AQUI / "ibm_diario.csv", parse_dates=["ts"])
    perfil_json = json.loads((AQUI / "ibm_perfil.json").read_text(encoding="utf-8"))
    c1 = pd.to_datetime(perfil_json["split"]["corte_train_val"]); c2 = pd.to_datetime(perfil_json["split"]["corte_val_test"])
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].bar(d["ts"], d["vol"], color="#4878a8", width=0.8)
    ax[0].set_ylabel("Transactions/day"); ax[0].set_title("IBM AML HI-Small: daily volume and laundering rate")
    ax[1].plot(d["ts"], 100 * d["tasa"], "o-", color="#c44e52", ms=4)
    ax[1].set_ylabel("Laundering rate (%)"); ax[1].set_xlabel("Date")
    for a in ax:
        a.axvline(c1, color="k", ls="--", lw=1); a.axvline(c2, color="k", ls="--", lw=1)
    ax[0].annotate("train", xy=(0.25, 0.9), xycoords="axes fraction"); ax[0].annotate("val", xy=(0.68, 0.9), xycoords="axes fraction"); ax[0].annotate("test", xy=(0.88, 0.9), xycoords="axes fraction")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(AQUI / "figuras" / "F3_ibm_temporal.png", dpi=150); plt.close(fig)
    # F4: montos log + F5: patrones
    df = cargar(usecols=["Amount Received", "Is Laundering"])
    df.columns = ["monto", "lavado"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    bins = np.logspace(np.log10(max(df["monto"].min(), 0.01)), np.log10(df["monto"].max()), 60)
    ax[0].hist(df.loc[df["lavado"] == 0, "monto"], bins=bins, alpha=0.7, label="Licit", color="#4878a8", density=True)
    ax[0].hist(df.loc[df["lavado"] == 1, "monto"], bins=bins, alpha=0.7, label="Laundering", color="#c44e52", density=True)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend(frameon=False)
    ax[0].set_xlabel("Amount received (log)"); ax[0].set_ylabel("Density"); ax[0].set_title("Amount distribution by class")
    P = pd.read_csv(AQUI / "ibm_patrones.csv")
    P["familia"] = P["tipo"].str.split(":").str[0].str.strip()
    orden = P.groupby("familia")["n_tx"].median().sort_values().index
    ax[1].boxplot([P.loc[P["familia"] == t, "n_tx"] for t in orden], vert=False, tick_labels=orden)
    ax[1].set_xlabel("Transactions per attempt"); ax[1].set_title("Laundering attempt size by typology (n=370)")
    fig.tight_layout(); fig.savefig(AQUI / "figuras" / "F4_ibm_montos_patrones.png", dpi=150); plt.close(fig)
    print("Figuras F3, F4 guardadas")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parte", choices=["perfil", "patrones", "figuras"], required=True)
    {"perfil": perfil, "patrones": patrones, "figuras": figuras}[ap.parse_args().parte]()
