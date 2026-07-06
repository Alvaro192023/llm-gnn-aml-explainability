"""M3 - Construccion del multigrafo IBM AML (clasificacion de aristas).
Uso: python construir_grafo_ibm.py --parte cargar|grafo|patrones|nodos
Artefactos en cache local (sandbox) o datos/procesado (Colab). Splits D009; sin 9 duplicados (D010)."""
import argparse, json
from hashlib import blake2b
from pathlib import Path

def hstable(b):  # hash determinista entre procesos (reproducibilidad exacta)
    return int.from_bytes(blake2b(b, digest_size=8).digest(), "big")
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
TRANS = RAIZ / "datos" / "ibm_aml" / "HI-Small_Trans.csv"
PATRONES = RAIZ / "datos" / "ibm_aml" / "HI-Small_Patterns.txt"
LOCAL = Path("/sessions/intelligent-charming-einstein/procesado_cache")
SALIDA = LOCAL if LOCAL.parent.exists() else RAIZ / "datos" / "procesado"
SALIDA.mkdir(parents=True, exist_ok=True)
PARQUET = SALIDA / "ibm_procesado.parquet"
C1, C2 = pd.Timestamp("2022-09-06 13:36:00"), pd.Timestamp("2022-09-08 16:12:00")  # D009

def cargar():
    df = pd.read_csv(TRANS, engine="pyarrow")
    df.columns = ["ts", "banco_o", "cta_o", "banco_d", "cta_d", "monto_rec", "mon_rec",
                  "monto_pag", "mon_pag", "formato", "lavado"]
    df["row_id"] = np.arange(len(df), dtype=np.int32)
    dup = df.drop(columns="row_id").duplicated()
    df = df.loc[~dup].copy()
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    # clave de entidad: verificar unicidad de cuenta entre bancos
    par_o = (df["banco_o"].astype(str) + "|" + df["cta_o"].astype(str))
    par_d = (df["banco_d"].astype(str) + "|" + df["cta_d"].astype(str))
    n_cta = pd.concat([df["cta_o"], df["cta_d"]]).nunique()
    n_par = pd.concat([par_o, par_d]).nunique()
    clave_compuesta = bool(n_par != n_cta)
    if clave_compuesta:  # cuentas repetidas entre bancos -> clave banco|cuenta
        cods, cats = pd.factorize(pd.concat([par_o, par_d]))
    else:
        cods, cats = pd.factorize(pd.concat([df["cta_o"], df["cta_d"]]))
    n = len(df)
    df["src"] = cods[:n].astype(np.int32)
    df["dst"] = cods[n:].astype(np.int32)
    mon_cats = pd.concat([df["mon_rec"], df["mon_pag"]]).value_counts().index.tolist()
    fmt_cats = df["formato"].value_counts().index.tolist()
    df["cur_rec"] = pd.Categorical(df["mon_rec"], categories=mon_cats).codes.astype(np.int8)
    df["cur_pag"] = pd.Categorical(df["mon_pag"], categories=mon_cats).codes.astype(np.int8)
    df["fmt"] = pd.Categorical(df["formato"], categories=fmt_cats).codes.astype(np.int8)
    df["same_bank"] = (df["banco_o"] == df["banco_d"]).astype(np.int8)
    out = df[["row_id", "src", "dst", "ts", "monto_rec", "monto_pag",
              "cur_rec", "cur_pag", "fmt", "same_bank", "lavado"]]
    out.to_parquet(PARQUET, index=False)
    info = {"filas_post_dup": int(n), "duplicados_eliminados": int(dup.sum()),
            "n_entidades": int(len(cats)), "clave_compuesta_banco_cuenta": clave_compuesta,
            "monedas": mon_cats, "formatos": fmt_cats}
    (SALIDA / "ibm_encoders.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k not in ("monedas", "formatos")}, indent=2))
    print("monedas:", len(mon_cats), "| formatos:", len(fmt_cats))

def grafo():
    df = pd.read_parquet(PARQUET)
    t0 = df["ts"].min()
    ef = np.stack([
        np.log1p(df["monto_rec"].to_numpy(np.float64)).astype(np.float32),
        np.log1p(df["monto_pag"].to_numpy(np.float64)).astype(np.float32),
        (df["cur_rec"].to_numpy() == df["cur_pag"].to_numpy()).astype(np.float32),
        (df["src"].to_numpy() == df["dst"].to_numpy()).astype(np.float32),
        df["same_bank"].to_numpy(np.float32),
        ((df["ts"] - t0).dt.total_seconds().to_numpy() / 86400.0).astype(np.float32),
        np.sin(2 * np.pi * df["ts"].dt.hour.to_numpy() / 24).astype(np.float32),
        np.cos(2 * np.pi * df["ts"].dt.hour.to_numpy() / 24).astype(np.float32)], axis=1)
    split = np.full(len(df), 2, dtype=np.int8)
    split[df["ts"] < C2] = 1
    split[df["ts"] < C1] = 0
    np.save(SALIDA / "ibm_src.npy", df["src"].to_numpy(np.int32))
    np.save(SALIDA / "ibm_dst.npy", df["dst"].to_numpy(np.int32))
    np.save(SALIDA / "ibm_ef_float.npy", ef)
    np.save(SALIDA / "ibm_ef_cat.npy", df[["cur_rec", "cur_pag", "fmt"]].to_numpy(np.int8))
    np.save(SALIDA / "ibm_y.npy", df["lavado"].to_numpy(np.int8))
    np.save(SALIDA / "ibm_split.npy", split)
    np.save(SALIDA / "ibm_row_id.npy", df["row_id"].to_numpy(np.int32))
    R = {"aristas": int(len(df)),
         "ef_float": "log1p_monto_rec, log1p_monto_pag, misma_moneda, self_transfer, mismo_banco, dias_desde_inicio, hora_sin, hora_cos",
         "ef_cat": "cur_rec, cur_pag, formato (indices para embeddings)",
         "splits": {s: {"tx": int((split == i).sum()), "lavado": int(df["lavado"].to_numpy()[split == i].sum())}
                    for i, s in enumerate(["train", "val", "test"])}}
    (SALIDA / "ibm_meta.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps(R, indent=2))

def patrones():
    # matching por hash de linea cruda: las lineas de Patterns son copias exactas de Trans.csv
    hash_a_fila = {}
    colisiones_dup = 0
    with open(TRANS, "rb") as f:
        f.readline()  # encabezado
        for i, linea in enumerate(f):
            h = hstable(linea.rstrip(b"\r\n"))
            if h not in hash_a_fila:
                hash_a_fila[h] = i
            else:
                colisiones_dup += 1
    df = pd.read_parquet(PARQUET, columns=["row_id"])
    fila_a_arista = np.full(int(df["row_id"].max()) + 1, -1, dtype=np.int32)
    fila_a_arista[df["row_id"].to_numpy()] = np.arange(len(df), dtype=np.int32)
    arista_intento = np.full(len(df), -1, dtype=np.int16)
    intentos, actual, tipo, no_match = [], None, None, 0
    for linea in PATRONES.read_bytes().splitlines():
        if linea.startswith(b"BEGIN LAUNDERING ATTEMPT"):
            tipo, actual = linea.decode()[len("BEGIN LAUNDERING ATTEMPT - "):].strip(), []
        elif linea.startswith(b"END LAUNDERING ATTEMPT"):
            aid = len(intentos)
            ar = [int(fila_a_arista[hash_a_fila[hstable(t)]]) if hstable(t) in hash_a_fila else -1 for t in actual]
            no_match += sum(1 for a in ar if a < 0)
            for a in ar:
                if a >= 0:
                    arista_intento[a] = aid
            fam = tipo.split(":")[0].strip()
            intentos.append({"attempt_id": aid, "familia": fam, "subtipo": tipo,
                             "n_tx": len(actual), "n_tx_mapeadas": sum(1 for a in ar if a >= 0)})
            actual, tipo = None, None
        elif actual is not None and linea.strip():
            actual.append(linea.rstrip(b"\r\n"))
    I = pd.DataFrame(intentos)
    I.to_csv(SALIDA / "ibm_intentos.csv", index=False)
    np.save(SALIDA / "ibm_arista_intento.npy", arista_intento)
    R = {"intentos": len(I), "tx_en_patrones": int(I["n_tx"].sum()),
         "tx_mapeadas_a_aristas": int(I["n_tx_mapeadas"].sum()), "tx_sin_match": int(no_match),
         "lineas_trans_duplicadas_en_hash": colisiones_dup,
         "aristas_marcadas": int((arista_intento >= 0).sum())}
    print(json.dumps(R, indent=2))

def nodos():
    df = pd.read_parquet(PARQUET, columns=["src", "dst", "ts", "monto_rec"])
    enc = json.loads((SALIDA / "ibm_encoders.json").read_text(encoding="utf-8"))
    n = enc["n_entidades"]
    for nombre, corte in [("c1", C1), ("c2", C2)]:
        sub = df[df["ts"] < corte]
        feats = np.zeros((n, 4), dtype=np.float32)
        feats[:, 0] = np.log1p(np.bincount(sub["src"], minlength=n))              # grado out
        feats[:, 1] = np.log1p(np.bincount(sub["dst"], minlength=n))              # grado in
        feats[:, 2] = np.log1p(np.bincount(sub["src"], weights=sub["monto_rec"], minlength=n))  # fuerza out
        feats[:, 3] = np.log1p(np.bincount(sub["dst"], weights=sub["monto_rec"], minlength=n))  # fuerza in
        np.save(SALIDA / f"ibm_nodo_feats_hasta_{nombre}.npy", feats)
        print(f"nodo_feats_hasta_{nombre}: {feats.shape}, tx usadas: {len(sub):,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parte", choices=["cargar", "grafo", "patrones", "nodos"], required=True)
    {"cargar": cargar, "grafo": grafo, "patrones": patrones, "nodos": nodos}[ap.parse_args().parte]()
