"""M3 - Construccion del grafo Elliptic (clasificacion de nodos).
Artefactos: elliptic_grafo.npz + elliptic_meta.json (splits D009; features sin time step, D011).
En el sandbox escribe a cache local (se copia a datos/procesado/); en Colab escribe directo."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
CACHE = Path("/sessions/intelligent-charming-einstein/elliptic_cache/elliptic_bitcoin_dataset")
if not CACHE.exists():
    import zipfile
    CACHE = RAIZ / "datos" / "elliptic" / "elliptic_bitcoin_dataset"
    if not CACHE.exists():
        zipfile.ZipFile(next((RAIZ / "datos" / "elliptic").glob("*.zip"))).extractall(RAIZ / "datos" / "elliptic")
SALIDA = Path("/sessions/intelligent-charming-einstein/procesado_cache")
if not SALIDA.parent.exists():
    SALIDA = RAIZ / "datos" / "procesado"
SALIDA.mkdir(parents=True, exist_ok=True)

dt = {0: np.int64, 1: np.int16}; dt.update({i: np.float32 for i in range(2, 167)})
F = pd.read_csv(CACHE / "elliptic_txs_features.csv", header=None, dtype=dt, engine="pyarrow")
txid = F[0].to_numpy()
step = F[1].to_numpy(np.int16)
x = F.iloc[:, 2:].to_numpy(np.float32)          # 93 locales + 72 agregadas (sin txId ni step: D011)

cl = pd.read_csv(CACHE / "elliptic_txs_classes.csv")
y = pd.Series(-1, index=txid, dtype=np.int8)     # -1 = unknown (fuera de la perdida)
m = cl.set_index("txId")["class"]
y.loc[m.index[m == "1"]] = 1                     # ilicita
y.loc[m.index[m == "2"]] = 0                     # licita
y = y.to_numpy()

ed = pd.read_csv(CACHE / "elliptic_txs_edgelist.csv")
pos = pd.Series(np.arange(len(txid), dtype=np.int32), index=txid)
edge_index = np.vstack([pos.loc[ed["txId1"]].to_numpy(), pos.loc[ed["txId2"]].to_numpy()]).astype(np.int32)

mask_train = (step >= 1) & (step <= 29)
mask_val = (step >= 30) & (step <= 34)
mask_test = step >= 35
assert int(mask_train.sum() + mask_val.sum() + mask_test.sum()) == len(y)
# sin fuga estructural (M2): ninguna arista cruza steps
assert int((step[edge_index[0]] != step[edge_index[1]]).sum()) == 0

def cuenta(mask):
    return {"nodos": int(mask.sum()), "ilicitas": int((y[mask] == 1).sum()),
            "licitas": int((y[mask] == 0).sum()), "unknown": int((y[mask] == -1).sum())}

meta = {"artefacto": "elliptic_grafo.npz", "tarea": "clasificacion de nodos",
        "n_nodos": int(len(y)), "n_aristas": int(edge_index.shape[1]),
        "n_features": int(x.shape[1]), "features": "93 locales + 72 agregadas; SIN time step (D011: evitar atajo temporal); slices: locales=x[:,:93], agregadas=x[:,93:]",
        "splits_D009": {"train": "steps 1-29", "val": "steps 30-34", "test": "steps 35-49"},
        "conteos": {"train": cuenta(mask_train), "val": cuenta(mask_val), "test": cuenta(mask_test)}}
np.savez(SALIDA / "elliptic_grafo.npz", x=x, step=step, y=y, edge_index=edge_index,
         mask_train=mask_train, mask_val=mask_val, mask_test=mask_test)
(SALIDA / "elliptic_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
