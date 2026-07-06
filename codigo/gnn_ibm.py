"""M4 - GNN IBM AML (clasificacion de aristas, GINe) - ejecutar en Colab (M5).
Uso: python gnn_ibm.py [--seed 42] [--datos CARPETA_procesado]
D012 (grafos de mensaje expandibles): train usa SOLO aristas train; val usa train+val; test usa todas.
Entrenamiento: por epoca se toman todos los positivos + negativos submuestreados (50:1) como aristas
etiquetadas, con muestreo de vecinos sobre el grafo de mensaje train. D013: umbral en val.
Regla dura D011: ibm_arista_intento NO se carga aqui."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GINEConv

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from metricas import evaluar

class MLP(torch.nn.Sequential):
    def __init__(self, e, s):
        super().__init__(torch.nn.Linear(e, s), torch.nn.ReLU(), torch.nn.Linear(s, s))

class GNNAristas(torch.nn.Module):
    def __init__(self, d_nodo, n_entidades, oculto=64, capas=2, emb_id=16, emb_cat=8):
        super().__init__()
        self.emb_id = torch.nn.Embedding(n_entidades, emb_id)
        self.emb_cur = torch.nn.Embedding(15, emb_cat)
        self.emb_fmt = torch.nn.Embedding(7, emb_cat)
        self.enc_nodo = torch.nn.Linear(d_nodo + emb_id, oculto)
        self.enc_arista = MLP(8 + 3 * emb_cat, oculto)
        self.convs = torch.nn.ModuleList([GINEConv(MLP(oculto, oculto), edge_dim=oculto) for _ in range(capas)])
        self.cabeza = torch.nn.Sequential(torch.nn.Linear(3 * oculto, oculto), torch.nn.ReLU(),
                                          torch.nn.Dropout(0.3), torch.nn.Linear(oculto, 1))

    def enc_e(self, ef, ec):
        return self.enc_arista(torch.cat([ef, self.emb_cur(ec[:, 0]), self.emb_cur(ec[:, 1]), self.emb_fmt(ec[:, 2])], dim=1))

    def forward(self, batch, ef_lbl, ec_lbl):
        h = F.relu(self.enc_nodo(torch.cat([batch.x, self.emb_id(batch.n_id)], dim=1)))
        e = self.enc_e(batch.edge_attr, batch.edge_cat)
        for conv in self.convs:
            h = F.relu(conv(h, batch.edge_index, e))
        s, d = batch.edge_label_index
        return self.cabeza(torch.cat([h[s], h[d], self.enc_e(ef_lbl, ec_lbl)], dim=1)).squeeze(-1)

def hacer_data(nf, src, dst, ef, ec, mascara):
    idx = np.flatnonzero(mascara)
    return Data(x=torch.tensor(nf), n_id=torch.arange(len(nf)),
                edge_index=torch.tensor(np.vstack([src[idx], dst[idx]]), dtype=torch.long),
                edge_attr=torch.tensor(ef[idx]), edge_cat=torch.tensor(ec[idx], dtype=torch.long),
                num_nodes=len(nf)), idx

def predecir(modelo, data, src, dst, ef, ec, idx_eval, vecinos, batch, dev):
    loader = LinkNeighborLoader(data, num_neighbors=vecinos, batch_size=batch, shuffle=False,
                                edge_label_index=torch.tensor(np.vstack([src[idx_eval], dst[idx_eval]]), dtype=torch.long),
                                edge_label=torch.tensor(idx_eval, dtype=torch.long))
    probs = np.empty(len(idx_eval), dtype=np.float32); pos = 0
    modelo.eval()
    with torch.no_grad():
        for b in loader:
            b = b.to(dev)
            g = idx_eval_global = b.edge_label.cpu().numpy()
            ef_l = torch.tensor(ef[g], device=dev); ec_l = torch.tensor(ec[g], dtype=torch.long, device=dev)
            p = torch.sigmoid(modelo(b, ef_l, ec_l)).cpu().numpy()
            probs[pos:pos + len(p)] = p; pos += len(p)
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--paciencia", type=int, default=4)
    ap.add_argument("--oculto", type=int, default=64)
    ap.add_argument("--capas", type=int, default=2)
    ap.add_argument("--neg-ratio", type=int, default=50)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--vecinos", default="15,10")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--datos", default=str(AQUI.parent / "datos" / "procesado"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = Path(a.datos)
    src = np.load(C / "ibm_src.npy"); dst = np.load(C / "ibm_dst.npy")
    ef = np.load(C / "ibm_ef_float.npy"); ec = np.load(C / "ibm_ef_cat.npy")
    y = np.load(C / "ibm_y.npy"); sp = np.load(C / "ibm_split.npy")
    nf1 = np.load(C / "ibm_nodo_feats_hasta_c1.npy"); nf2 = np.load(C / "ibm_nodo_feats_hasta_c2.npy")
    vecinos = [int(v) for v in a.vecinos.split(",")]

    data_tr, idx_tr = hacer_data(nf1, src, dst, ef, ec, sp == 0)          # D012: mensaje = train
    data_va, _ = hacer_data(nf1, src, dst, ef, ec, sp <= 1)               # mensaje = train+val
    data_te, _ = hacer_data(nf2, src, dst, ef, ec, np.ones_like(sp, bool))  # mensaje = todo
    idx_va = np.flatnonzero(sp == 1); idx_te = np.flatnonzero(sp == 2)
    pos_tr = idx_tr[y[idx_tr] == 1]; neg_tr = idx_tr[y[idx_tr] == 0]

    modelo = GNNAristas(nf1.shape[1], len(nf1), a.oculto, a.capas).to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=a.lr)
    pw = torch.tensor(a.pos_weight, device=dev)
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(a.seed)
    mejor = {"pr": -1, "estado": None, "ep": -1}
    t0 = time.time()
    for ep in range(a.epocas):
        etiquetadas = np.concatenate([pos_tr, rng.choice(neg_tr, size=len(pos_tr) * a.neg_ratio, replace=False)])
        rng.shuffle(etiquetadas)
        loader = LinkNeighborLoader(data_tr, num_neighbors=vecinos, batch_size=a.batch, shuffle=True,
                                    edge_label_index=torch.tensor(np.vstack([src[etiquetadas], dst[etiquetadas]]), dtype=torch.long),
                                    edge_label=torch.tensor(etiquetadas, dtype=torch.long))
        modelo.train()
        for b in loader:
            b = b.to(dev)
            g = b.edge_label.cpu().numpy()
            ef_l = torch.tensor(ef[g], device=dev); ec_l = torch.tensor(ec[g], dtype=torch.long, device=dev)
            logit = modelo(b, ef_l, ec_l)
            loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(y[g], dtype=torch.float32, device=dev), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        p_va = predecir(modelo, data_va, src, dst, ef, ec, idx_va, vecinos, a.batch, dev)
        pr = average_precision_score(y[idx_va], p_va)
        print(f"epoca {ep}: PR-AUC val {pr:.4f}")
        if pr > mejor["pr"]:
            mejor = {"pr": pr, "estado": {k: v.detach().clone() for k, v in modelo.state_dict().items()}, "ep": ep}
        if ep - mejor["ep"] >= a.paciencia:
            break
    modelo.load_state_dict(mejor["estado"])
    p_va = predecir(modelo, data_va, src, dst, ef, ec, idx_va, vecinos, a.batch, dev)
    p_te = predecir(modelo, data_te, src, dst, ef, ec, idx_te, vecinos, a.batch, dev)
    R = {"dataset": "ibm", "modelo": "gnn_gine", "seed": a.seed, "oculto": a.oculto, "capas": a.capas,
         "neg_ratio": a.neg_ratio, "pos_weight": a.pos_weight, "mejor_epoca": mejor["ep"],
         "segundos": round(time.time() - t0, 1), "dispositivo": dev,
         **evaluar(y[idx_va], p_va, y[idx_te], p_te)}
    salida = AQUI / "resultados"; salida.mkdir(exist_ok=True)
    (salida / f"ibm_gnn_gine_s{a.seed}.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    np.save(salida / f"ibm_gnn_gine_s{a.seed}_probs_test.npy", p_te)
    print(json.dumps(R, indent=2))

if __name__ == "__main__":
    main()
