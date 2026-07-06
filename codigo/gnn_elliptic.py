"""M4 - GNN Elliptic (clasificacion de nodos) - ejecutar en Colab (M5).
Uso: python gnn_elliptic.py --arquitectura gcn|sage|gat --variante af|lf [--seed 42] [--datos RUTA]
D012: en Elliptic los 49 steps son componentes aisladas (0 aristas inter-step, verificado por assert),
por lo que el paso de mensajes con el grafo completo NO filtra informacion entre splits.
D013: umbral en val, metricas en test. Resultados: resultados/elliptic_gnn_*.json"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_undirected

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from metricas import evaluar, f1_por_step, umbral_optimo_f1

class GNN(torch.nn.Module):
    def __init__(self, arquitectura, d_in, oculto=128, dropout=0.5):
        super().__init__()
        Conv = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}[arquitectura]
        kw = {"heads": 4, "concat": False} if arquitectura == "gat" else {}
        self.c1 = Conv(d_in, oculto, **kw)
        self.c2 = Conv(oculto, oculto, **kw)
        self.cabeza = torch.nn.Linear(oculto, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.c1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c2(h, edge_index))
        return self.cabeza(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquitectura", choices=["gcn", "sage", "gat"], required=True)
    ap.add_argument("--variante", choices=["af", "lf"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epocas", type=int, default=400)
    ap.add_argument("--paciencia", type=int, default=50)
    ap.add_argument("--oculto", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--datos", default=str(AQUI.parent / "datos" / "procesado" / "elliptic_grafo.npz"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    D = np.load(a.datos)
    x = D["x"][:, :93] if a.variante == "lf" else D["x"]
    step = D["step"]; y = D["y"]
    # D012 (verificacion): sin aristas inter-step
    ei = D["edge_index"]
    assert (step[ei[0]] == step[ei[1]]).all(), "aristas inter-step: revisar D012"
    X = torch.tensor(x, dtype=torch.float32, device=dev)
    Y = torch.tensor(y, dtype=torch.long, device=dev)
    EI = to_undirected(torch.tensor(ei, dtype=torch.long, device=dev))  # mensaje bidireccional
    etiq = y >= 0
    m_tr = torch.tensor(D["mask_train"] & etiq, device=dev)
    m_va = torch.tensor(D["mask_val"] & etiq, device=dev)
    m_te = torch.tensor(D["mask_test"] & etiq, device=dev)
    n_pos = int((y[D["mask_train"] & etiq] == 1).sum()); n_neg = int((y[D["mask_train"] & etiq] == 0).sum())
    pesos = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=dev)

    modelo = GNN(a.arquitectura, X.shape[1], a.oculto).to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=a.lr, weight_decay=5e-4)
    mejor = {"pr_auc": -1, "estado": None, "epoca": -1}
    from sklearn.metrics import average_precision_score
    t0 = time.time()
    for ep in range(a.epocas):
        modelo.train(); opt.zero_grad()
        logits = modelo(X, EI)
        loss = F.cross_entropy(logits[m_tr], Y[m_tr], weight=pesos)
        loss.backward(); opt.step()
        modelo.eval()
        with torch.no_grad():
            p = F.softmax(modelo(X, EI), dim=1)[:, 1].cpu().numpy()
        pr_va = average_precision_score(y[m_va.cpu().numpy()], p[m_va.cpu().numpy()])
        if pr_va > mejor["pr_auc"]:
            mejor = {"pr_auc": pr_va, "estado": {k: v.detach().clone() for k, v in modelo.state_dict().items()}, "epoca": ep}
        if ep - mejor["epoca"] >= a.paciencia:
            break
    modelo.load_state_dict(mejor["estado"]); modelo.eval()
    with torch.no_grad():
        p = F.softmax(modelo(X, EI), dim=1)[:, 1].cpu().numpy()
    mva, mte = m_va.cpu().numpy(), m_te.cpu().numpy()
    R = {"dataset": "elliptic", "modelo": f"gnn_{a.arquitectura}", "variante": a.variante, "seed": a.seed,
         "oculto": a.oculto, "lr": a.lr, "mejor_epoca": mejor["epoca"], "segundos": round(time.time() - t0, 1),
         "dispositivo": dev, **evaluar(y[mva], p[mva], y[mte], p[mte])}
    R["f1_por_step_test"] = f1_por_step(y[mte], p[mte], step[mte], umbral_optimo_f1(y[mva], p[mva]), range(35, 50))
    salida = AQUI / "resultados"; salida.mkdir(exist_ok=True)
    (salida / f"elliptic_gnn_{a.arquitectura}_{a.variante}_s{a.seed}.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    np.save(salida / f"elliptic_gnn_{a.arquitectura}_{a.variante}_s{a.seed}_probs.npy", p)
    print(json.dumps({k: v for k, v in R.items() if k != "f1_por_step_test"}, indent=2))

if __name__ == "__main__":
    main()
