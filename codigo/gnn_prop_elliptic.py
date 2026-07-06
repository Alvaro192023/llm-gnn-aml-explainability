"""M5 - GNN de propagacion (familia SGC/SIGN) para Elliptic - corre en CPU.
Base matematica: paso de mensajes con adyacencia sym-normalizada precomputada;
features aumentadas [X, AX, A2X] + cabeza no lineal (XGB o MLP).
Uso: --cabeza xgb|mlp --variante af|lf [--seed 42] [--hops 2]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, eye, diags

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from metricas import evaluar, f1_por_step, umbral_optimo_f1

NPZ = Path("/sessions/intelligent-charming-einstein/procesado_cache/elliptic_grafo.npz")
if not NPZ.exists():
    NPZ = AQUI.parent / "datos" / "procesado" / "elliptic_grafo.npz"

def propagar(x, ei, n, hops):
    """A_hat = D^-1/2 (A+A^T+I) D^-1/2; devuelve [X, AX, ..., A^k X]."""
    r = np.concatenate([ei[0], ei[1], np.arange(n)])
    c = np.concatenate([ei[1], ei[0], np.arange(n)])
    A = coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0  # colapsar multiaristas
    d = np.asarray(A.sum(1)).ravel()
    Dm = diags((d ** -0.5).astype(np.float32))
    Ah = (Dm @ A @ Dm).astype(np.float32)
    caps, h = [x], x
    for _ in range(hops):
        h = Ah @ h
        caps.append(h)
    return np.hstack(caps).astype(np.float32)

class MLPNumpy:
    """MLP 2 capas, CE ponderada, Adam, early stopping por PR-AUC val (SIGN-style)."""
    def __init__(self, d, oculto, seed, peso_pos):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((d, oculto)) * np.sqrt(2 / d)).astype(np.float32)
        self.b1 = np.zeros(oculto, np.float32)
        self.W2 = (rng.standard_normal((oculto, 1)) * np.sqrt(2 / oculto)).astype(np.float32)
        self.b2 = np.zeros(1, np.float32)
        self.m = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self.v = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self.t = 0; self.peso_pos = peso_pos

    def forward(self, X):
        self.h = np.maximum(X @ self.W1 + self.b1, 0)
        return (self.h @ self.W2 + self.b2).ravel()

    def prob(self, X):
        return 1 / (1 + np.exp(-np.clip(self.forward(X), -30, 30)))

    def paso(self, X, y, lr=1e-3):
        n = len(y)
        z = self.forward(X)
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        w = np.where(y == 1, self.peso_pos, 1.0).astype(np.float32)
        g = (w * (p - y) / w.sum()).astype(np.float32)
        gW2 = self.h.T @ g[:, None]; gb2 = g.sum(keepdims=True)
        gh = np.outer(g, self.W2.ravel()) * (self.h > 0)
        gW1 = X.T @ gh; gb1 = gh.sum(0)
        self.t += 1
        for i, (par, gr) in enumerate(zip((self.W1, self.b1, self.W2, self.b2), (gW1, gb1, gW2, gb2))):
            self.m[i] = 0.9 * self.m[i] + 0.1 * gr
            self.v[i] = 0.999 * self.v[i] + 0.001 * gr * gr
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            par -= lr * mh / (np.sqrt(vh) + 1e-8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cabeza", choices=["xgb", "mlp"], required=True)
    ap.add_argument("--variante", choices=["af", "lf"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--epocas", type=int, default=600)
    ap.add_argument("--paciencia", type=int, default=80)
    a = ap.parse_args()
    np.random.seed(a.seed)
    D = np.load(NPZ)
    x = D["x"][:, :93] if a.variante == "lf" else D["x"]
    y, step = D["y"], D["step"]
    H = propagar(x, D["edge_index"], len(y), a.hops)
    etiq = y >= 0
    mtr = D["mask_train"] & etiq; mva = D["mask_val"] & etiq; mte = D["mask_test"] & etiq
    Xtr, ytr = H[mtr], y[mtr]; Xva, yva = H[mva], y[mva]; Xte, yte = H[mte], y[mte]
    from sklearn.metrics import average_precision_score
    if a.cabeza == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, tree_method="hist",
                          scale_pos_weight=float((ytr == 0).sum() / (ytr == 1).sum()),
                          eval_metric="aucpr", random_state=a.seed, n_jobs=-1)
        m.fit(Xtr, ytr)
        pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        extra = {}
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        m = MLPNumpy(Ztr.shape[1], 128, a.seed, peso_pos=float((ytr == 0).sum() / (ytr == 1).sum()))
        mejor = {"pr": -1, "pesos": None, "ep": -1}
        for ep in range(a.epocas):
            m.paso(Ztr, ytr.astype(np.float32))
            if ep % 5 == 0:
                pr = average_precision_score(yva, m.prob(Zva))
                if pr > mejor["pr"]:
                    mejor = {"pr": pr, "pesos": [p.copy() for p in (m.W1, m.b1, m.W2, m.b2)], "ep": ep}
                elif ep - mejor["ep"] >= a.paciencia:
                    break
        m.W1, m.b1, m.W2, m.b2 = mejor["pesos"]
        pva, pte = m.prob(Zva), m.prob(Zte)
        extra = {"mejor_epoca": mejor["ep"]}
    R = {"dataset": "elliptic", "modelo": f"gnnprop_{a.cabeza}", "variante": a.variante,
         "hops": a.hops, "seed": a.seed, **extra, **evaluar(yva, pva, yte, pte)}
    R["f1_por_step_test"] = f1_por_step(yte, pte, step[mte], umbral_optimo_f1(yva, pva), range(35, 50))
    (AQUI / "resultados" / f"elliptic_gnnprop_{a.cabeza}_{a.variante}_s{a.seed}.json").write_text(
        json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in R.items() if k != "f1_por_step_test"}, indent=2))

if __name__ == "__main__":
    main()
