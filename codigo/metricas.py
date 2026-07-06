"""M4 - Protocolo de metricas congelado (D013).
Regla de oro: el umbral de decision se elige SOLO en validacion y se aplica a test."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

def umbral_optimo_f1(y_val, p_val):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])

def evaluar(y_val, p_val, y_test, p_test, y_train=None, p_train=None):
    """Devuelve dict con el protocolo D013 completo."""
    t = umbral_optimo_f1(y_val, p_val)
    pred = (p_test >= t).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    R = {"pr_auc_test": float(average_precision_score(y_test, p_test)),
         "roc_auc_test": float(roc_auc_score(y_test, p_test)),
         "pr_auc_val": float(average_precision_score(y_val, p_val)),
         "umbral_val": t,
         "f1_test": 2 * prec * rec / max(prec + rec, 1e-12),
         "precision_test": prec, "recall_test": rec,
         "tp": tp, "fp": fp, "fn": fn,
         "positivos_test": int((y_test == 1).sum()), "n_test": int(len(y_test)),
         "prevalencia_test": float((y_test == 1).mean())}
    if y_train is not None:
        R["pr_auc_train"] = float(average_precision_score(y_train, p_train))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in R.items()}

def f1_por_step(y, p, step, umbral, steps):
    """F1 de la clase ilicita por time step (Elliptic, robustez temporal)."""
    salida = {}
    pred = (p >= umbral).astype(int)
    for s in steps:
        m = step == s
        if (y[m] == 1).sum() == 0:
            continue
        tp = int(((pred[m] == 1) & (y[m] == 1)).sum())
        fp = int(((pred[m] == 1) & (y[m] == 0)).sum())
        fn = int(((pred[m] == 0) & (y[m] == 1)).sum())
        salida[int(s)] = round(2 * tp / max(2 * tp + fp + fn, 1), 4)
    return salida
