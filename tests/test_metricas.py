"""Unit tests for the frozen metric protocol (codigo/metricas.py, decision D013)."""
import numpy as np
import pytest

from codigo.metricas import evaluar, f1_por_step, umbral_optimo_f1


@pytest.fixture
def separable():
    """A cleanly separable validation/test split (labels vs scores)."""
    y_val = np.array([0, 0, 0, 1, 1, 1])
    p_val = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y_test = np.array([0, 0, 1, 1])
    p_test = np.array([0.15, 0.25, 0.75, 0.85])
    return y_val, p_val, y_test, p_test


def test_umbral_optimo_f1_en_rango(separable):
    y_val, p_val, *_ = separable
    thr = umbral_optimo_f1(y_val, p_val)
    assert isinstance(thr, float)
    assert 0.0 <= thr <= 1.0


def test_evaluar_deteccion_perfecta(separable):
    r = evaluar(*separable)
    assert r["recall_test"] == 1.0
    assert r["precision_test"] == 1.0
    assert r["fp"] == 0 and r["fn"] == 0
    assert 0.0 <= r["pr_auc_test"] <= 1.0
    assert r["n_test"] == 4 and r["positivos_test"] == 2


def test_evaluar_cumple_contrato_de_claves(separable):
    r = evaluar(*separable)
    for k in ("pr_auc_test", "roc_auc_test", "umbral_val", "f1_test", "prevalencia_test"):
        assert k in r


def test_f1_por_step(separable):
    y_val, p_val, y_test, p_test = separable
    thr = umbral_optimo_f1(y_val, p_val)
    step = np.array([1, 1, 2, 2])
    out = f1_por_step(y_test, p_test, step, thr, steps=[1, 2])
    assert set(out).issubset({1, 2})
    assert all(0.0 <= v <= 1.0 for v in out.values())
