"""Unit tests for the deterministic SAR factuality verifier and prompt builder
(codigo/generar_sar.py) -- the mechanism behind the 21x hallucination reduction."""
import pytest

from codigo.generar_sar import construir_prompt, verificar_factualidad


@pytest.fixture
def evidencia():
    return {
        "transaccion_focal": {
            "tx_id": 1001, "monto_recibido": 15000.0, "monto_pagado": 15000.0,
            "cuenta_origen": "A1B2C3D4", "cuenta_destino": "E5F6A7B8",
            "banco_origen": "BankOne", "banco_destino": "BankTwo",
            "timestamp": "2023-05-01 10:30",
        },
        "vecindario_relevante": [
            {"tx_id": 1002, "monto_recibido": 9000.0, "monto_pagado": 9000.0,
             "cuenta_origen": "E5F6A7B8", "cuenta_destino": "C0D1E2F3",
             "banco_origen": "BankTwo", "banco_destino": "BankThree",
             "timestamp": "2023-05-01 11:00"},
        ],
        "score_modelo": 0.87,
        "umbral_operativo": 0.5,
    }


def test_sar_limpio_no_tiene_alucinaciones(evidencia):
    sar = (
        "SUBJECT INFORMATION: account A1B2C3D4 at BankOne sent funds to E5F6A7B8 [tx_1001]. "
        "SUPPORTING TRANSACTION DETAIL: a further transfer to C0D1E2F3 for 9,000.00 [tx_1002]. "
        "The focal amount was 15,000.00 [tx_1001]."
    )
    r = verificar_factualidad(sar, evidencia)
    assert r["alucinaciones_detectadas"] == 0
    assert r["validez_citas"] == 1.0
    assert r["citas_invalidas"] == []


def test_sar_con_alucinaciones_se_detecta(evidencia):
    sar = (
        "The subject moved funds via account FFFFFFFF [tx_1001] for USD 50,000.00, "
        "then routed them onward [tx_9999]."
    )
    r = verificar_factualidad(sar, evidencia)
    assert "9999" in r["citas_invalidas"]
    assert "FFFFFFFF" in r["cuentas_o_bancos_sin_soporte"]
    assert any("50,000" in m for m in r["montos_sin_soporte"])
    assert r["alucinaciones_detectadas"] >= 3
    assert r["validez_citas"] == 0.5


def test_construir_prompt_por_condicion(evidencia):
    grounded = construir_prompt(evidencia, "grounded")
    assert "STRICT EVIDENCE CONTRACT" in grounded and "1001" in grounded
    raw = construir_prompt(evidencia, "raw")
    assert "0.87" in raw
    score_only = construir_prompt(evidencia, "score_only")
    assert "1001" in score_only
