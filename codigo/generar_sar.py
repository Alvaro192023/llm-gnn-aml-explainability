"""M6 - Generador de SAR con grounding y verificador de factualidad.
Condiciones experimentales (H3): grounded (JSON+contrato) | raw (volcado sin contrato) | score_only.
Backends: anthropic | openai | ollama (requieren clave/servidor; se configuran en M7).
Uso: --evidencia evidencias/evidencia_tx_X.json --condicion grounded --backend anthropic --modelo <id>
El verificador es importable: from generar_sar import verificar_factualidad"""
import argparse, json, re
from pathlib import Path

AQUI = Path(__file__).resolve().parent

PROMPT_GROUNDED = """You are a senior AML compliance officer drafting a Suspicious Activity Report (SAR) narrative.

STRICT EVIDENCE CONTRACT — violations invalidate the report:
1. Use ONLY facts present in the EVIDENCE JSON below. Do not introduce any account, bank, amount, date, or entity not present in it.
2. Every quantitative or factual claim MUST cite its source transaction as [tx_<tx_id>].
3. If information needed for a section is not in the evidence, write "Not available in evidence."
4. The model risk factors provided are the actual drivers of the alert; translate them faithfully, do not invent reasons.
5. Do not assert guilt; describe observed patterns and their consistency with known laundering typologies.

Produce the SAR narrative with EXACTLY these sections:
1. SUBJECT INFORMATION - accounts and banks involved (focal transaction parties).
2. SUSPICIOUS ACTIVITY SUMMARY - what, when, how much (aggregates allowed only if computable from listed transactions or provided aggregates).
3. OBSERVED PATTERN - plain-language description of the transaction flow structure seen in the evidence neighborhood; if consistent with a known typology (e.g., cycle, fan-in, scatter-gather), say so and explain the structural evidence.
4. SUPPORTING TRANSACTION DETAIL - the specific transactions, each cited as [tx_<id>].
5. BASIS FOR SUSPICION - the model's risk factors, translated for a compliance audience, with the alert score.
6. LIMITATIONS AND CONFIDENCE - evidence window, what was not examined, false-positive considerations.

EVIDENCE JSON:
{evidencia}
"""

PROMPT_RAW = """You are an AML compliance officer. A transaction was flagged (score {score}).
Here is the flagged transaction and nearby transactions as raw data. Write a SAR narrative.

{volcado}
"""

PROMPT_SCORE_ONLY = """You are an AML compliance officer. Transaction {tx_id} was flagged by our
detection model with risk score {score}. Write a SAR narrative for this alert.
"""

def construir_prompt(evidencia: dict, condicion: str) -> str:
    if condicion == "grounded":
        return PROMPT_GROUNDED.format(evidencia=json.dumps(evidencia, indent=1, ensure_ascii=False))
    if condicion == "raw":
        filas = [evidencia["transaccion_focal"]] + evidencia["vecindario_relevante"]
        volcado = "\n".join(json.dumps(f, ensure_ascii=False) for f in filas)
        return PROMPT_RAW.format(score=evidencia["score_modelo"], volcado=volcado)
    return PROMPT_SCORE_ONLY.format(tx_id=evidencia["transaccion_focal"]["tx_id"], score=evidencia["score_modelo"])

def llamar_llm(prompt: str, backend: str, modelo: str, temperatura: float = 0.0) -> str:
    if backend == "anthropic":
        import anthropic
        clave = (AQUI / "api_key.txt").read_text(encoding="utf-8").strip().removeprefix("anthropic:")
        cliente = anthropic.Anthropic(api_key=clave)
        # max_tokens holgado: el SAR de 6 secciones ronda 700-1000 tokens; se deja margen.
        # 'temperature' esta deprecado en los modelos de la familia 5 (D017): NO se envia.
        # El determinismo se obtiene por instruccion en el prompt + salida factual verificada.
        kw = dict(model=modelo, max_tokens=2500, messages=[{"role": "user", "content": prompt}])
        try:
            r = cliente.messages.create(**kw)
        except anthropic.BadRequestError:
            r = cliente.messages.create(temperature=temperatura, **kw)  # modelos antiguos que si lo aceptan
        texto = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        if not texto.strip():
            raise RuntimeError(f"respuesta sin texto (stop={r.stop_reason}, blocks={[b.type for b in r.content]})")
        return texto
    if backend == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(model=modelo, max_tokens=1500, temperature=temperatura,
                                             messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    if backend == "ollama":
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
                                     data=json.dumps({"model": modelo, "prompt": prompt, "stream": False,
                                                      "options": {"temperature": temperatura}}).encode())
        return json.loads(urllib.request.urlopen(req).read())["response"]
    raise ValueError(backend)

# ---------------- Verificador de factualidad (nucleo de la metrica de M7) ----------------
def verificar_factualidad(sar: str, evidencia: dict) -> dict:
    """Coteja citas, montos, cuentas y bancos del SAR contra la evidencia. Devuelve metricas."""
    txs = {str(evidencia["transaccion_focal"]["tx_id"]): evidencia["transaccion_focal"]}
    for v in evidencia["vecindario_relevante"]:
        txs[str(v["tx_id"])] = v
    citas = re.findall(r"\[tx[_ ]?(\d+)\]", sar)
    citas_validas = [c for c in citas if c in txs]
    citas_invalidas = [c for c in citas if c not in txs]
    montos_ev = set()
    for t in txs.values():
        for k in ("monto_recibido", "monto_pagado"):
            m = str(t[k])
            montos_ev |= {m, f"{float(m):,.2f}", f"{float(m):,.0f}", str(int(float(m)))}
    for r_ in (evidencia.get("entidad_origen_resumen", {}), evidencia.get("entidad_destino_resumen", {})):
        montos_ev |= {str(x) for x in r_.values()}
    cuentas_ev = set()
    fechas_ev = set()
    for t in txs.values():
        cuentas_ev |= {t["cuenta_origen"], t["cuenta_destino"], t["banco_origen"], t["banco_destino"]}
        fechas_ev |= set(re.findall(r"\d+", t["timestamp"]))  # anio/mes/dia/hora citables
    score = evidencia.get("score_modelo", "")
    lista_blanca = (montos_ev | cuentas_ev | fechas_ev | set(txs) |
                    {str(evidencia.get("umbral_operativo", "")), str(score), f"{score:.1f}", f"{score:.2f}", f"{score:.4f}"})
    montos_sar = re.findall(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,}(?:\.\d+)?)(?![\w])", sar)
    montos_sin_soporte = [m for m in set(montos_sar)
                          if m not in lista_blanca and m.replace(",", "") not in lista_blanca]
    cuentas_sar = set(re.findall(r"\b[0-9A-F]{7,10}\b", sar))
    cuentas_sin_soporte = sorted(cuentas_sar - cuentas_ev)
    return {"citas_totales": len(citas), "citas_validas": len(citas_validas),
            "citas_invalidas": citas_invalidas,
            "validez_citas": round(len(citas_validas) / max(len(citas), 1), 4),
            "montos_sin_soporte": sorted(montos_sin_soporte),
            "cuentas_o_bancos_sin_soporte": cuentas_sin_soporte,
            "alucinaciones_detectadas": len(citas_invalidas) + len(montos_sin_soporte) + len(cuentas_sin_soporte)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidencia", required=True)
    ap.add_argument("--condicion", choices=["grounded", "raw", "score_only"], default="grounded")
    ap.add_argument("--backend", choices=["anthropic", "openai", "ollama"], required=True)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--temperatura", type=float, default=0.0)
    a = ap.parse_args()
    ev = json.loads(Path(a.evidencia).read_text(encoding="utf-8"))
    sar = llamar_llm(construir_prompt(ev, a.condicion), a.backend, a.modelo, a.temperatura)
    ver = verificar_factualidad(sar, ev)
    base = Path(a.evidencia).stem.replace("evidencia_", "")
    salida = AQUI / "sars" / f"sar_{base}_{a.condicion}_{a.backend}.md"
    salida.parent.mkdir(exist_ok=True)
    encabezado = (f"---\ncaso: {base}\ncondicion: {a.condicion}\nbackend: {a.backend}\nmodelo: {a.modelo}\n"
                  f"temperatura: {a.temperatura}\nverificacion: {json.dumps(ver)}\n---\n\n")
    salida.write_text(encabezado + sar, encoding="utf-8")
    print(json.dumps(ver, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
