"""H3 - Runner de generacion SAR con checkpoint (disenado para Colab; tambien corre local).
Genera SARs para todos los casos x condiciones x modelos, verifica factualidad y completitud
de rubrica, y anexa a h3_resultados.jsonl (reanudable: salta combinaciones ya hechas).
Uso: python correr_h3.py [--modelos claude-sonnet-5] [--condiciones grounded,raw,score_only]
     [--max-casos 62] [--hilos 4]"""
import argparse, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

AQUI = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(AQUI))
from generar_sar import construir_prompt, llamar_llm, verificar_factualidad

SECCIONES = ["SUBJECT INFORMATION", "SUSPICIOUS ACTIVITY SUMMARY", "OBSERVED PATTERN",
             "SUPPORTING TRANSACTION DETAIL", "BASIS FOR SUSPICION", "LIMITATIONS"]

def rubrica(sar: str) -> dict:
    presentes = [s for s in SECCIONES if re.search(s.replace(" ", r"\s+"), sar, re.IGNORECASE)]
    return {"secciones_presentes": len(presentes), "secciones_totales": len(SECCIONES),
            "faltantes": [s for s in SECCIONES if s not in presentes]}

def procesar(tarea):
    caso, condicion, modelo, ev = tarea
    t0 = time.time()
    try:
        sar = llamar_llm(construir_prompt(ev, condicion), "anthropic", modelo, 0.0)
        ver = verificar_factualidad(sar, ev)
        return {"caso": caso, "condicion": condicion, "modelo": modelo, "ok": True,
                "verificacion": ver, "rubrica": rubrica(sar), "segundos": round(time.time() - t0, 1),
                "sar": sar}
    except Exception as e:
        return {"caso": caso, "condicion": condicion, "modelo": modelo, "ok": False,
                "error": f"{type(e).__name__}: {e}", "segundos": round(time.time() - t0, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default="claude-sonnet-5")
    ap.add_argument("--condiciones", default="grounded,raw,score_only")
    ap.add_argument("--max-casos", type=int, default=62)
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    # cargar todas las evidencias e indexar por caso (los archivos usan row_id)
    todas = {}
    for f in (AQUI / "evidencias").glob("evidencia_tx_*.json"):
        ev = json.loads(f.read_text(encoding="utf-8"))
        todas[ev["caso"]] = ev
    salida = AQUI / "h3_resultados.jsonl"
    hechas = set()
    if salida.exists():
        for linea in salida.read_text(encoding="utf-8").splitlines():
            r = json.loads(linea)
            if r.get("ok"):
                hechas.add((r["caso"], r["condicion"], r["modelo"]))
    tareas = []
    lista_casos = sorted(todas.keys())[:a.max_casos]
    for modelo in a.modelos.split(","):
        for condicion in a.condiciones.split(","):
            for caso in lista_casos:
                if (caso, condicion, modelo) not in hechas:
                    tareas.append((caso, condicion, modelo, todas[caso]))
    print(f"pendientes: {len(tareas)} generaciones ({len(hechas)} ya hechas)")
    with ThreadPoolExecutor(max_workers=a.hilos) as ex, open(salida, "a", encoding="utf-8") as out:
        for fut in as_completed([ex.submit(procesar, t) for t in tareas]):
            r = fut.result()
            out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
            et = "OK " if r["ok"] else "ERR"
            extra = f"aluc={r['verificacion']['alucinaciones_detectadas']} rubr={r['rubrica']['secciones_presentes']}/6" if r["ok"] else r["error"][:60]
            print(f"[{et}] {r['caso']} {r['condicion']} {r['modelo']} {r['segundos']}s {extra}")
    print("listo")

if __name__ == "__main__":
    main()
