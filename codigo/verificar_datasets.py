"""M1 - Verificacion de datasets (v2).
Uso: python verificar_datasets.py [--solo elliptic|ibm|todo]
- Elliptic se verifica directamente desde el .zip (copia canonica, no requiere extraer 1 GB).
- IBM AML se verifica con una sola pasada binaria (rapida) sobre HI-Small_Trans.csv.
Genera datasets_reporte.txt (entregable de cierre del Modulo 1)."""
import argparse
import csv
import io
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
CARPETA_REPORTES = Path(__file__).resolve().parent

def contar_lineas_stream(f):
    n, ultimo = 0, b""
    while True:
        b = f.read(8 * 1024 * 1024)
        if not b:
            break
        n += b.count(b"\n")
        ultimo = b
    if ultimo and not ultimo.endswith(b"\n"):
        n += 1
    return n

def check(L, nombre, valor, esperado=None):
    marca = ""
    if esperado is not None:
        marca = "  [OK]" if abs(valor - esperado) <= 1 else f"  [AVISO: se esperaban ~{esperado:,}]"
    L.append(f"  {nombre}: {valor:,}{marca}")

def verificar_elliptic():
    L = ["--- Elliptic (verificado desde el zip canonico) ---"]
    ell = DATOS / "elliptic"
    zips = sorted(ell.glob("*.zip")) if ell.exists() else []
    if not zips:
        L.append("  *** No se encontro .zip en datos/elliptic ***")
        return L
    z = zipfile.ZipFile(zips[0])
    L.append(f"  Archivo: {zips[0].name}")
    miembros = {Path(m).name: m for m in z.namelist() if m.endswith(".csv")}
    esperados = {"elliptic_txs_features.csv": 203769,
                 "elliptic_txs_classes.csv": 203770,
                 "elliptic_txs_edgelist.csv": 234356}
    for nombre, filas_esp in esperados.items():
        if nombre in miembros:
            with z.open(miembros[nombre]) as f:
                check(L, f"{nombre} lineas", contar_lineas_stream(f), filas_esp)
        else:
            L.append(f"  *** FALTA {nombre} dentro del zip ***")
    if "elliptic_txs_classes.csv" in miembros:
        with z.open(miembros["elliptic_txs_classes.csv"]) as f:
            cuentas = {}
            for fila in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                cuentas[fila["class"]] = cuentas.get(fila["class"], 0) + 1
        ilicito, licito = cuentas.get("1", 0), cuentas.get("2", 0)
        L.append(f"  Ilicitas: {ilicito:,} | Licitas: {licito:,} | Desconocidas: {cuentas.get('unknown', 0):,}")
        if ilicito + licito:
            L.append(f"  Tasa ilicita entre etiquetadas: {ilicito / (ilicito + licito):.2%} (referencia ~9.8%)")
    return L

def verificar_ibm():
    L = ["--- IBM AML (HI-Small) ---"]
    aml = DATOS / "ibm_aml"
    trans = sorted(aml.glob("*Trans.csv")) if aml.exists() else []
    if not trans:
        L.append("  *** No se encontro *Trans.csv en datos/ibm_aml ***")
    for t in trans:
        with open(t, "rb") as f:
            encabezado = f.readline()
            if b"Is Laundering" not in encabezado:
                L.append(f"  [AVISO] Encabezado inesperado en {t.name}: {encabezado[:80]!r}")
            filas = lav1 = lav0 = 0
            resto = b""
            while True:
                b = f.read(16 * 1024 * 1024)
                if not b:
                    break
                buf = resto + b
                corte = buf.rfind(b"\n")
                if corte < 0:
                    resto = buf
                    continue
                proc, resto = buf[:corte + 1], buf[corte + 1:]
                filas += proc.count(b"\n")
                lav1 += proc.count(b",1\n") + proc.count(b",1\r\n")
                lav0 += proc.count(b",0\n") + proc.count(b",0\r\n")
            if resto:
                filas += 1
                lav1 += resto.endswith(b",1")
                lav0 += resto.endswith(b",0")
        check(L, f"{t.name} transacciones", filas)
        L.append(f"  Lavado=1: {lav1:,} ({lav1 / filas:.4%}) | Lavado=0: {lav0:,} - desbalanceo extremo esperado")
        if filas and (filas - lav0 - lav1) / filas > 0.005:
            L.append(f"  [AVISO] {filas - lav0 - lav1:,} filas sin flag 0/1 reconocible al final de linea")
    patrones = sorted(aml.glob("*Patterns.txt")) if aml.exists() else []
    if not patrones:
        L.append("  *** No se encontro *Patterns.txt - CRITICO: es el ground truth de explicacion (C2) ***")
    for p in patrones:
        texto = p.read_text(encoding="utf-8", errors="ignore").upper()
        L.append(f"  {p.name}: {texto.count('BEGIN LAUNDERING ATTEMPT'):,} intentos de lavado etiquetados")
        for tipo in ["FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "BIPARTITE", "STACK", "RANDOM"]:
            n = texto.count(f"BEGIN LAUNDERING ATTEMPT - {tipo}")
            if n:
                L.append(f"    {tipo}: {n:,}")
    return L

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=["elliptic", "ibm", "todo"], default="todo")
    modo = ap.parse_args().solo
    if modo in ("elliptic", "todo"):
        (CARPETA_REPORTES / "_seccion_elliptic.txt").write_text("\n".join(verificar_elliptic()), encoding="utf-8")
    if modo in ("ibm", "todo"):
        (CARPETA_REPORTES / "_seccion_ibm.txt").write_text("\n".join(verificar_ibm()), encoding="utf-8")
    secciones = [CARPETA_REPORTES / "_seccion_elliptic.txt", CARPETA_REPORTES / "_seccion_ibm.txt"]
    if all(s.exists() for s in secciones):
        cuerpo = "\n\n".join(s.read_text(encoding="utf-8") for s in secciones)
        reporte = ("=== REPORTE DE DATASETS - M1 ===\n"
                   f"Carpeta de datos: {DATOS}\n\n{cuerpo}\n\n"
                   "Nota: conteos de referencia pueden variar levemente segun la version de Kaggle; un [AVISO] se reporta, no invalida.\n")
        (CARPETA_REPORTES / "datasets_reporte.txt").write_text(reporte, encoding="utf-8")
        for s in secciones:
            s.unlink()
        print(reporte)
    else:
        print(f"Seccion '{modo}' verificada. Ejecuta la seccion restante para consolidar el reporte.")
