"""
tacho_local.py
──────────────
Script standalone para el Tacho Inteligente físico.

Flujo:
  1. Descarga el modelo desde Hugging Face con requests (streaming, sin dependencias externas).
  2. Carga el modelo Keras en memoria.
  3. Identifica al alumno por DNI consultando TiDB.
  4. Abre la cámara web con OpenCV.
  5. Clasifica el residuo en tiempo real y dibuja bounding box + etiqueta.
  6. Al superar el umbral de confianza, captura automáticamente y muestra el tacho asignado.
  7. Pregunta al usuario si tiene otro residuo y repite en bucle.
  8. Al finalizar, suma 10 puntos por residuo registrado en TiDB y muestra el ranking.

Dependencias:
  pip install opencv-python tensorflow requests pymysql cryptography pillow numpy
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import cv2
import numpy as np
import pymysql
import pymysql.cursors
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

HF_MODEL_URL     = "https://huggingface.co/JuliEt10/tacho-ia-modelo/resolve/main/modelo_residuos.h5?download=true"
RUTA_MODELO      = Path(__file__).parent / "modelo_residuos.h5"
CAPTURAS_DIR     = Path(__file__).parent / "capturas"
IMG_SIZE         = (224, 224)
UMBRAL_CONFIANZA = 0.70          # 70 % mínimo para aceptar la clasificación
PUNTOS_POR_ITEM  = 10

# Categorías devueltas por el modelo (mismo orden del entrenamiento)
CATEGORIAS = ["glass", "organic", "metal", "others", "plastic", "paper"]

# Traducción amigable: categoría → nombre, tacho, color BGR del recuadro
TACHO_INFO = {
    "glass":   {"nombre": "Vidrio",          "tacho": "Tacho BLANCO",   "color": (220, 220, 220)},
    "organic": {"nombre": "Organico",        "tacho": "Tacho MARRON",   "color": (42,  87,  130)},
    "metal":   {"nombre": "Metal / Lata",    "tacho": "Tacho AMARILLO", "color": (0,  210, 230)},
    "others":  {"nombre": "Otro residuo",    "tacho": "Tacho NEGRO",    "color": (60,   60,  60)},
    "plastic": {"nombre": "Plastico",        "tacho": "Tacho AZUL",     "color": (210,  80,  20)},
    "paper":   {"nombre": "Papel / Carton",  "tacho": "Tacho AZUL",     "color": (210,  80,  20)},
}

# Cuántos frames saltar entre cada inferencia (para mantener fluidez)
INFERENCIA_CADA_N_FRAMES = 5

# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCARGA LIMPIA DEL MODELO (Hugging Face vía requests)
# ─────────────────────────────────────────────────────────────────────────────

def _verificar_modelo(ruta: Path) -> bool:
    """Devuelve True si el archivo existe y tiene un tamaño razonable (> 1 MB)."""
    return ruta.is_file() and ruta.stat().st_size > 1_000_000


def descargar_modelo(ruta: Path = RUTA_MODELO) -> None:
    """
    Descarga el modelo desde Hugging Face usando requests en modo streaming.
    - Omite la descarga si el archivo ya existe y no está corrupto.
    - Muestra el tamaño en MB del archivo para validar que no esté vacío.
    - Escribe en bloques de 8 KB para no saturar la memoria.
    """
    if _verificar_modelo(ruta):
        size_mb = ruta.stat().st_size / 1_048_576
        print(f"[Modelo] Archivo '{ruta.name}' ya existe ({size_mb:.2f} MB). Omitiendo descarga.")
        return

    print(f"[Modelo] Descargando modelo desde Hugging Face...")
    print(f"         URL: {HF_MODEL_URL}")
    ruta.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(HF_MODEL_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            descargado = 0
            with open(ruta, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        descargado += len(chunk)
                        if total:
                            pct = descargado / total * 100
                            print(f"\r         Progreso: {pct:.1f}%  ({descargado/1_048_576:.1f} MB)", end='', flush=True)
        print()  # salto de línea tras el progreso
    except requests.RequestException as exc:
        if ruta.exists():
            ruta.unlink()  # eliminar archivo incompleto
        raise RuntimeError(f"[Modelo] Error al descargar: {exc}") from exc

    if not _verificar_modelo(ruta):
        ruta.unlink()
        raise RuntimeError("[Modelo] El archivo descargado parece corrupto (tamaño insuficiente).")

    size_mb = ruta.stat().st_size / 1_048_576
    print(f"[Modelo] Descarga completada. Tamaño: {size_mb:.2f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGA DEL MODELO KERAS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo(ruta: Path = RUTA_MODELO):
    """
    Carga el modelo Keras desde disco.
    Retarda el import de TensorFlow hasta que sea necesario para un arranque más rápido.
    """
    import tensorflow as tf  # import lazy intencional
    print("[Modelo] Cargando modelo Keras en memoria...")
    modelo = tf.keras.models.load_model(str(ruta))
    print("[Modelo] Modelo listo.")
    return modelo


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO DE IMAGEN
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte un frame BGR de OpenCV al tensor que espera el modelo:
      - Redimensiona a IMG_SIZE
      - Normaliza a [0, 1]
      - Añade dimensión de batch
    """
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img   = Image.fromarray(rgb).resize(IMG_SIZE)
    arr   = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def clasificar(modelo, frame_bgr: np.ndarray) -> tuple[str, float]:
    """
    Ejecuta el modelo sobre un frame y retorna (categoria, confianza).
    """
    tensor     = preprocesar_frame(frame_bgr)
    prediccion = modelo.predict(tensor, verbose=0)[0]
    idx        = int(np.argmax(prediccion))
    return CATEGORIAS[idx], float(prediccion[idx])


# ─────────────────────────────────────────────────────────────────────────────
# 3. BASE DE DATOS — conexión TiDB (pymysql)
# ─────────────────────────────────────────────────────────────────────────────

def _obtener_conexion() -> pymysql.connections.Connection:
    """
    Crea una conexión a TiDB leyendo las variables de entorno.
    Las mismas variables que usa el backend FastAPI.
    """
    host     = os.environ.get("DB_HOST")
    user     = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    database = os.environ.get("DB_NAME")
    port     = int(os.environ.get("DB_PORT", 4000))

    faltantes = [k for k, v in {"DB_HOST": host, "DB_USER": user,
                                 "DB_PASSWORD": password, "DB_NAME": database}.items() if not v]
    if faltantes:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(faltantes)}")

    return pymysql.connect(
        host=host, user=user, password=password,
        database=database, port=port,
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ssl": {}},
    )


def _ejecutar(sql: str, params: tuple = (), *, fetchone=False, fetchall=False, commit=False):
    """Ejecuta una consulta parametrizada de forma segura (sin inyección SQL)."""
    conn = None
    try:
        conn = _obtener_conexion()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if commit:
                conn.commit()
                return cur.lastrowid
            if fetchone:
                return cur.fetchone()
            if fetchall:
                return cur.fetchall()
    except pymysql.MySQLError as exc:
        if conn:
            conn.rollback()
        raise RuntimeError(f"[DB] Error en consulta: {exc}") from exc
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# OPERACIONES DE NEGOCIO EN BASE DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def identificar_alumno_por_dni(dni: str) -> dict | None:
    """Busca al alumno en la tabla 'usuarios' por DNI. Retorna dict o None."""
    return _ejecutar(
        "SELECT id, nombre, apellido, aula_id FROM usuarios WHERE dni = %s AND rol = 'alumno'",
        (dni,),
        fetchone=True,
    )


def registrar_residuo(usuario_id: int, tipo_residuo: str, puntos: int = PUNTOS_POR_ITEM) -> None:
    """Inserta una fila en 'registro_residuos' y actualiza los puntos del aula."""
    # Insertar en historial
    _ejecutar(
        "INSERT INTO registro_residuos (usuario_id, tipo_residuo, puntos_ganados, fecha_registro) "
        "VALUES (%s, %s, %s, %s)",
        (usuario_id, tipo_residuo, puntos, datetime.utcnow()),
        commit=True,
    )
    # Obtener aula_id del alumno
    alumno = _ejecutar(
        "SELECT aula_id FROM usuarios WHERE id = %s", (usuario_id,), fetchone=True
    )
    if alumno and alumno.get("aula_id"):
        _ejecutar(
            "UPDATE aulas SET puntos_totales = puntos_totales + %s WHERE id = %s",
            (puntos, alumno["aula_id"]),
            commit=True,
        )


def obtener_ranking_aulas(limite: int = 5) -> list[dict]:
    """Retorna el top N de aulas ordenadas por puntos_totales."""
    return _ejecutar(
        "SELECT grado_seccion, puntos_totales FROM aulas ORDER BY puntos_totales DESC LIMIT %s",
        (limite,),
        fetchall=True,
    ) or []


# ─────────────────────────────────────────────────────────────────────────────
# 4. CÁMARA — captura y detección en tiempo real
# ─────────────────────────────────────────────────────────────────────────────

def _texto_con_fondo(
    frame: np.ndarray,
    texto: str,
    origen: tuple[int, int],
    escala: float,
    color_texto: tuple,
    color_fondo: tuple,
    grosor: int = 2,
    padding: int = 6,
) -> None:
    """Escribe texto con un rectángulo de fondo para máxima legibilidad."""
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(texto, fuente, escala, grosor)
    x, y = origen
    cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + baseline + padding // 2), color_fondo, -1)
    cv2.putText(frame, texto, (x, y), fuente, escala, color_texto, grosor, cv2.LINE_AA)


def _dibujar_resultado(frame: np.ndarray, categoria: str, confianza: float) -> np.ndarray:
    """
    Dibuja sobre el frame:
      - Bounding box centrado con el color del tacho asignado.
      - Overlay semitransparente dentro del recuadro.
      - Etiqueta superior: nombre del residuo + confianza %.
      - Etiqueta inferior: nombre del tacho de reciclaje.
      - Barra de progreso de auto-captura (solo cuando supera el umbral).
    """
    h, w = frame.shape[:2]
    x1, y1 = int(w * 0.15), int(h * 0.15)
    x2, y2 = int(w * 0.85), int(h * 0.85)

    aceptado = confianza >= UMBRAL_CONFIANZA
    info     = TACHO_INFO.get(categoria, {"nombre": categoria, "tacho": "?", "color": (100, 100, 100)})
    color    = info["color"] if aceptado else (0, 200, 255)   # color del tacho o amarillo

    # ── Overlay semitransparente dentro del recuadro ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    # ── Bounding box ──
    grosor_box = 3 if aceptado else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor_box)
    # Esquinas decorativas
    largo = 20
    for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * largo, py), color, 4)
        cv2.line(frame, (px, py), (px, py + dy * largo), color, 4)

    # ── Etiqueta superior: nombre + confianza ──
    etiq_nombre = f"{info['nombre']}  {confianza * 100:.1f}%"
    _texto_con_fondo(frame, etiq_nombre, (x1, y1 - 8), 0.75, (255, 255, 255), color)

    # ── Etiqueta inferior: tacho asignado (solo si supera umbral) ──
    if aceptado:
        etiq_tacho = f"  {info['tacho']}  "
        _texto_con_fondo(frame, etiq_tacho, (x1, y2 + 32), 0.80, (255, 255, 255), color, grosor=2)

    # ── Instrucciones en esquina inferior izquierda ──
    cv2.putText(frame, "ESPACIO: capturar  |  ESC: cancelar",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def _dibujar_barra_autocaptura(frame: np.ndarray, transcurrido: float, tiempo_total: float) -> None:
    """Dibuja una barra de progreso de auto-captura en la parte superior del frame."""
    h, w = frame.shape[:2]
    pct     = min(transcurrido / tiempo_total, 1.0)
    restante = max(0.0, tiempo_total - transcurrido)
    barra_w  = int(w * pct)

    # Fondo gris
    cv2.rectangle(frame, (0, 0), (w, 22), (40, 40, 40), -1)
    # Barra verde
    cv2.rectangle(frame, (0, 0), (barra_w, 22), (0, 210, 60), -1)
    # Texto
    txt = f"Capturando en {restante:.1f}s..."
    cv2.putText(frame, txt, (w // 2 - 90, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _pantalla_resultado(frame: np.ndarray, categoria: str, confianza: float) -> np.ndarray:
    """Genera un frame final con el resultado resaltado para mostrarlo 2 s tras la captura."""
    panel = frame.copy()
    h, w  = panel.shape[:2]
    info  = TACHO_INFO.get(categoria, {"nombre": categoria, "tacho": "?", "color": (100, 100, 100)})
    color = info["color"]

    # Overlay oscuro
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, panel, 0.55, 0, panel)

    # Texto central grande
    cy = h // 2
    _texto_con_fondo(panel, "RESIDUO DETECTADO",  (w // 2 - 130, cy - 60), 0.9,  (255, 255, 255), color, grosor=2)
    _texto_con_fondo(panel, info["nombre"],        (w // 2 - 130, cy - 10), 1.1,  (255, 255, 255), (30, 30, 30), grosor=3)
    _texto_con_fondo(panel, f"{confianza*100:.1f}% confianza", (w // 2 - 80, cy + 40), 0.75, (200, 255, 200), (30, 30, 30))
    _texto_con_fondo(panel, info["tacho"],         (w // 2 - 100, cy + 85), 0.95, (255, 255, 255), color, grosor=2)

    return panel


def capturar_residuo(modelo) -> tuple[str, float] | None:
    """
    Abre la cámara y muestra el feed en tiempo real clasificando residuos.

    Comportamiento:
      - Clasifica cada INFERENCIA_CADA_N_FRAMES frames para mantener fluidez.
      - Dibuja bounding box con el color del tacho asignado.
      - Al superar UMBRAL_CONFIANZA durante TIEMPO_UMBRAL segundos → auto-captura.
      - ESPACIO → captura manual inmediata.
      - ESC → cancelar.
      - Tras captura: muestra pantalla de resultado 2 s y guarda la imagen.

    Retorna (categoria, confianza) o None si se canceló.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("[Camara] No se pudo abrir la cámara web.")

    # Resolución recomendada para fluidez
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[Camara] Mostrando video. Apunta el residuo al recuadro.")
    print("         ESPACIO → capturar manual   |   ESC → cancelar\n")

    TITULO      = "Tacho Inteligente — Eco-School High"
    TIEMPO_UMBRAL = 1.5

    resultado      = None
    frame_actual   = None          # último frame leído (para guardar imagen)
    categoria      = "others"
    confianza      = 0.0
    inicio_umbral  = None
    n_frame        = 0             # contador de frames para sub-muestreo de inferencia

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Camara] Error al leer frame.")
            break
        frame_actual = frame
        n_frame += 1

        # ── Inferencia cada N frames ──
        if n_frame % INFERENCIA_CADA_N_FRAMES == 0:
            categoria, confianza = clasificar(modelo, frame)

        # ── Dibujar visualización ──
        frame_visual = _dibujar_resultado(frame.copy(), categoria, confianza)

        # ── Lógica de auto-captura ──
        if confianza >= UMBRAL_CONFIANZA:
            if inicio_umbral is None:
                inicio_umbral = time.time()
            transcurrido = time.time() - inicio_umbral
            _dibujar_barra_autocaptura(frame_visual, transcurrido, TIEMPO_UMBRAL)
            if transcurrido >= TIEMPO_UMBRAL:
                resultado = (categoria, confianza)
                print(f"\n[Camara] ✅ Auto-captura: {TACHO_INFO.get(categoria, {}).get('nombre', categoria)} ({confianza*100:.1f}%)")
                break
        else:
            inicio_umbral = None

        cv2.imshow(TITULO, frame_visual)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:   # ESC
            print("[Camara] Captura cancelada.")
            break
        if tecla == 32:   # ESPACIO
            resultado = (categoria, confianza)
            print(f"\n[Camara] ✅ Captura manual: {TACHO_INFO.get(categoria, {}).get('nombre', categoria)} ({confianza*100:.1f}%)")
            break

    # ── Mostrar pantalla de resultado 2 s ──
    if resultado is not None and frame_actual is not None:
        panel = _pantalla_resultado(frame_actual.copy(), resultado[0], resultado[1])
        cv2.imshow(TITULO, panel)
        cv2.waitKey(2000)

    cap.release()
    cv2.destroyAllWindows()

    # ── Guardar imagen capturada ──
    if resultado is not None and frame_actual is not None:
        CAPTURAS_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_img = CAPTURAS_DIR / f"{ts}_{resultado[0]}.jpg"
        cv2.imwrite(str(ruta_img), frame_actual)
        print(f"[Camara] Imagen guardada → '{ruta_img}'")

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# 5. FLUJO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def preguntar_continuar() -> bool:
    """Pregunta al usuario si desea registrar otro residuo. Retorna True/False."""
    while True:
        resp = input("\n¿Tienes otro residuo para registrar? (s/n): ").strip().lower()
        if resp in ("s", "si", "sí", "y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("  Por favor ingresa 's' o 'n'.")


def mostrar_ranking(limite: int = 5) -> None:
    """Imprime el ranking de aulas en consola."""
    print("\n" + "─" * 40)
    print("   🏆  RANKING DE AULAS  🏆")
    print("─" * 40)
    try:
        ranking = obtener_ranking_aulas(limite)
        for i, fila in enumerate(ranking, 1):
            medalla = ["🥇", "🥈", "🥉"].pop(0) if i <= 3 else f" {i}."
            print(f"  {medalla}  {fila['grado_seccion']:<20}  {fila['puntos_totales']:>6} pts")
    except RuntimeError as exc:
        print(f"  (No se pudo cargar el ranking: {exc})")
    print("─" * 40 + "\n")


def main() -> None:
    # ── Paso 1: Descargar modelo si es necesario ──
    try:
        descargar_modelo()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # ── Paso 2: Cargar modelo en memoria ──
    try:
        modelo = cargar_modelo()
    except Exception as exc:
        print(f"[ERROR] No se pudo cargar el modelo: {exc}")
        sys.exit(1)

    # ── Paso 3: Identificar alumno por DNI ──
    print("\n" + "=" * 50)
    print("   TACHO INTELIGENTE — ECO-SCHOOL HIGH")
    print("=" * 50)

    alumno = None
    while alumno is None:
        dni = input("\nIngresa tu DNI para comenzar: ").strip()
        if not dni:
            continue
        try:
            alumno = identificar_alumno_por_dni(dni)
        except RuntimeError as exc:
            print(f"[DB] Error al buscar DNI: {exc}")
            print("     Continuando en modo sin base de datos...")
            alumno = {"id": None, "nombre": "Invitado", "apellido": "", "aula_id": None}
            break

        if alumno is None:
            print("  ❌ DNI no encontrado. Intenta de nuevo.")

    print(f"\n  ✅ Bienvenido/a, {alumno['nombre']} {alumno.get('apellido', '')}!\n")

    # ── Pasos 4-6: Bucle de captura y registro ──
    residuos_sesion = []   # lista de (categoria, confianza) registrados en esta sesión

    while True:
        print("\n[INFO] Preparando cámara para el siguiente residuo...")
        try:
            resultado = capturar_residuo(modelo)
        except RuntimeError as exc:
            print(f"[ERROR Cámara] {exc}")
            break

        if resultado is None:
            print("[INFO] Captura cancelada. No se registra este residuo.")
        else:
            categoria, confianza = resultado
            info = TACHO_INFO.get(categoria, {"nombre": categoria, "tacho": "?"})

            print(f"\n  Residuo detectado : {info['nombre']}")
            print(f"  Confianza         : {confianza * 100:.1f}%")
            print(f"  Deposita en       : {info['tacho']}")

            # Registrar en base de datos
            if alumno.get("id") is not None:
                try:
                    registrar_residuo(alumno["id"], categoria, PUNTOS_POR_ITEM)
                    print(f"  ✅ +{PUNTOS_POR_ITEM} puntos registrados en la base de datos.")
                except RuntimeError as exc:
                    print(f"  ⚠️  No se pudo guardar en BD: {exc}")
            else:
                print(f"  ℹ️  Modo sin BD — se registra localmente.")

            residuos_sesion.append(resultado)

        # Preguntar si continúa
        if not preguntar_continuar():
            break

    # ── Paso 7: Resumen de sesión ──
    total       = len(residuos_sesion)
    puntos_sess = total * PUNTOS_POR_ITEM

    print("\n" + "=" * 50)
    print("   RESUMEN DE SESIÓN")
    print("=" * 50)
    print(f"  Residuos registrados : {total}")
    print(f"  Puntos obtenidos     : {puntos_sess}")

    if residuos_sesion:
        print("\n  Detalle:")
        for i, (cat, conf) in enumerate(residuos_sesion, 1):
            info = TACHO_INFO.get(cat, {"nombre": cat})
            print(f"    {i}. {info['nombre']:<18}  {conf * 100:.1f}%  →  +{PUNTOS_POR_ITEM} pts")

    # ── Paso 8: Mostrar ranking actualizado ──
    mostrar_ranking()

    print("¡Gracias por reciclar! 🌿\n")


if __name__ == "__main__":
    main()
