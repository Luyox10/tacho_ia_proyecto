"""
tacho_local.py
──────────────
Script standalone para el Tacho Inteligente físico.

Flujo:
  1. Descarga el modelo de Google Drive (con gdown, maneja la advertencia de archivos grandes).
  2. Carga el modelo Keras en memoria.
  3. Identifica al alumno por DNI consultando TiDB.
  4. Abre la cámara web con OpenCV.
  5. Clasifica el residuo en tiempo real y dibuja bounding box + etiqueta.
  6. Al superar el umbral de confianza, captura automáticamente y muestra el tacho asignado.
  7. Pregunta al usuario si tiene otro residuo y repite en bucle.
  8. Al finalizar, suma 10 puntos por residuo registrado en TiDB y muestra el ranking.

Dependencias:
  pip install opencv-python tensorflow gdown pymysql cryptography pillow numpy
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import gdown
import numpy as np
import pymysql
import pymysql.cursors
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_FILE_ID    = "1br48ms-wZiWBcbBR9LvP-QogoGz7_OOv"
RUTA_MODELO      = Path(__file__).parent / "modelo_ia.h5"
CAPTURAS_DIR     = Path(__file__).parent / "capturas"
IMG_SIZE         = (224, 224)
UMBRAL_CONFIANZA = 0.70          # 70 % mínimo para aceptar la clasificación
PUNTOS_POR_ITEM  = 10

# Categorías devueltas por el modelo (mismo orden del entrenamiento)
CATEGORIAS = ["glass", "organic", "metal", "others", "plastic", "paper"]

# Traducción amigable de la categoría → nombre + tacho de color
TACHO_INFO = {
    "glass":   {"nombre": "Vidrio",          "tacho": "Tacho Blanco  ⬜"},
    "organic": {"nombre": "Organico",        "tacho": "Tacho Marron  🟫"},
    "metal":   {"nombre": "Metal / Lata",    "tacho": "Tacho Amarillo 🟨"},
    "others":  {"nombre": "Otro residuo",    "tacho": "Tacho Negro   ⬛"},
    "plastic": {"nombre": "Plastico",        "tacho": "Tacho Azul    🟦"},
    "paper":   {"nombre": "Papel / Carton",  "tacho": "Tacho Azul    🟦"},
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCARGA SEGURA DEL MODELO (gdown supera la advertencia de Drive)
# ─────────────────────────────────────────────────────────────────────────────

def _verificar_modelo(ruta: Path) -> bool:
    """Devuelve True si el archivo existe y tiene un tamaño razonable (> 1 MB)."""
    return ruta.is_file() and ruta.stat().st_size > 1_000_000


def descargar_modelo(ruta: Path = RUTA_MODELO) -> None:
    """
    Descarga el modelo desde Google Drive con gdown.
    - Usa fuzzy=True para manejar automáticamente la cookie de confirmación.
    - Omite la descarga si el archivo ya existe y no está corrupto.
    """
    if _verificar_modelo(ruta):
        print(f"[Modelo] Archivo encontrado en '{ruta}'. Omitiendo descarga.")
        return

    print("[Modelo] Descargando modelo desde Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    try:
        gdown.download(url, str(ruta), quiet=False, fuzzy=True)
    except Exception as exc:
        raise RuntimeError(f"[Modelo] Error al descargar: {exc}") from exc

    if not _verificar_modelo(ruta):
        raise RuntimeError("[Modelo] El archivo descargado parece corrupto (tamaño insuficiente).")
    print("[Modelo] Descarga completada.")


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

def _dibujar_resultado(frame: np.ndarray, categoria: str, confianza: float) -> np.ndarray:
    """
    Dibuja un bounding box centrado, el nombre del residuo y la confianza sobre el frame.
    El color del recuadro cambia según el umbral (verde = aceptado, amarillo = procesando).
    """
    h, w = frame.shape[:2]
    # Bounding box fijo en el centro (20 % de margen)
    x1, y1 = int(w * 0.20), int(h * 0.20)
    x2, y2 = int(w * 0.80), int(h * 0.80)

    aceptado = confianza >= UMBRAL_CONFIANZA
    color    = (0, 220, 0) if aceptado else (0, 200, 255)  # verde / amarillo

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    info   = TACHO_INFO.get(categoria, {"nombre": categoria, "tacho": "?"})
    etiq   = f"{info['nombre']}  {confianza * 100:.1f}%"
    fuente = cv2.FONT_HERSHEY_SIMPLEX

    # Fondo del texto para legibilidad
    (tw, th), _ = cv2.getTextSize(etiq, fuente, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, etiq, (x1 + 4, y1 - 5), fuente, 0.7, (0, 0, 0), 2)

    if aceptado:
        tacho_txt = info["tacho"]
        cv2.putText(frame, tacho_txt, (x1, y2 + 28), fuente, 0.65, color, 2)

    return frame


def capturar_residuo(modelo) -> tuple[str, float] | None:
    """
    Abre la cámara y muestra el feed en tiempo real clasificando cada frame.
    - Si la confianza supera UMBRAL_CONFIANZA durante 1.5 s consecutivos → captura automática.
    - El usuario también puede presionar ESPACIO para capturar manualmente.
    - Presionar ESC cancela.

    Retorna (categoria, confianza) del frame capturado, o None si se canceló.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("[Camara] No se pudo abrir la cámara web.")

    print("\n[Camara] Mostrando video. Apunta al residuo.")
    print("         ESPACIO → capturar   |   ESC → cancelar\n")

    resultado      = None
    inicio_umbral  = None        # momento en que la confianza superó el umbral
    TIEMPO_UMBRAL  = 1.5         # segundos consecutivos sobre el umbral para auto-captura

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Camara] Error al leer frame.")
            break

        categoria, confianza = clasificar(modelo, frame)
        frame_visual = _dibujar_resultado(frame.copy(), categoria, confianza)

        # ── Lógica de auto-captura por tiempo ──
        if confianza >= UMBRAL_CONFIANZA:
            if inicio_umbral is None:
                inicio_umbral = time.time()
            transcurrido = time.time() - inicio_umbral
            restante     = max(0.0, TIEMPO_UMBRAL - transcurrido)
            barra        = int((transcurrido / TIEMPO_UMBRAL) * 20)
            cv2.putText(
                frame_visual,
                f"Auto-captura en {restante:.1f}s  [{'#' * barra}{'.' * (20 - barra)}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2,
            )
            if transcurrido >= TIEMPO_UMBRAL:
                resultado = (categoria, confianza)
                print(f"[Camara] Auto-captura: {categoria} ({confianza*100:.1f}%)")
                break
        else:
            inicio_umbral = None  # reinicia el contador si cae bajo el umbral

        cv2.imshow("Tacho Inteligente - ESC: cancelar | ESPACIO: capturar", frame_visual)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:          # ESC → cancelar
            print("[Camara] Captura cancelada por el usuario.")
            break
        if tecla == 32:          # ESPACIO → captura manual
            resultado = (categoria, confianza)
            print(f"[Camara] Captura manual: {categoria} ({confianza*100:.1f}%)")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Guarda el último frame como imagen si hubo resultado
    if resultado is not None:
        CAPTURAS_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_img = CAPTURAS_DIR / f"{ts}_{resultado[0]}.jpg"
        cv2.imwrite(str(ruta_img), frame)
        print(f"[Camara] Imagen guardada en '{ruta_img}'")

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
