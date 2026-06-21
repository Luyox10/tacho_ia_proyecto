"""
main.py
───────
Backend FastAPI para el proyecto de reciclaje escolar con IA.

Endpoints:
  • POST /api/login                         – Login unificado (DNI o email)
  • GET  /api/tacho/identificar/{dni}       – Identificación rápida para simulador físico
  • POST /api/simulador/clasificar          – Clasificación de imagen con modelo Keras
  • GET  /api/dashboard/alumno/{alumno_id}  – Panel del alumno (puntos + logros)
  • GET  /api/dashboard/docente/{docente_id}– Panel del docente (alumnos + ranking)
  • GET  /api/dashboard/director            – Panel del director (métricas globales)
"""

import base64
import io
import os
import requests
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from base_datos import ejecutar_consulta

# ─────────────────────────────────────────────
# Configuración de la app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Tacho-IA Backend",
    description="API del proyecto de reciclaje escolar con clasificación por IA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Carga del modelo Keras (una sola vez al iniciar)
# ─────────────────────────────────────────────

CATEGORIAS = ["glass", "organic", "metal", "others", "plastic", "paper"]
IMG_SIZE = (224, 224)

_modelo = None


DRIVE_MODEL_URL = "https://docs.google.com/uc?export=download&id=1br48ms-wZiWBcbBR9LvP-QogoGz7_OOv"


def _descargar_modelo(ruta: str) -> None:
    """
    Descarga el modelo desde Google Drive en bloques de 8192 bytes.

    Args:
        ruta: ruta local donde guardar el archivo .h5.
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    print("Descargando modelo de IA desde Google Drive...")
    with requests.get(DRIVE_MODEL_URL, stream=True) as resp:
        resp.raise_for_status()
        with open(ruta, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Descarga del modelo completada.")


def cargar_modelo():
    """
    Carga el modelo Keras desde disco de forma lazy (singleton).
    Se invoca en la primera petición de clasificación para no bloquear
    el arranque del servidor si el archivo no existe todavía.

    Returns:
        tensorflow.keras.Model: modelo compilado listo para predicción.

    Raises:
        HTTPException 503: si el archivo del modelo no se encuentra.
    """
    global _modelo
    if _modelo is None:
        ruta = os.path.join(os.path.dirname(__file__), "modelo_ia.h5")
        if not os.path.isfile(ruta):
            _descargar_modelo(ruta)
        import tensorflow as tf
        _modelo = tf.keras.models.load_model(ruta)
    return _modelo


# ─────────────────────────────────────────────
# Utilidades de imagen
# ─────────────────────────────────────────────

def preprocesar_imagen(imagen_bytes: bytes) -> np.ndarray:
    """
    Convierte bytes de imagen en un array NumPy normalizado listo para el modelo.

    Args:
        imagen_bytes: contenido binario de la imagen (JPEG/PNG).

    Returns:
        np.ndarray de shape (1, 224, 224, 3) con valores en [0, 1].

    Raises:
        HTTPException 400: si la imagen no se puede procesar.
    """
    try:
        img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo procesar la imagen: {e}",
        )


# ═════════════════════════════════════════════
#  ENDPOINTS
# ═════════════════════════════════════════════


# ── 1. LOGIN UNIFICADO ───────────────────────

@app.post("/api/login")
async def login(dni: Optional[str] = Form(None), email: Optional[str] = Form(None),
                contrasena: str = Form(...)):
    """
    Login unificado para la plataforma web.

    - Si se envía **dni**: busca en `usuarios` con rol alumno o docente.
    - Si se envía **email**: busca en `usuarios` con rol administrador (director).

    Form fields:
        dni (opcional), email (opcional), contrasena (obligatorio).

    Returns:
        dict con id, nombre, rol y aula_id del usuario autenticado.

    Raises:
        HTTPException 400: si no se envía ni DNI ni email.
        HTTPException 401: credenciales inválidas.
        HTTPException 500: error interno de base de datos.
    """
    if not dni and not email:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar 'dni' o 'email' para iniciar sesión.",
        )

    try:
        if dni:
            usuario = ejecutar_consulta(
                "SELECT id, nombre, rol, aula_id, contrasena "
                "FROM usuarios WHERE dni = %s AND rol IN ('alumno', 'docente')",
                (dni,),
                fetchone=True,
            )
        else:
            usuario = ejecutar_consulta(
                "SELECT id, nombre, rol, aula_id, contrasena "
                "FROM usuarios WHERE email = %s AND rol = 'administrador'",
                (email,),
                fetchone=True,
            )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not usuario:
        raise HTTPException(status_code=401, detail="Usuario no encontrado.")

    if usuario["contrasena"] != contrasena:
        raise HTTPException(status_code=401, detail="Contraseña incorrecta.")

    return {
        "id": usuario["id"],
        "nombre": usuario["nombre"],
        "rol": usuario["rol"],
        "aula_id": usuario["aula_id"],
    }


@app.get("/probar-db")
def probar_db():
    try:
        # Esto le pide a TiDB que nos muestre cómo es la tabla por dentro
        columnas = ejecutar_consulta("DESCRIBE registro_residuos;")
        return {"status": "Estructura de la tabla", "columnas": columnas}
    except Exception as e:
        return {"status": "Error", "error": str(e)}

# ── 2. IDENTIFICACIÓN RÁPIDA (TACHO FÍSICO) ──

@app.get("/api/tacho/identificar/{dni}")
async def identificar_alumno(dni: str):
    """
    Identificación rápida para el simulador de patio.
    No requiere contraseña; solo devuelve datos públicos mínimos.

    Path params:
        dni: DNI del alumno.

    Returns:
        dict con alumno_id, nombre y aula_id.

    Raises:
        HTTPException 404: alumno no encontrado.
        HTTPException 500: error de base de datos.
    """
    try:
        alumno = ejecutar_consulta(
            "SELECT id, nombre, aula_id FROM usuarios WHERE dni = %s AND rol = 'alumno'",
            (dni,),
            fetchone=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not alumno:
        raise HTTPException(status_code=404, detail="Alumno no encontrado con ese DNI.")

    return {
        "alumno_id": alumno["id"],
        "nombre": alumno["nombre"],
        "aula_id": alumno["aula_id"],
    }


# ── 3. CLASIFICACIÓN CON IA ──────────────────

@app.post("/api/simulador/clasificar")
async def clasificar_residuo(
    alumno_id: int = Form(...),
    imagen_base64: Optional[str] = Form(None),
    imagen_archivo: Optional[UploadFile] = File(None),
):
    """
    Recibe una imagen (base64 o multipart), la clasifica con el modelo Keras
    y registra el resultado en la base de datos.

    Form fields:
        alumno_id:      ID del alumno que deposita el residuo.
        imagen_base64:  (opcional) imagen codificada en base64.
        imagen_archivo: (opcional) archivo de imagen subido.

    Flujo:
        1. Preprocesa la imagen → array (1, 224, 224, 3).
        2. Ejecuta predicción con modelo_ia.h5.
        3. Inserta registro en `registro_residuos`.
        4. Suma +10 puntos al aula del alumno en `aulas`.

    Returns:
        dict con categoria_detectada, confianza y puntos_sumados.

    Raises:
        HTTPException 400: si no se envía imagen.
        HTTPException 404: si el alumno no existe.
        HTTPException 500: error interno.
    """
    # ── Obtener bytes de la imagen ──
    if imagen_base64:
        try:
            imagen_bytes = base64.b64decode(imagen_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="El base64 de la imagen es inválido.")
    elif imagen_archivo:
        imagen_bytes = await imagen_archivo.read()
    else:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar 'imagen_base64' o 'imagen_archivo'.",
        )

    # ── Verificar alumno ──
    try:
        alumno = ejecutar_consulta(
            "SELECT id, aula_id FROM usuarios WHERE id = %s AND rol = 'alumno'",
            (alumno_id,),
            fetchone=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not alumno:
        raise HTTPException(status_code=404, detail="Alumno no encontrado.")

    # ── Clasificar ──
    modelo = cargar_modelo()
    tensor = preprocesar_imagen(imagen_bytes)
    prediccion = modelo.predict(tensor, verbose=0)
    indice = int(np.argmax(prediccion[0]))
    categoria = CATEGORIAS[indice]
    confianza = float(prediccion[0][indice])

    # ── Registrar en BD ──
    puntos = 10
    try:
        ejecutar_consulta(
            "INSERT INTO registro_residuos (alumno_id, aula_id, categoria, confianza, puntos, fecha) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (alumno_id, alumno["aula_id"], categoria, round(confianza, 4), puntos, datetime.utcnow()),
            commit=True,
        )
        ejecutar_consulta(
            "UPDATE aulas SET puntos_totales = puntos_totales + %s WHERE id = %s",
            (puntos, alumno["aula_id"]),
            commit=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "categoria_detectada": categoria,
        "confianza": round(confianza, 4),
        "puntos_sumados": puntos,
    }


# ── 4. DASHBOARD ALUMNO ──────────────────────

@app.get("/api/dashboard/alumno/{alumno_id}")
async def dashboard_alumno(alumno_id: int):
    """
    Datos para el panel del alumno: puntos acumulados, cantidad de registros
    y logros obtenidos.

    Path params:
        alumno_id: ID del alumno.

    Returns:
        dict con nombre, puntos_totales, total_registros y lista de logros.

    Raises:
        HTTPException 404: alumno no encontrado.
        HTTPException 500: error de base de datos.
    """
    try:
        alumno = ejecutar_consulta(
            "SELECT id, nombre, aula_id FROM usuarios WHERE id = %s AND rol = 'alumno'",
            (alumno_id,),
            fetchone=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not alumno:
        raise HTTPException(status_code=404, detail="Alumno no encontrado.")

    try:
        resumen = ejecutar_consulta(
            "SELECT COALESCE(SUM(puntos), 0) AS puntos_totales, "
            "       COUNT(*) AS total_registros "
            "FROM registro_residuos WHERE alumno_id = %s",
            (alumno_id,),
            fetchone=True,
        )
        logros = ejecutar_consulta(
            "SELECT id, titulo, descripcion, fecha_obtencion "
            "FROM logros WHERE alumno_id = %s ORDER BY fecha_obtencion DESC",
            (alumno_id,),
            fetchall=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "alumno_id": alumno["id"],
        "nombre": alumno["nombre"],
        "aula_id": alumno["aula_id"],
        "puntos_totales": resumen["puntos_totales"] if resumen else 0,
        "total_registros": resumen["total_registros"] if resumen else 0,
        "logros": logros or [],
    }


# ── 5. DASHBOARD DOCENTE ─────────────────────

@app.get("/api/dashboard/docente/{docente_id}")
async def dashboard_docente(docente_id: int):
    """
    Datos para el panel del docente: lista de hasta 25 alumnos de su aula
    con sus puntos individuales, y el ranking del aula a nivel escuela.

    Path params:
        docente_id: ID del docente.

    Returns:
        dict con aula, lista de alumnos (top 25) y posición del aula en ranking.

    Raises:
        HTTPException 404: docente no encontrado.
        HTTPException 500: error de base de datos.
    """
    try:
        docente = ejecutar_consulta(
            "SELECT id, nombre, aula_id FROM usuarios WHERE id = %s AND rol = 'docente'",
            (docente_id,),
            fetchone=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not docente:
        raise HTTPException(status_code=404, detail="Docente no encontrado.")

    aula_id = docente["aula_id"]

    try:
        # Info del aula
        aula = ejecutar_consulta(
            "SELECT id, nombre, puntos_totales FROM aulas WHERE id = %s",
            (aula_id,),
            fetchone=True,
        )

        # Top 25 alumnos del aula por puntos
        alumnos = ejecutar_consulta(
            "SELECT u.id, u.nombre, COALESCE(SUM(r.puntos), 0) AS puntos "
            "FROM usuarios u "
            "LEFT JOIN registro_residuos r ON r.alumno_id = u.id "
            "WHERE u.aula_id = %s AND u.rol = 'alumno' "
            "GROUP BY u.id, u.nombre "
            "ORDER BY puntos DESC "
            "LIMIT 25",
            (aula_id,),
            fetchall=True,
        )

        # Ranking del aula entre todas las aulas
        ranking = ejecutar_consulta(
            "SELECT id, nombre, puntos_totales FROM aulas ORDER BY puntos_totales DESC",
            fetchall=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Calcular posición del aula
    posicion = next(
        (i + 1 for i, a in enumerate(ranking or []) if a["id"] == aula_id),
        None,
    )

    return {
        "docente": {"id": docente["id"], "nombre": docente["nombre"]},
        "aula": aula,
        "posicion_ranking": posicion,
        "total_aulas": len(ranking) if ranking else 0,
        "alumnos": alumnos or [],
    }


# ── 6. DASHBOARD DIRECTOR (MÉTRICAS GLOBALES) ─

@app.get("/api/dashboard/director")
async def dashboard_director():
    """
    Métricas globales para el panel del director / administrador:
    total de residuos clasificados, desglose por categoría, ranking de aulas,
    y actividad reciente.

    Returns:
        dict con total_registros, desglose_categorias, ranking_aulas y actividad_reciente.

    Raises:
        HTTPException 500: error de base de datos.
    """
    try:
        total = ejecutar_consulta(
            "SELECT COUNT(*) AS total FROM registro_residuos",
            fetchone=True,
        )

        por_categoria = ejecutar_consulta(
            "SELECT categoria, COUNT(*) AS cantidad, COALESCE(SUM(puntos), 0) AS puntos "
            "FROM registro_residuos GROUP BY categoria ORDER BY cantidad DESC",
            fetchall=True,
        )

        ranking_aulas = ejecutar_consulta(
            "SELECT id, nombre, puntos_totales FROM aulas ORDER BY puntos_totales DESC",
            fetchall=True,
        )

        actividad_reciente = ejecutar_consulta(
            "SELECT r.id, u.nombre AS alumno, a.nombre AS aula, "
            "       r.categoria, r.puntos, r.fecha "
            "FROM registro_residuos r "
            "JOIN usuarios u ON u.id = r.alumno_id "
            "JOIN aulas a ON a.id = r.aula_id "
            "ORDER BY r.fecha DESC LIMIT 20",
            fetchall=True,
        )

        total_alumnos = ejecutar_consulta(
            "SELECT COUNT(*) AS total FROM usuarios WHERE rol = 'alumno'",
            fetchone=True,
        )

        total_docentes = ejecutar_consulta(
            "SELECT COUNT(*) AS total FROM usuarios WHERE rol = 'docente'",
            fetchone=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "total_registros": total["total"] if total else 0,
        "total_alumnos": total_alumnos["total"] if total_alumnos else 0,
        "total_docentes": total_docentes["total"] if total_docentes else 0,
        "desglose_categorias": por_categoria or [],
        "ranking_aulas": ranking_aulas or [],
        "actividad_reciente": _serializar_fechas(actividad_reciente or []),
    }


# ─────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────

def _serializar_fechas(registros: list[dict]) -> list[dict]:
    """
    Convierte objetos datetime en strings ISO 8601 para JSON serialization.

    Args:
        registros: lista de dicts provenientes de la BD.

    Returns:
        La misma lista con campos datetime convertidos a str.
    """
    for reg in registros:
        for key, val in reg.items():
            if isinstance(val, datetime):
                reg[key] = val.isoformat()
    return registros


# ─────────────────────────────────────────────
# Punto de entrada para desarrollo local
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
