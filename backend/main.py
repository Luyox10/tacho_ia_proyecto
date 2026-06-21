"""
main.py
───────
Backend FastAPI para el proyecto de reciclaje escolar con IA.

Endpoints:
  • POST /api/login                         – Login unificado (DNI o email)
  • GET  /api/tacho/identificar/{dni}       – Identificación rápida para simulador físico
  • POST /api/simulador/clasificar          – Clasificación de imagen con modelo Keras
  • GET  /api/dashboard/alumno/{usuario_id}  – Panel del alumno (puntos + logros)
  • GET  /api/dashboard/docente/{usuario_id} – Panel del docente (alumnos + ranking)
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
async def login(dni: Optional[str] = Form(None), correo: Optional[str] = Form(None),
                contrasena: str = Form(...)):
    """
    Login unificado para la plataforma web.

    - Si se envía **dni**: busca en `usuarios` con rol alumno o docente.
    - Si se envía **correo**: busca en `usuarios` con rol director.

    Form fields:
        dni (opcional), correo (opcional), contrasena (obligatorio).

    Returns:
        dict con id, nombre, rol y aula_id del usuario autenticado.

    Raises:
        HTTPException 400: si no se envía ni DNI ni email.
        HTTPException 401: credenciales inválidas.
        HTTPException 500: error interno de base de datos.
    """
    if not dni and not correo:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar 'dni' o 'correo' para iniciar sesión.",
        )

    try:
        if dni:
            usuario = ejecutar_consulta(
                "SELECT id, nombre, apellido, rol, aula_id, contrasena_hash "
                "FROM usuarios WHERE dni = %s AND rol IN ('alumno', 'docente')",
                (dni,),
                fetchone=True,
            )
        else:
            usuario = ejecutar_consulta(
                "SELECT id, nombre, apellido, rol, aula_id, contrasena_hash "
                "FROM usuarios WHERE correo = %s AND rol = 'director'",
                (correo,),
                fetchone=True,
            )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not usuario:
        raise HTTPException(status_code=401, detail="Usuario no encontrado.")

    if usuario["contrasena_hash"] != contrasena:
        raise HTTPException(status_code=401, detail="Contraseña incorrecta.")

    return {
        "usuario_id": usuario["id"],
        "nombre": usuario["nombre"],
        "apellido": usuario["apellido"],
        "rol": usuario["rol"],
        "aula_id": usuario["aula_id"],
    }



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
        "usuario_id": alumno["id"],
        "nombre": alumno["nombre"],
        "aula_id": alumno["aula_id"],
    }


# ── 3. CLASIFICACIÓN CON IA ──────────────────

@app.post("/api/simulador/clasificar")
async def clasificar_residuo(
    usuario_id: int = Form(...),
    imagen_base64: Optional[str] = Form(None),
    imagen_archivo: Optional[UploadFile] = File(None),
):
    """
    Recibe una imagen (base64 o multipart), la clasifica con el modelo Keras
    y registra el resultado en la base de datos.

    Form fields:
        usuario_id:     ID del alumno que deposita el residuo.
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
            # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
            b64_clean = imagen_base64
            if "," in b64_clean:
                b64_clean = b64_clean.split(",", 1)[1]
            # Strip whitespace/newlines that can corrupt decoding
            b64_clean = b64_clean.strip()
            imagen_bytes = base64.b64decode(b64_clean)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"El base64 de la imagen es inválido: {e}",
            )
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
            (usuario_id,),
            fetchone=True,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al consultar alumno en BD: {e}",
        )

    if not alumno:
        raise HTTPException(status_code=404, detail="Alumno no encontrado.")

    # ── Clasificar con modelo IA ──
    try:
        modelo = cargar_modelo()
        tensor = preprocesar_imagen(imagen_bytes)
        prediccion = modelo.predict(tensor, verbose=0)
        indice = int(np.argmax(prediccion[0]))
        tipo_residuo = CATEGORIAS[indice]
        confianza = float(prediccion[0][indice])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la inferencia del modelo IA: {e}",
        )

    # ── Registrar en BD ──
    puntos = 10
    try:
        ejecutar_consulta(
            "INSERT INTO registro_residuos (usuario_id, tipo_residuo, puntos_ganados, fecha_registro) "
            "VALUES (%s, %s, %s, %s)",
            (usuario_id, tipo_residuo, puntos, datetime.utcnow()),
            commit=True,
        )
        ejecutar_consulta(
            "UPDATE aulas SET puntos_totales = puntos_totales + %s WHERE id = %s",
            (puntos, alumno["aula_id"]),
            commit=True,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Clasificación exitosa ({tipo_residuo}) pero error al guardar en BD: {e}",
        )

    # ── Verificar logros desbloqueados ──
    nuevo_logro = _verificar_logros(usuario_id, tipo_residuo)

    return {
        "tipo_residuo_detectado": tipo_residuo,
        "confianza": round(confianza, 4),
        "puntos_sumados": puntos,
        "nuevo_logro": nuevo_logro,
    }


# ── 3b. LÓGICA DE LOGROS ─────────────────────

LOGROS_CONFIG = [
    {"nombre": "Eco-Guardian", "condicion": "total", "umbral": 50, "descripcion": "Reciclar 50 residuos en total"},
    {"nombre": "Plastic King", "condicion": "plastic", "umbral": 10, "descripcion": "Reciclar 10 items de plastico"},
    {"nombre": "Paper Saver", "condicion": "paper", "umbral": 10, "descripcion": "Reciclar 10 items de papel"},
    {"nombre": "Glass Collector", "condicion": "glass", "umbral": 10, "descripcion": "Reciclar 10 items de vidrio"},
    {"nombre": "Metal Master", "condicion": "metal", "umbral": 10, "descripcion": "Reciclar 10 items de metal"},
    {"nombre": "Organic Hero", "condicion": "organic", "umbral": 10, "descripcion": "Reciclar 10 items organicos"},
    {"nombre": "First Step", "condicion": "total", "umbral": 1, "descripcion": "Reciclar tu primer residuo"},
    {"nombre": "Recycler Pro", "condicion": "total", "umbral": 25, "descripcion": "Reciclar 25 residuos en total"},
    {"nombre": "Century Club", "condicion": "total", "umbral": 100, "descripcion": "Reciclar 100 residuos en total"},
]


def _verificar_logros(usuario_id: int, tipo_residuo: str) -> Optional[dict]:
    """
    Verifica si el alumno ha desbloqueado un nuevo logro tras su último registro.
    Compara el conteo actual de residuos con los umbrales definidos en LOGROS_CONFIG.

    Args:
        usuario_id: ID del alumno.
        tipo_residuo: categoría del residuo recién clasificado.

    Returns:
        dict con nombre del logro si se desbloqueó uno nuevo, None en caso contrario.
    """
    try:
        # Conteo total
        total_row = ejecutar_consulta(
            "SELECT COUNT(*) AS cnt FROM registro_residuos WHERE usuario_id = %s",
            (usuario_id,),
            fetchone=True,
        )
        total = total_row["cnt"] if total_row else 0

        # Conteo por categoría del residuo actual
        cat_row = ejecutar_consulta(
            "SELECT COUNT(*) AS cnt FROM registro_residuos WHERE usuario_id = %s AND tipo_residuo = %s",
            (usuario_id, tipo_residuo),
            fetchone=True,
        )
        cat_count = cat_row["cnt"] if cat_row else 0

        # Logros ya obtenidos
        logros_existentes = ejecutar_consulta(
            "SELECT nombre_medalla FROM logros WHERE usuario_id = %s",
            (usuario_id,),
            fetchall=True,
        )
        nombres_existentes = {l["nombre_medalla"] for l in (logros_existentes or [])}

        # Verificar cada logro
        nuevo_logro = None
        for logro in LOGROS_CONFIG:
            if logro["nombre"] in nombres_existentes:
                continue

            cumple = False
            if logro["condicion"] == "total" and total >= logro["umbral"]:
                cumple = True
            elif logro["condicion"] == tipo_residuo and cat_count >= logro["umbral"]:
                cumple = True

            if cumple:
                ejecutar_consulta(
                    "INSERT INTO logros (usuario_id, nombre_medalla, fecha_ganado) VALUES (%s, %s, %s)",
                    (usuario_id, logro["nombre"], datetime.utcnow()),
                    commit=True,
                )
                nuevo_logro = {"nombre": logro["nombre"], "descripcion": logro["descripcion"]}
                break  # Solo notificar un logro por clasificación

        return nuevo_logro

    except RuntimeError:
        return None


# ── 3c. ENDPOINT LOGROS DEL ALUMNO ──────────

@app.get("/api/logros/{usuario_id}")
async def obtener_logros(usuario_id: int):
    """
    Devuelve todos los logros del alumno y la lista completa de logros disponibles
    para mostrar cuáles están desbloqueados y cuáles faltan.

    Path params:
        usuario_id: ID del alumno.

    Returns:
        dict con logros_obtenidos y logros_disponibles.

    Raises:
        HTTPException 500: error de base de datos.
    """
    try:
        obtenidos = ejecutar_consulta(
            "SELECT nombre_medalla, fecha_ganado FROM logros WHERE usuario_id = %s ORDER BY fecha_ganado DESC",
            (usuario_id,),
            fetchall=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    nombres_obtenidos = {l["nombre_medalla"] for l in (obtenidos or [])}

    disponibles = []
    for logro in LOGROS_CONFIG:
        disponibles.append({
            "nombre": logro["nombre"],
            "descripcion": logro["descripcion"],
            "umbral": logro["umbral"],
            "condicion": logro["condicion"],
            "desbloqueado": logro["nombre"] in nombres_obtenidos,
        })

    return {
        "logros_obtenidos": _serializar_fechas(obtenidos or []),
        "logros_disponibles": disponibles,
    }


# ── 4. DASHBOARD ALUMNO ──────────────────────

@app.get("/api/dashboard/alumno/{usuario_id}")
async def dashboard_alumno(usuario_id: int):
    """
    Datos para el panel del alumno: puntos acumulados, cantidad de registros
    y logros obtenidos.

    Path params:
        usuario_id: ID del usuario con rol alumno.

    Returns:
        dict con nombre, puntos_totales, total_registros y lista de logros.

    Raises:
        HTTPException 404: alumno no encontrado.
        HTTPException 500: error de base de datos.
    """
    try:
        alumno = ejecutar_consulta(
            "SELECT id, nombre, apellido, aula_id FROM usuarios WHERE id = %s AND rol = 'alumno'",
            (usuario_id,),
            fetchone=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not alumno:
        raise HTTPException(status_code=404, detail="Alumno no encontrado.")

    try:
        resumen = ejecutar_consulta(
            "SELECT COALESCE(SUM(puntos_ganados), 0) AS puntos_totales, "
            "       COUNT(*) AS total_registros "
            "FROM registro_residuos WHERE usuario_id = %s",
            (usuario_id,),
            fetchone=True,
        )
        logros = ejecutar_consulta(
            "SELECT id, nombre_medalla, fecha_ganado "
            "FROM logros WHERE usuario_id = %s ORDER BY fecha_ganado DESC",
            (usuario_id,),
            fetchall=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "usuario_id": alumno["id"],
        "nombre": alumno["nombre"],
        "apellido": alumno["apellido"],
        "aula_id": alumno["aula_id"],
        "puntos_totales": resumen["puntos_totales"] if resumen else 0,
        "total_registros": resumen["total_registros"] if resumen else 0,
        "logros": logros or [],
    }


# ── 5. DASHBOARD DOCENTE ─────────────────────

@app.get("/api/dashboard/docente/{usuario_id}")
async def dashboard_docente(usuario_id: int):
    """
    Datos para el panel del docente: lista de hasta 25 alumnos de su aula
    con sus puntos individuales, y el ranking del aula a nivel escuela.

    Path params:
        usuario_id: ID del usuario con rol docente.

    Returns:
        dict con aula, lista de alumnos (top 25) y posición del aula en ranking.

    Raises:
        HTTPException 404: docente no encontrado.
        HTTPException 500: error de base de datos.
    """
    try:
        docente = ejecutar_consulta(
            "SELECT id, nombre, apellido, aula_id FROM usuarios WHERE id = %s AND rol = 'docente'",
            (usuario_id,),
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
            "SELECT id, grado_seccion, puntos_totales FROM aulas WHERE id = %s",
            (aula_id,),
            fetchone=True,
        )

        # Top 25 alumnos del aula por puntos
        alumnos = ejecutar_consulta(
            "SELECT u.id, u.nombre, u.apellido, COALESCE(SUM(r.puntos_ganados), 0) AS puntos "
            "FROM usuarios u "
            "LEFT JOIN registro_residuos r ON r.usuario_id = u.id "
            "WHERE u.aula_id = %s AND u.rol = 'alumno' "
            "GROUP BY u.id, u.nombre, u.apellido "
            "ORDER BY puntos DESC "
            "LIMIT 25",
            (aula_id,),
            fetchall=True,
        )

        # Ranking del aula entre todas las aulas
        ranking = ejecutar_consulta(
            "SELECT id, grado_seccion, puntos_totales FROM aulas ORDER BY puntos_totales DESC",
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
        "docente": {"usuario_id": docente["id"], "nombre": docente["nombre"], "apellido": docente["apellido"]},
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
            "SELECT tipo_residuo, COUNT(*) AS cantidad, COALESCE(SUM(puntos_ganados), 0) AS puntos "
            "FROM registro_residuos GROUP BY tipo_residuo ORDER BY cantidad DESC",
            fetchall=True,
        )

        ranking_aulas = ejecutar_consulta(
            "SELECT id, grado_seccion, puntos_totales FROM aulas ORDER BY puntos_totales DESC",
            fetchall=True,
        )

        actividad_reciente = ejecutar_consulta(
            "SELECT r.id, u.nombre AS alumno, u.apellido, a.grado_seccion AS aula, "
            "       r.tipo_residuo, r.puntos_ganados, r.fecha_registro "
            "FROM registro_residuos r "
            "JOIN usuarios u ON u.id = r.usuario_id "
            "JOIN aulas a ON a.id = u.aula_id "
            "ORDER BY r.fecha_registro DESC LIMIT 20",
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


def _serializar_fechas(registros: list[dict]) -> list[dict]:
    for reg in registros:
        for key, val in reg.items():
            if isinstance(val, datetime):
                reg[key] = val.isoformat()
    return registros


# ── 7. ADMIN: LISTAR DOCENTES ──────────────────
@app.get("/api/admin/docentes")
async def listar_docentes():
    """
    Lista todos los docentes con su aula asignada.
    """
    try:
        docentes = ejecutar_consulta(
            "SELECT u.id, u.dni, u.nombre, u.apellido, u.correo, u.aula_id, "
            "       a.grado_seccion "
            "FROM usuarios u "
            "LEFT JOIN aulas a ON a.id = u.aula_id "
            "WHERE u.rol = 'docente' "
            "ORDER BY u.nombre ASC",
            fetchall=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"docentes": docentes or []}


# ── 8. ADMIN: LISTAR ALUMNOS ──────────────────
@app.get("/api/admin/alumnos")
async def listar_alumnos():
    """
    Lista todos los alumnos con su aula y puntos individuales.
    """
    try:
        alumnos = ejecutar_consulta(
            "SELECT u.id, u.dni, u.nombre, u.apellido, u.aula_id, "
            "       a.grado_seccion, "
            "       COALESCE(SUM(r.puntos_ganados), 0) AS puntos "
            "FROM usuarios u "
            "LEFT JOIN aulas a ON a.id = u.aula_id "
            "LEFT JOIN registro_residuos r ON r.usuario_id = u.id "
            "WHERE u.rol = 'alumno' "
            "GROUP BY u.id, u.dni, u.nombre, u.apellido, u.aula_id, a.grado_seccion "
            "ORDER BY u.nombre ASC",
            fetchall=True,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"alumnos": alumnos or []}


# ── 9. ADMIN: CREAR ALUMNO ──────────────────────
@app.post("/api/admin/alumnos")
async def crear_alumno(
    dni: str = Form(...),
    nombre: str = Form(...),
    apellido: str = Form(...),
    contrasena_hash: str = Form(...),
    aula_id: int = Form(...),
):
    """
    Registra un nuevo alumno en la tabla `usuarios` con rol='alumno'.

    Form fields:
        dni, nombre, apellido, contrasena_hash, aula_id.

    Returns:
        dict con el id generado y los datos del alumno creado.

    Raises:
        HTTPException 409: si el DNI ya existe.
        HTTPException 500: error de base de datos.
    """
    try:
        existente = ejecutar_consulta(
            "SELECT id FROM usuarios WHERE dni = %s",
            (dni,),
            fetchone=True,
        )
        if existente:
            raise HTTPException(status_code=409, detail=f"El DNI {dni} ya está registrado.")

        nuevo_id = ejecutar_consulta(
            "INSERT INTO usuarios (dni, nombre, apellido, contrasena_hash, rol, aula_id, fecha_creacion) "
            "VALUES (%s, %s, %s, %s, 'alumno', %s, %s)",
            (dni, nombre, apellido, contrasena_hash, aula_id, datetime.utcnow()),
            commit=True,
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "id": nuevo_id,
        "dni": dni,
        "nombre": nombre,
        "apellido": apellido,
        "rol": "alumno",
        "aula_id": aula_id,
    }


# ── 10. ADMIN: EDITAR ALUMNO ─────────────────────
@app.put("/api/admin/alumnos/{alumno_id}")
async def editar_alumno(
    alumno_id: int,
    nombre: str = Form(...),
    apellido: str = Form(...),
    aula_id: int = Form(...),
    contrasena_hash: Optional[str] = Form(None),
):
    """
    Actualiza los datos de un alumno existente.
    Si se envía `contrasena_hash`, también actualiza la contraseña.

    Path params:
        alumno_id: ID del alumno a editar.

    Returns:
        dict confirmando los campos actualizados.

    Raises:
        HTTPException 404: alumno no encontrado.
        HTTPException 500: error de base de datos.
    """
    try:
        alumno = ejecutar_consulta(
            "SELECT id FROM usuarios WHERE id = %s AND rol = 'alumno'",
            (alumno_id,),
            fetchone=True,
        )
        if not alumno:
            raise HTTPException(status_code=404, detail="Alumno no encontrado.")

        if contrasena_hash:
            ejecutar_consulta(
                "UPDATE usuarios SET nombre = %s, apellido = %s, aula_id = %s, contrasena_hash = %s "
                "WHERE id = %s AND rol = 'alumno'",
                (nombre, apellido, aula_id, contrasena_hash, alumno_id),
                commit=True,
            )
        else:
            ejecutar_consulta(
                "UPDATE usuarios SET nombre = %s, apellido = %s, aula_id = %s "
                "WHERE id = %s AND rol = 'alumno'",
                (nombre, apellido, aula_id, alumno_id),
                commit=True,
            )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"id": alumno_id, "nombre": nombre, "apellido": apellido, "aula_id": aula_id}


# ── 11. ADMIN: ELIMINAR ALUMNO ───────────────────
@app.delete("/api/admin/alumnos/{alumno_id}")
async def eliminar_alumno(alumno_id: int):
    """
    Elimina un alumno de `usuarios` junto con sus registros de residuos y logros.

    Path params:
        alumno_id: ID del alumno a eliminar.

    Raises:
        HTTPException 404: alumno no encontrado.
        HTTPException 500: error de base de datos.
    """
    try:
        alumno = ejecutar_consulta(
            "SELECT id FROM usuarios WHERE id = %s AND rol = 'alumno'",
            (alumno_id,),
            fetchone=True,
        )
        if not alumno:
            raise HTTPException(status_code=404, detail="Alumno no encontrado.")

        ejecutar_consulta("DELETE FROM logros WHERE usuario_id = %s", (alumno_id,), commit=True)
        ejecutar_consulta("DELETE FROM registro_residuos WHERE usuario_id = %s", (alumno_id,), commit=True)
        ejecutar_consulta("DELETE FROM usuarios WHERE id = %s AND rol = 'alumno'", (alumno_id,), commit=True)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"eliminado": True, "id": alumno_id}


# ── 12. ADMIN: AUDITORIA DOCENTE ──────────────────
@app.get("/api/admin/auditoria")
async def listar_auditoria():
    """
    Lista el registro de acciones de docentes (últimas 50 entradas).
    Si la tabla no existe aún, devuelve lista vacía.
    """
    try:
        registros = ejecutar_consulta(
            "SELECT ad.id, ad.accion, ad.detalle, ad.fecha, "
            "       u.nombre, u.apellido, a.grado_seccion "
            "FROM auditoria_docente ad "
            "JOIN usuarios u ON u.id = ad.docente_id "
            "LEFT JOIN aulas a ON a.id = u.aula_id "
            "ORDER BY ad.fecha DESC LIMIT 50",
            fetchall=True,
        )
    except RuntimeError:
        # Table might not exist yet
        registros = []

    return {"auditoria": _serializar_fechas(registros or [])}


# ─────────────────────────────────────────────
# Punto de entrada para desarrollo local
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
