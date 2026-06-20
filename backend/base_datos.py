"""
base_datos.py
─────────────
Módulo de conexión a la base de datos.
Todas las credenciales se leen de variables de entorno; nunca se hardcodean.
"""

import os

import pymysql
import pymysql.cursors


DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = int(os.environ.get("DB_PORT", 4000))


def obtener_conexion():
    """
    Crea y devuelve una conexión a la base de datos.

    Variables de entorno requeridas:
        DB_HOST     – Host del servidor de base de datos
        DB_USER     – Usuario de la base de datos
        DB_PASSWORD – Contraseña del usuario
        DB_NAME     – Nombre de la base de datos
        DB_PORT     – Puerto (por defecto 4000)

    Returns:
        pymysql.connections.Connection: conexión activa.

    Raises:
        RuntimeError: si alguna variable de entorno falta o la conexión falla.
    """
    required_vars = {"DB_HOST": DB_HOST, "DB_USER": DB_USER,
                     "DB_PASSWORD": DB_PASSWORD, "DB_NAME": DB_NAME}
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise RuntimeError(
            f"Faltan variables de entorno obligatorias: {', '.join(missing)}"
        )

    try:
        return pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor,
            ssl={"ssl": {}},
        )
    except pymysql.MySQLError as e:
        raise RuntimeError(f"Error al conectar con la base de datos: {e}") from e


def ejecutar_consulta(sql: str, params: tuple = None, fetchone: bool = False,
                      fetchall: bool = False, commit: bool = False):
    """
    Ejecuta una consulta SQL de forma segura y devuelve resultados si se solicita.

    Args:
        sql:      Cadena SQL parametrizada (usar %s para parámetros).
        params:   Tupla de parámetros para la consulta.
        fetchone: Si True, devuelve una sola fila como diccionario.
        fetchall: Si True, devuelve todas las filas como lista de diccionarios.
        commit:   Si True, hace commit de la transacción (INSERT / UPDATE / DELETE).

    Returns:
        dict | list[dict] | int | None:
            - dict          si fetchone=True
            - list[dict]    si fetchall=True
            - int (lastrowid) si commit=True
            - None          en cualquier otro caso.

    Raises:
        RuntimeError: ante cualquier error en la ejecución.
    """
    conexion = None
    try:
        conexion = obtener_conexion()
        with conexion.cursor() as cursor:
            cursor.execute(sql, params or ())

            if commit:
                conexion.commit()
                return cursor.lastrowid

            if fetchone:
                return cursor.fetchone()

            if fetchall:
                return cursor.fetchall()

            return None

    except pymysql.MySQLError as e:
        if conexion:
            conexion.rollback()
        raise RuntimeError(f"Error ejecutando consulta: {e}") from e

    finally:
        if conexion:
            conexion.close()
