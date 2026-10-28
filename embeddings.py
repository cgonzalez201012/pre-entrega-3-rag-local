"""Fabrica de embeddings: Gemini real o simulado (para tests y demo sin API key)."""
from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata

from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL
from errores import ConfiguracionError


class EmbeddingsSimulados(Embeddings):
    """Embeddings deterministas basados en palabras (sin red, sin clave).

    NO entienden significado: dos textos se parecen si comparten palabras. Sirven para
    testear la logica del sistema (ingesta, recuperacion, citas) sin gastar cuota, igual
    que `modelo_simulado.py` en la pre-entrega 2. Usan SHA-256 (no `hash()`) para que el
    resultado sea identico entre ejecuciones.
    """

    nombre_modelo = "simulado-bolsa-de-palabras"

    def __init__(self, dimensiones: int = 256) -> None:
        self.dimensiones = dimensiones
        self.llamadas_documentos = 0  # cuantos textos se "embebieron" (los tests lo usan)
        self.llamadas_consultas = 0

    @staticmethod
    def _palabras(texto: str) -> list[str]:
        sin_tildes = unicodedata.normalize("NFD", texto.lower())
        sin_tildes = "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")
        return re.findall(r"[a-z0-9]+", sin_tildes)

    def _vector(self, texto: str) -> list[float]:
        v = [0.0] * self.dimensiones
        for palabra in self._palabras(texto):
            indice = int(hashlib.sha256(palabra.encode()).hexdigest(), 16) % self.dimensiones
            v[indice] += 1.0
        norma = math.sqrt(sum(x * x for x in v))
        return [x / norma for x in v] if norma else v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.llamadas_documentos += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.llamadas_consultas += 1
        return self._vector(text)


def crear_embeddings(simulado: bool = False) -> Embeddings:
    """Devuelve el modelo de embeddings a usar.

    Con `simulado=False` usa Gemini (`gemini-embedding-001`). La libreria pide por
    defecto `RETRIEVAL_DOCUMENT` al indexar y `RETRIEVAL_QUERY` al consultar: son dos
    "modos" del MISMO modelo, pensados para busqueda (no es mezclar modelos).
    """
    if simulado:
        return EmbeddingsSimulados()

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise ConfiguracionError(
            "Falta GOOGLE_API_KEY. Crea una clave gratis en https://aistudio.google.com/apikey "
            "y agrega la clave al archivo .env (copia .env.example). Para probar sin clave: --simular."
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def nombre_del_modelo(embeddings: Embeddings) -> str:
    """Nombre con el que se registra el modelo en la coleccion (para detectar desajustes)."""
    return getattr(embeddings, "nombre_modelo", None) or getattr(embeddings, "model", None) or type(embeddings).__name__
