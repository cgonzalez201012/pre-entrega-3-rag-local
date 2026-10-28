"""Capa de recuperacion (retriever): pregunta -> embedding -> fragmentos mas parecidos de ChromaDB.

Es asincrona: el embedding de la pregunta es una llamada de red (a Gemini), asi que se hace con
`await` para no bloquear el event loop. La busqueda en ChromaDB es local (disco), pero es codigo
sincrono, asi que se manda a un hilo con `asyncio.to_thread` por la misma razon.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

import config
from errores import BaseVectorialVaciaError
from ingest import abrir_coleccion
from schemas import FragmentoRecuperado


class Recuperador:
    """Devuelve los `k` fragmentos mas cercanos (distancia coseno) a una pregunta."""

    def __init__(self, db: Chroma, embeddings: Embeddings, k: int = config.TOP_K) -> None:
        if not config.TOP_K_MIN <= k <= config.TOP_K_MAX:
            raise ValueError(
                f"top_k debe estar entre {config.TOP_K_MIN} y {config.TOP_K_MAX} (recibido: {k}). "
                "Pasar demasiados fragmentos al modelo degrada su atencion (Lost in the Middle)."
            )
        self.db = db
        self.embeddings = embeddings
        self.k = k

    async def recuperar(self, pregunta: str) -> list[FragmentoRecuperado]:
        """Fragmentos ordenados del mas cercano al mas lejano; `referencia` es su posicion (1, 2, ...)."""
        if not pregunta or not pregunta.strip():
            raise ValueError("La pregunta esta vacia.")

        # 1) La pregunta se convierte en un vector con el MISMO modelo que se uso para indexar.
        vector = await self.embeddings.aembed_query(pregunta)

        # 2) Busqueda por similitud en ChromaDB. El "score" que devuelve es una DISTANCIA coseno.
        resultados = await asyncio.to_thread(
            self.db.similarity_search_by_vector_with_relevance_scores, vector, self.k
        )

        return [
            FragmentoRecuperado(
                referencia=posicion,
                archivo=doc.metadata.get("source", "?"),
                seccion=doc.metadata.get("seccion", ""),
                chunk_id=doc.metadata.get("chunk_id", ""),
                distancia=max(0.0, float(distancia)),  # evita -0.0000001 por redondeo de punto flotante
                texto=doc.page_content,
            )
            for posicion, (doc, distancia) in enumerate(resultados, start=1)
        ]


def abrir_recuperador(
    embeddings: Embeddings,
    persist_dir: Path | str = config.PERSIST_DIR,
    coleccion: str = config.COLLECTION_NAME,
    k: int = config.TOP_K,
) -> Recuperador:
    """Abre la base ya indexada y arma el retriever.

    Usa `abrir_coleccion` (la misma funcion que usa la ingesta), asi que si la base se creo
    con otro modelo de embeddings lanza EmbeddingsNoCoincidenError en vez de devolver ruido.
    """
    db = abrir_coleccion(embeddings, persist_dir, coleccion)
    if not db.get(limit=1, include=[])["ids"]:
        raise BaseVectorialVaciaError(
            f"La base '{coleccion}' esta vacia o no existe en {persist_dir}. "
            "Indexa los documentos primero con: python ingest.py"
        )
    return Recuperador(db, embeddings, k)
