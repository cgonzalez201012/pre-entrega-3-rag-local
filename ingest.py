"""Ingesta de documentos: /data -> limpieza -> chunking por TOKENS -> ChromaDB persistente.

Uso:
    python ingest.py              # indexa /data (si la base ya esta al dia, no hace nada)
    python ingest.py --reset      # borra la coleccion y la reconstruye desde cero
    python ingest.py --simular    # embeddings simulados (sin API key, solo para probar la logica)

Decisiones de diseno (para defenderlas):
* El ID de cada fragmento es un hash de (archivo + texto). Es determinista: reindexar el
  mismo contenido produce los mismos IDs, asi que NO se vuelve a llamar a la API de
  embeddings ni se duplican fragmentos. Si un archivo cambia, solo cambian sus fragmentos.
* La base mantiene un espejo exacto de /data: los fragmentos de archivos cambiados o
  eliminados se borran.
* La coleccion guarda el nombre del modelo de embeddings con el que se creo. Al abrirla con
  otro modelo se lanza `EmbeddingsNoCoincidenError` (error #1 de la consigna).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
import sys
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

import config
from embeddings import crear_embeddings, nombre_del_modelo
from errores import EmbeddingsNoCoincidenError, IngestaError, RagError
from schemas import ReporteIngesta

logger = logging.getLogger("rag.ingesta")

ContadorDeTokens = Callable[[str], int]

_ENCABEZADO_MD = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


# --------------------------------------------------------------------------- tokens
def contar_tokens_aproximado(texto: str) -> int:
    """Estimacion gruesa (caracteres / CARACTERES_POR_TOKEN_APROX). Solo es el plan B."""
    return math.ceil(len(texto) / config.CARACTERES_POR_TOKEN_APROX)


def crear_contador_de_tokens() -> tuple[ContadorDeTokens, str]:
    """Devuelve (funcion que cuenta tokens, descripcion de como se mide).

    Usa tiktoken. tiktoken baja su vocabulario de internet la primera vez; si no hay red
    (por ejemplo un entorno de CI), se usa una aproximacion y se avisa en el log.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(config.TOKEN_ENCODING)
    except Exception as exc:  # noqa: BLE001 - ImportError o errores de red al bajar el vocabulario
        logger.warning(
            "tiktoken no pudo cargar '%s' (%s). Se usa una aproximacion de %s caracteres por token.",
            config.TOKEN_ENCODING, type(exc).__name__, config.CARACTERES_POR_TOKEN_APROX,
        )
        return contar_tokens_aproximado, f"aproximado ({config.CARACTERES_POR_TOKEN_APROX} caracteres por token)"

    def contar(texto: str) -> int:
        return len(encoding.encode(texto, disallowed_special=()))

    return contar, f"tiktoken/{config.TOKEN_ENCODING}"


# --------------------------------------------------------------------------- carga
def limpiar_texto(texto: str) -> str:
    """Preprocesamiento minimo: saltos de linea uniformes, sin espacios al final, sin bloques en blanco."""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = "\n".join(linea.rstrip() for linea in texto.split("\n"))
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def cargar_documentos(data_dir: Path | str = config.DATA_DIR) -> list[Document]:
    """Lee los .md / .txt de la carpeta (orden alfabetico) y devuelve un Document por archivo."""
    carpeta = Path(data_dir)
    if not carpeta.is_dir():
        raise IngestaError(f"No existe la carpeta de documentos: {carpeta}")

    archivos = sorted(
        p for p in carpeta.iterdir()
        if p.is_file() and p.suffix.lower() in config.EXTENSIONES_VALIDAS
    )
    if not archivos:
        raise IngestaError(
            f"No hay archivos {', '.join(config.EXTENSIONES_VALIDAS)} en {carpeta}. "
            "Agrega al menos uno para poder indexar."
        )

    documentos: list[Document] = []
    for ruta in archivos:
        try:
            crudo = ruta.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise IngestaError(f"{ruta.name} no esta codificado en UTF-8: guardalo como UTF-8.") from exc
        texto = limpiar_texto(crudo)
        if not texto:
            logger.warning("%s esta vacio: se omite.", ruta.name)
            continue
        documentos.append(Document(page_content=texto, metadata={"source": ruta.name}))

    if not documentos:
        raise IngestaError(f"Todos los archivos de {carpeta} estan vacios.")
    return documentos


# --------------------------------------------------------------------------- chunking
def crear_splitter(
    contador: ContadorDeTokens,
    chunk_size: int = config.CHUNK_SIZE_TOKENS,
    chunk_overlap: int = config.CHUNK_OVERLAP_TOKENS,
    extension: str = ".md",
) -> RecursiveCharacterTextSplitter:
    """RecursiveCharacterTextSplitter que MIDE en tokens (length_function).

    Para .md usa los separadores de Markdown (titulos, parrafos, lineas) en ese orden de
    prioridad, asi un fragmento intenta no cortar una seccion a la mitad.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")
    opciones = dict(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=contador,
        add_start_index=True,
    )
    if extension.lower() == ".md":
        return RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN, **opciones)
    return RecursiveCharacterTextSplitter(**opciones)


def id_de_fragmento(source: str, texto: str) -> str:
    """ID determinista (hash del contenido + archivo). Mismo contenido => mismo ID."""
    return hashlib.sha256(f"{source}\n{texto}".encode("utf-8")).hexdigest()[:16]


MAX_TITULOS_EN_LA_CITA = 3
SEPARADOR_DE_TITULOS = " / "


def _seccion_del_fragmento(texto: str, inicio: int, fin: int) -> str:
    """Titulos de Markdown que cubre el fragmento [inicio, fin), para poder citarlo con precision.

    Un fragmento de ~500 tokens suele cruzar secciones: empieza al final de una y sigue en la
    siguiente. Citar solo el titulo donde EMPIEZA haria que la cita apunte a la seccion equivocada
    (la respuesta puede estar en la segunda). Por eso se listan todos, en orden:

        "3. Medicion y reporte / 4. Clasificacion y resolucion de incidentes"

    Reglas: el titulo en vigor al inicio entra primero, salvo que sea el del documento (nivel 1) y
    el fragmento tenga subtitulos propios (el titulo del documento no ayuda a ubicar nada). Se
    muestran como maximo MAX_TITULOS_EN_LA_CITA; si hay mas, se termina con "...".
    """
    previo, nivel_previo, dentro = "", 0, []
    for m in _ENCABEZADO_MD.finditer(texto):
        if m.start() >= fin:
            break
        nivel = len(m.group(1))
        if m.start() <= inicio:
            previo, nivel_previo = m.group(2), nivel
        elif nivel > 1:
            dentro.append(m.group(2))
    titulos = ([previo] if previo and not (nivel_previo == 1 and dentro) else []) + dentro
    if len(titulos) > MAX_TITULOS_EN_LA_CITA:
        titulos = titulos[:MAX_TITULOS_EN_LA_CITA] + ["..."]
    return SEPARADOR_DE_TITULOS.join(titulos)


def fragmentar(
    documentos: list[Document],
    contador: ContadorDeTokens,
    chunk_size: int = config.CHUNK_SIZE_TOKENS,
    chunk_overlap: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[Document]:
    """Parte cada documento en fragmentos y les agrega metadatos para poder citar la fuente."""
    fragmentos: list[Document] = []
    vistos: set[str] = set()
    for doc in documentos:
        source = doc.metadata["source"]
        splitter = crear_splitter(contador, chunk_size, chunk_overlap, Path(source).suffix)
        for indice, fragmento in enumerate(splitter.split_documents([doc])):
            chunk_id = id_de_fragmento(source, fragmento.page_content)
            if chunk_id in vistos:  # fragmento idéntico dentro del mismo archivo
                continue
            vistos.add(chunk_id)
            inicio = fragmento.metadata["start_index"]
            fragmento.metadata.update(
                chunk_id=chunk_id,
                chunk_index=indice,
                seccion=_seccion_del_fragmento(doc.page_content, inicio, inicio + len(fragmento.page_content)),
                tokens=contador(fragmento.page_content),
            )
            fragmentos.append(fragmento)
    return fragmentos


# --------------------------------------------------------------------------- ChromaDB
def abrir_coleccion(
    embeddings: Embeddings,
    persist_dir: Path | str = config.PERSIST_DIR,
    coleccion: str = config.COLLECTION_NAME,
    *,
    reset: bool = False,
) -> Chroma:
    """Abre (o crea) la coleccion persistente y verifica que el modelo de embeddings coincida.

    Se usa tanto al indexar como al consultar, para que sea imposible mezclar modelos.
    `reset=True` borra la coleccion (por la API de ChromaDB, no borrando archivos) y la crea de nuevo.
    """
    modelo = nombre_del_modelo(embeddings)
    cliente = chromadb.PersistentClient(path=str(persist_dir))  # persistente: sobrevive al cierre del script

    if reset:
        try:
            cliente.delete_collection(coleccion)
        except NotFoundError:
            pass  # no habia nada que borrar

    try:
        registrado = (cliente.get_collection(coleccion).metadata or {}).get("embedding_model")
    except NotFoundError:
        registrado = None  # coleccion nueva
    if registrado and registrado != modelo:
        raise EmbeddingsNoCoincidenError(
            f"La base '{coleccion}' se creo con el modelo '{registrado}' y ahora se usa '{modelo}'. "
            "Los vectores de modelos distintos no son comparables. "
            "Reconstruila con: python ingest.py --reset"
        )

    return Chroma(
        client=cliente,
        collection_name=coleccion,
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},  # similitud coseno, el estandar en RAG
        collection_metadata={"embedding_model": modelo},
    )


def ingestar(
    data_dir: Path | str = config.DATA_DIR,
    persist_dir: Path | str = config.PERSIST_DIR,
    embeddings: Embeddings | None = None,
    *,
    coleccion: str = config.COLLECTION_NAME,
    chunk_size: int = config.CHUNK_SIZE_TOKENS,
    chunk_overlap: int = config.CHUNK_OVERLAP_TOKENS,
    reset: bool = False,
    contador: ContadorDeTokens | None = None,
) -> ReporteIngesta:
    """Sincroniza la base vectorial con la carpeta de documentos y devuelve un reporte."""
    if contador is None:
        contador, medicion = crear_contador_de_tokens()
    else:
        medicion = "contador inyectado"
    embeddings = embeddings or crear_embeddings()

    documentos = cargar_documentos(data_dir)
    fragmentos = fragmentar(documentos, contador, chunk_size, chunk_overlap)

    db = abrir_coleccion(embeddings, persist_dir, coleccion, reset=reset)
    existentes = set(db.get(include=[])["ids"])
    vigentes = {f.metadata["chunk_id"] for f in fragmentos}

    nuevos = [f for f in fragmentos if f.metadata["chunk_id"] not in existentes]
    obsoletos = sorted(existentes - vigentes)

    if obsoletos:
        db.delete(ids=obsoletos)
    if nuevos:
        db.add_documents(nuevos, ids=[f.metadata["chunk_id"] for f in nuevos])

    tokens = [f.metadata["tokens"] for f in fragmentos]
    return ReporteIngesta(
        archivos=len(documentos),
        fragmentos=len(fragmentos),
        nuevos=len(nuevos),
        sin_cambios=len(fragmentos) - len(nuevos),
        eliminados=len(obsoletos),
        tokens_min=min(tokens),
        tokens_max=max(tokens),
        tokens_promedio=round(sum(tokens) / len(tokens), 1),
        medicion_tokens=medicion,
        modelo_embeddings=nombre_del_modelo(embeddings),
        coleccion=coleccion,
        persist_dir=str(persist_dir),
    )


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Indexa los documentos de /data en ChromaDB.")
    parser.add_argument("--data", default=str(config.DATA_DIR), help="carpeta con los .md/.txt")
    parser.add_argument("--persist-dir", default=str(config.PERSIST_DIR), help="carpeta de la base vectorial")
    parser.add_argument("--reset", action="store_true", help="borra la coleccion y la reconstruye desde cero")
    parser.add_argument("--simular", action="store_true", help="embeddings simulados (sin API key)")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        reporte = ingestar(
            args.data, args.persist_dir, crear_embeddings(simulado=args.simular), reset=args.reset
        )
    except RagError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    estado = "la base ya estaba al dia (no se escribio nada)" if reporte.base_al_dia else "base actualizada"
    print(f"\nIngesta completa: {estado}")
    print(f"  Archivos leidos ....... {reporte.archivos}")
    print(f"  Fragmentos vigentes ... {reporte.fragmentos}")
    print(f"  Nuevos (embebidos) .... {reporte.nuevos}")
    print(f"  Sin cambios ........... {reporte.sin_cambios}")
    print(f"  Eliminados ............ {reporte.eliminados}")
    print(f"  Tokens por fragmento .. min {reporte.tokens_min} / prom {reporte.tokens_promedio} / max {reporte.tokens_max}")
    print(f"  Medicion de tokens .... {reporte.medicion_tokens}")
    print(f"  Modelo de embeddings .. {reporte.modelo_embeddings}")
    print(f"  Base en disco ......... {reporte.persist_dir} (coleccion '{reporte.coleccion}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
