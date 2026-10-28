"""Configuracion central del sistema RAG.

Todas las decisiones de diseno (tamano de fragmento, top_k, modelo de embeddings)
viven aca, en un solo lugar, para poder defenderlas y cambiarlas sin buscar por el codigo.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"            # documentos de origen (.md / .txt)
PERSIST_DIR = BASE_DIR / "vectorstore"  # carpeta donde ChromaDB guarda la base en disco
COLLECTION_NAME = "gobierno_de_datos"
EXTENSIONES_VALIDAS = (".md", ".txt")

# --- Embeddings -------------------------------------------------------------
# El MISMO modelo se usa para indexar y para consultar (error comun #1 de la consigna).
# El nombre se guarda en la coleccion y se verifica al abrirla (ver ingest.py).
EMBEDDING_MODEL = "gemini-embedding-001"

# --- Chunking (consigna: 500 tokens con 50 de overlap) ----------------------
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOKEN_ENCODING = "cl100k_base"      # codificacion de tiktoken usada para MEDIR fragmentos
CARACTERES_POR_TOKEN_APROX = 3.5    # solo se usa si tiktoken no puede cargar su vocabulario (sin red)

# --- Recuperacion -----------------------------------------------------------
TOP_K = 4                           # la consigna pide entre 3 y 5
TOP_K_MIN, TOP_K_MAX = 3, 5         # mas de 5 fragmentos degrada la atencion del modelo (Lost in the Middle)

# --- Generacion -------------------------------------------------------------
LLM_MODEL = "claude-haiku-4-5"
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT_S = 60.0
MAX_INTENTOS = 3                    # intentos totales (el primero + los reintentos)
