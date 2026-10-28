"""Generacion grounded: cadena LCEL + PydanticOutputParser + punto de entrada asincrono.

Flujo de `get_rag_response(query)`:
    a) Recuperacion: embedding de la pregunta y busqueda de similitud en ChromaDB (retriever.py).
    b) Prompt: se arma con los fragmentos recuperados numerados [1], [2], ... (prompts.py).
    c) Generacion: llamada ASINCRONA al LLM (Claude) con reintentos.
    d) Parseo: la salida se valida con PydanticOutputParser contra `RespuestaLLM`; los numeros que
       el modelo cita se convierten en fuentes reales (archivo + seccion) en `RespuestaRAG`.
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import ValidationError

import config
from embeddings import crear_embeddings, nombre_del_modelo
from errores import ConfiguracionError, ProveedorError, RagError, RespuestaInvalidaError
from prompts import crear_prompt
from retriever import Recuperador, abrir_recuperador
from schemas import FragmentoRecuperado, Metricas, RespuestaLLM, RespuestaRAG

logger = logging.getLogger("rag")


# --------------------------------------------------------------------------- #
# Errores que vale la pena reintentar
# --------------------------------------------------------------------------- #
def _transitorios_llm() -> tuple[type[BaseException], ...]:
    """Errores de transporte de Anthropic que SI conviene reintentar (red, 429, 5xx, timeout)."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return ()
    return (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)


def _transitorios_gemini() -> tuple[type[BaseException], ...]:
    """Errores 5xx de Gemini (el 429 NO se reintenta: es la cuota gratuita, esperar segundos no alcanza)."""
    try:
        from google.genai import errors
    except ImportError:  # pragma: no cover
        return ()
    return (errors.ServerError,)


ERRORES_REINTENTABLES_LLM: tuple[type[BaseException], ...] = (
    RespuestaInvalidaError,
    ValidationError,
    OutputParserException,
    *_transitorios_llm(),
)


# --------------------------------------------------------------------------- #
# Modelo de lenguaje
# --------------------------------------------------------------------------- #
def build_model(
    model: str | None = None,
    max_tokens: int = config.LLM_MAX_TOKENS,
    timeout_s: float = config.LLM_TIMEOUT_S,
) -> BaseChatModel:
    """Crea el chat model de Claude. La clave se lee de ANTHROPIC_API_KEY (archivo .env)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ConfiguracionError(
            "Falta ANTHROPIC_API_KEY. Agrega la clave al archivo .env (copia .env.example). "
            "Para probar sin claves: python main.py --simular"
        )
    from langchain_anthropic import ChatAnthropic

    # Sin temperature (igual que en la pre-entrega 2). max_retries=0: la unica politica de
    # reintentos es la de LangChain (.with_retry), para poder contar los intentos.
    return ChatAnthropic(model=model or config.LLM_MODEL, max_tokens=max_tokens, timeout=timeout_s, max_retries=0)


def nombre_del_llm(modelo: BaseChatModel) -> str:
    return getattr(modelo, "model", None) or getattr(modelo, "model_name", None) or modelo._llm_type


# --------------------------------------------------------------------------- #
# Transformador de documentos y verificacion de las citas
# --------------------------------------------------------------------------- #
def formatear_contexto(fragmentos: list[FragmentoRecuperado]) -> str:
    """Convierte los fragmentos en el texto numerado que ve el modelo: '[1] archivo · seccion'."""
    bloques = []
    for f in fragmentos:
        titulo = f"{f.archivo} · {f.seccion}" if f.seccion else f.archivo
        bloques.append(f"[{f.referencia}] {titulo}\n{f.texto}")
    return "\n\n".join(bloques)


def _verificar_referencias(datos: dict[str, Any]) -> RespuestaLLM:
    """Pydantic valida la FORMA; esto valida la VERDAD: el modelo solo puede citar fragmentos que recibio.

    Si cita un numero que no existe (por ejemplo [7] habiendo recibido 4), la respuesta se rechaza
    y se reintenta, en vez de devolver una fuente inventada.
    """
    salida: RespuestaLLM = datos["salida"]
    recibidos = len(datos["documentos"])
    inexistentes = [n for n in salida.referencias if n > recibidos]
    if inexistentes:
        raise RespuestaInvalidaError(
            f"El modelo cito el/los fragmento(s) {inexistentes}, pero solo recibio {recibidos}."
        )
    return salida


# --------------------------------------------------------------------------- #
# Logs: intentos, tokens
# --------------------------------------------------------------------------- #
class RagLogHandler(AsyncCallbackHandler):
    """Cuenta las llamadas al modelo y suma los tokens de TODOS los intentos (cada reintento se paga)."""

    def __init__(self) -> None:
        self.intentos = 0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    async def on_chat_model_start(self, serialized, messages, *, tags=None, **kwargs) -> None:
        self.intentos += 1
        if self.intentos == 1:
            logger.info("Llamando al modelo (intento 1)...")
        else:
            logger.warning("REINTENTO: llamando al modelo (intento %d)...", self.intentos)

    async def on_llm_end(self, response, **kwargs) -> None:
        for generaciones in response.generations:
            for generacion in generaciones:
                uso = getattr(getattr(generacion, "message", None), "usage_metadata", None)
                if not uso:
                    continue
                entrada, salida = uso.get("input_tokens", 0), uso.get("output_tokens", 0)
                self.tokens_entrada += entrada
                self.tokens_salida += salida
                logger.info("Tokens del intento %d: entrada=%d, salida=%d.", self.intentos, entrada, salida)


# --------------------------------------------------------------------------- #
# La cadena LCEL
# --------------------------------------------------------------------------- #
def build_chain(
    model: BaseChatModel,
    max_intentos: int = config.MAX_INTENTOS,
    esperar_entre_intentos: bool = True,
) -> Runnable:
    """Arma la cadena que recibe {'pregunta', 'documentos'} y devuelve un RespuestaLLM verificado.

        documentos -> contexto numerado | prompt | modelo | PydanticOutputParser | verificar citas

    Los reintentos (respuesta invalida, red, 429, 5xx) envuelven solo la parte que llama al modelo.
    """
    parser = PydanticOutputParser(pydantic_object=RespuestaLLM)
    prompt = crear_prompt(parser.get_format_instructions())

    generacion = (
        RunnablePassthrough.assign(salida=prompt | model | parser) | RunnableLambda(_verificar_referencias)
    ).with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES_LLM,
        wait_exponential_jitter=esperar_entre_intentos,
        stop_after_attempt=max_intentos,
    )

    transformador_de_documentos = RunnablePassthrough.assign(
        contexto=RunnableLambda(lambda datos: formatear_contexto(datos["documentos"]))
    )
    return transformador_de_documentos | generacion


# --------------------------------------------------------------------------- #
# Sistema completo: retriever + cadena
# --------------------------------------------------------------------------- #
def _traducir_error(exc: BaseException) -> ProveedorError:
    """Convierte un error de un SDK en un mensaje claro y accionable."""
    try:
        from google.genai import errors as gerr

        if isinstance(exc, gerr.ClientError):
            if getattr(exc, "code", None) == 429:
                return ProveedorError(
                    "Gemini: se alcanzo el limite de uso del plan gratuito. Espera un minuto y vuelve a intentar."
                )
            return ProveedorError(
                f"Gemini rechazo la solicitud (codigo {getattr(exc, 'code', '?')}). Revisa GOOGLE_API_KEY en el .env."
            )
    except ImportError:  # pragma: no cover
        pass
    try:
        import anthropic

        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return ProveedorError("Claude rechazo la clave. Revisa ANTHROPIC_API_KEY en el .env.")
    except ImportError:  # pragma: no cover
        pass
    return ProveedorError(f"{type(exc).__name__}: {exc}")


class SistemaRAG:
    """Une el retriever con la cadena y arma la `RespuestaRAG` (con fuentes y metricas)."""

    def __init__(
        self,
        recuperador: Recuperador,
        chain: Runnable,
        modelo_llm: str,
        modelo_embeddings: str,
        max_intentos: int = config.MAX_INTENTOS,
        esperar_entre_intentos: bool = True,
    ) -> None:
        self.recuperador = recuperador
        self.chain = chain
        self.modelo_llm = modelo_llm
        self.modelo_embeddings = modelo_embeddings
        self._recuperar = RunnableLambda(recuperador.recuperar).with_retry(
            retry_if_exception_type=_transitorios_gemini(),
            wait_exponential_jitter=esperar_entre_intentos,
            stop_after_attempt=max_intentos,
        )

    async def responder(self, pregunta: str) -> RespuestaRAG:
        """Responde `pregunta` usando solo los documentos indexados.

        Raises:
            ValueError: la pregunta esta vacia (error de quien llama).
            RespuestaInvalidaError: tras agotar los reintentos, el modelo no cumplio el contrato.
            ProveedorError: falla de Gemini o Claude (clave, cuota, red).
        """
        if not pregunta or not pregunta.strip():
            raise ValueError("La pregunta esta vacia.")
        pregunta = pregunta.strip()

        handler = RagLogHandler()
        inicio = time.perf_counter()
        try:
            fragmentos = await self._recuperar.ainvoke(pregunta)  # a) similitud en ChromaDB
            fin_recuperacion = time.perf_counter()
            salida: RespuestaLLM = await self.chain.ainvoke(  # b) prompt + c) LLM async + d) parseo
                {"pregunta": pregunta, "documentos": fragmentos},
                config={"callbacks": [handler], "run_name": "rag_grounded"},
            )
        except RagError as exc:
            logger.error("RAG fallo tras %d intento(s) del modelo: %s", handler.intentos, exc)
            raise
        # OJO con el orden: OutputParserException y ValidationError HEREDAN de ValueError,
        # asi que deben ir ANTES del "except ValueError" o este se las traga.
        except (OutputParserException, ValidationError) as exc:  # el modelo nunca cumplio el contrato
            logger.error("El modelo no cumplio el contrato tras %d intento(s): %s", handler.intentos, exc)
            raise RespuestaInvalidaError(
                f"El modelo no devolvio una respuesta valida tras {handler.intentos} intento(s): {str(exc).splitlines()[0]}"
            ) from exc
        except ValueError:
            raise
        except Exception as exc:  # clave invalida, red caida tras los reintentos, cuota...
            logger.error("Error de proveedor: %s: %s", type(exc).__name__, exc)
            raise _traducir_error(exc) from exc
        fin = time.perf_counter()

        metricas = Metricas(
            recuperacion_s=round(fin_recuperacion - inicio, 3),
            generacion_s=round(fin - fin_recuperacion, 3),
            total_s=round(fin - inicio, 3),
            intentos=handler.intentos,
            tokens_entrada=handler.tokens_entrada,
            tokens_salida=handler.tokens_salida,
        )
        logger.info(
            "Consulta OK en %.2fs (recuperacion %.2fs, generacion %.2fs; intentos: %d; tokens: %d/%d).",
            metricas.total_s, metricas.recuperacion_s, metricas.generacion_s,
            metricas.intentos, metricas.tokens_entrada, metricas.tokens_salida,
        )
        return RespuestaRAG(
            pregunta=pregunta,
            respuesta=salida.respuesta,
            encontrada=salida.encontrada,
            fuentes=[f for f in fragmentos if f.referencia in salida.referencias],
            recuperados=fragmentos,
            modelo_llm=self.modelo_llm,
            modelo_embeddings=self.modelo_embeddings,
            metricas=metricas,
        )


@lru_cache(maxsize=1)
def get_default_sistema() -> SistemaRAG:
    """Sistema por defecto: Gemini (embeddings) + Claude (generacion), segun el .env. Se crea una sola vez."""
    from dotenv import load_dotenv

    load_dotenv(config.BASE_DIR / ".env")
    embeddings = crear_embeddings()
    recuperador = abrir_recuperador(embeddings, config.PERSIST_DIR, config.COLLECTION_NAME, int(os.getenv("RAG_TOP_K", config.TOP_K)))
    modelo = build_model(
        model=os.getenv("LLM_MODEL") or None,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", config.LLM_MAX_TOKENS)),
        timeout_s=float(os.getenv("LLM_TIMEOUT_S", config.LLM_TIMEOUT_S)),
    )
    return SistemaRAG(
        recuperador,
        build_chain(modelo, max_intentos=int(os.getenv("LLM_MAX_INTENTOS", config.MAX_INTENTOS))),
        modelo_llm=nombre_del_llm(modelo),
        modelo_embeddings=nombre_del_modelo(embeddings),
    )


# --------------------------------------------------------------------------- #
# Punto de entrada pedido por la consigna
# --------------------------------------------------------------------------- #
async def get_rag_response(query: str, sistema: SistemaRAG | None = None) -> RespuestaRAG:
    """Responde `query` con la informacion de los documentos y devuelve la respuesta con sus fuentes.

    Args:
        query: pregunta del usuario.
        sistema: sistema a usar; si no se pasa, se usa el configurado por el .env.
    """
    return await (sistema or get_default_sistema()).responder(query)
