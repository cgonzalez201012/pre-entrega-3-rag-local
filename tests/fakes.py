"""Piezas falsas para probar el RAG sin API keys ni red.

- `LLMGuion`: un chat model que responde lo que le indiquemos, en orden (o lanza la excepcion
  que le indiquemos). Sirve para forzar escenarios que un modelo real casi nunca produce:
  JSON roto, una cita a un fragmento que no existe, un error de red, etc.
- `EmbeddingsConFallas`: embeddings simulados que fallan las primeras N veces.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from embeddings import EmbeddingsSimulados

TOKENS_ENTRADA = 100
TOKENS_SALIDA = 20


def json_respuesta(respuesta: str, encontrada: bool = True, referencias: list[int] | None = None) -> str:
    """El JSON que debe devolver el modelo para cumplir el contrato `RespuestaLLM`."""
    return json.dumps(
        {"respuesta": respuesta, "encontrada": encontrada, "referencias": [1] if referencias is None else referencias},
        ensure_ascii=False,
    )


class LLMGuion(BaseChatModel):
    """Responde segun un guion: cada elemento es un texto (respuesta) o una excepcion (se lanza).

    Cuando el guion se acaba, repite el ultimo elemento.
    """

    guion: list[Any]
    llamadas: int = 0
    mensajes_recibidos: list[Any] = []
    pausa_s: float = 0.0          # espera simulada de red
    esperar_n: int = 0            # si > 0, cada llamada espera a que haya n llamadas A LA VEZ (si no llegan: TimeoutError)
    en_vuelo: int = 0             # cuantas llamadas estan en curso en este instante
    max_en_vuelo: int = 0         # el maximo que hubo a la vez
    _evento: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "guion"

    def _siguiente(self, messages: list[BaseMessage]) -> ChatResult:
        elemento = self.guion[min(self.llamadas, len(self.guion) - 1)]
        self.llamadas += 1
        self.mensajes_recibidos.append(messages)
        if isinstance(elemento, BaseException):
            raise elemento
        mensaje = AIMessage(
            content=elemento,
            usage_metadata={
                "input_tokens": TOKENS_ENTRADA,
                "output_tokens": TOKENS_SALIDA,
                "total_tokens": TOKENS_ENTRADA + TOKENS_SALIDA,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> ChatResult:
        return self._siguiente(messages)

    async def _agenerate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> ChatResult:
        self.en_vuelo += 1
        self.max_en_vuelo = max(self.max_en_vuelo, self.en_vuelo)
        try:
            if self.esperar_n:
                # Barrera: solo se sigue cuando hay `esperar_n` llamadas en vuelo al mismo tiempo.
                # Con consultas en serie nunca se alcanza, y wait_for lanza TimeoutError: no es un test de tiempos.
                if self._evento is None:
                    self._evento = asyncio.Event()
                if self.en_vuelo >= self.esperar_n:
                    self._evento.set()
                await asyncio.wait_for(self._evento.wait(), timeout=5)
            if self.pausa_s:
                await asyncio.sleep(self.pausa_s)
            return self._siguiente(messages)
        finally:
            self.en_vuelo -= 1


class EmbeddingsConFallas(EmbeddingsSimulados):
    """Embeddings simulados cuya consulta lanza `error` las primeras `fallas` veces."""

    def __init__(self, error: BaseException, fallas: int) -> None:
        super().__init__()
        self._error = error
        self._fallas = fallas
        self.intentos_consulta = 0

    async def aembed_query(self, text: str) -> list[float]:
        self.intentos_consulta += 1
        if self.intentos_consulta <= self._fallas:
            raise self._error
        return await super().aembed_query(text)
