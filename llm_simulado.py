"""LLM simulado: un chat model determinista para tests y para probar el sistema sin API key.

NO entiende nada: elige el fragmento que comparte mas palabras con la pregunta y devuelve la
oracion mas parecida. Sirve para comprobar la LOGICA del sistema (prompt, parseo, citas,
reintentos, metricas), igual que `modelo_simulado.py` en la pre-entrega 2. No sirve para medir
la calidad de las respuestas: eso se prueba con el modelo real.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from schemas import MENSAJE_NO_SE

_PALABRAS_VACIAS = {
    "cual", "cuales", "cuanto", "cuantos", "cuanta", "cuantas", "como", "donde", "quien", "quienes",
    "cuando", "tiene", "tienen", "tenia", "deben", "debe", "puede", "pueden", "sobre", "entre",
    "segun", "hacer", "hace", "esta", "estan", "este", "estos", "para", "desde", "hasta", "empresa",
    "compania", "documentos", "documentacion",
}
_MIN_PALABRAS_EN_COMUN = 2
_FRACCION_MINIMA_DE_LA_PREGUNTA = 0.6  # que parte de las palabras de la pregunta debe aparecer en el fragmento


def _palabras(texto: str) -> set[str]:
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    palabras = (p for p in re.findall(r"[a-z0-9]+", sin_tildes) if len(p) >= 5 and p not in _PALABRAS_VACIAS)
    return {p[:-1] if p.endswith("s") else p for p in palabras}  # "incidentes" y "incidente" cuentan igual


class LLMSimulado(BaseChatModel):
    """Responde con el JSON que espera `RespuestaLLM`, usando solo coincidencia de palabras."""

    llamadas: int = 0  # cuantas veces se lo llamo (los tests lo usan)

    @property
    def _llm_type(self) -> str:
        return "simulado"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.llamadas += 1
        entrada = str(messages[-1].content)
        contenido = self._responder(entrada)
        usage = {
            "input_tokens": len(entrada.split()),
            "output_tokens": len(contenido.split()),
            "total_tokens": len(entrada.split()) + len(contenido.split()),
        }
        mensaje = AIMessage(content=contenido, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    @staticmethod
    def _responder(entrada: str) -> str:
        pregunta = entrada.split("PREGUNTA:", 1)[-1]
        contexto = entrada.split("CONTEXTO:", 1)[-1].split("PREGUNTA:", 1)[0]
        fragmentos = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\n\[\d+\]|\Z)", contexto, flags=re.S)

        palabras_pregunta = _palabras(pregunta)
        mejor_numero, mejor_texto, mejor_comunes = 0, "", 0
        for numero, texto in fragmentos:
            comunes = len(palabras_pregunta & _palabras(texto))
            if comunes > mejor_comunes:
                mejor_numero, mejor_texto, mejor_comunes = int(numero), texto, comunes

        fraccion = mejor_comunes / len(palabras_pregunta) if palabras_pregunta else 0.0
        if mejor_comunes < _MIN_PALABRAS_EN_COMUN or fraccion < _FRACCION_MINIMA_DE_LA_PREGUNTA:
            return json.dumps(
                {"respuesta": MENSAJE_NO_SE, "encontrada": False, "referencias": []}, ensure_ascii=False
            )

        oraciones = [o.strip() for o in re.split(r"(?<=[.!?])\s+|\n+", mejor_texto) if len(o.strip()) > 20]
        oracion = max(oraciones, key=lambda o: len(palabras_pregunta & _palabras(o)), default=mejor_texto)
        return json.dumps(
            {"respuesta": re.sub(r"[*`#]+", "", oracion).lstrip("- ").strip(), "encontrada": True, "referencias": [mejor_numero]},
            ensure_ascii=False,
        )
