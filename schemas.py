"""Modelos Pydantic del sistema RAG (contratos de datos)."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class ReporteIngesta(BaseModel):
    """Resultado de una corrida de ingesta: sirve como evidencia en el README y en los tests."""

    archivos: int = Field(ge=0, description="Archivos leidos de /data")
    fragmentos: int = Field(ge=0, description="Fragmentos vigentes (chunks) generados")
    nuevos: int = Field(ge=0, description="Fragmentos embebidos y guardados en esta corrida")
    sin_cambios: int = Field(ge=0, description="Fragmentos que ya estaban en la base (no se re-embebieron)")
    eliminados: int = Field(ge=0, description="Fragmentos obsoletos borrados (archivo cambiado o quitado)")
    tokens_min: int = Field(ge=0)
    tokens_max: int = Field(ge=0)
    tokens_promedio: float = Field(ge=0)
    medicion_tokens: str = Field(description="Como se midieron los tokens (tiktoken o aproximacion)")
    modelo_embeddings: str
    coleccion: str
    persist_dir: str

    @property
    def base_al_dia(self) -> bool:
        """True si no hubo que escribir nada: la base ya estaba actualizada."""
        return self.nuevos == 0 and self.eliminados == 0


# =========================================================================== #
# Respuesta del RAG
# =========================================================================== #
MENSAJE_NO_SE = "No lo sé: esa información no está en los documentos disponibles."


def _empieza_con_no_lo_se(texto: str) -> bool:
    normalizado = texto.strip().lower().replace("é", "e")
    return normalizado.startswith("no lo se")


class RespuestaLLM(BaseModel):
    """Contrato que debe cumplir lo que devuelve el modelo (lo parsea PydanticOutputParser).

    Ojo: el modelo NO escribe nombres de archivo: solo cita los NUMEROS [n] de los
    fragmentos que recibio. Los archivos y secciones los agrega el codigo a partir de esos
    numeros, asi el modelo no puede inventar una fuente.
    """

    respuesta: str = Field(
        min_length=1,
        description="Respuesta en español, basada SOLO en el contexto. Si la información no está "
        "en el contexto, debe empezar con 'No lo sé'.",
    )
    encontrada: bool = Field(
        description="true si la respuesta sale del contexto; false si la información no está en los documentos.",
    )
    referencias: list[int] = Field(
        default_factory=list,
        description="Números [n] de los fragmentos del contexto en los que se apoya la respuesta. "
        "Lista vacía si encontrada es false.",
    )

    @field_validator("referencias")
    @classmethod
    def _referencias_validas(cls, valor: list[int]) -> list[int]:
        if any(n < 1 for n in valor):
            raise ValueError("las referencias son numeros de fragmento y empiezan en 1")
        return list(dict.fromkeys(valor))  # sin repetidos, conservando el orden

    @model_validator(mode="after")
    def _coherencia(self) -> "RespuestaLLM":
        if self.encontrada and not self.referencias:
            raise ValueError("si encontrada es true, 'referencias' debe citar al menos un fragmento")
        if not self.encontrada:
            if self.referencias:
                raise ValueError("si encontrada es false, 'referencias' debe estar vacia")
            if not _empieza_con_no_lo_se(self.respuesta):
                raise ValueError("si encontrada es false, la respuesta debe empezar con 'No lo sé'")
        return self


class FragmentoRecuperado(BaseModel):
    """Un fragmento devuelto por el retriever, con todo lo necesario para citarlo."""

    referencia: int = Field(ge=1, description="Numero [n] con el que se le presenta al modelo")
    archivo: str
    seccion: str = ""
    chunk_id: str
    distancia: float = Field(ge=0, description="Distancia coseno: 0 = identico, mayor = mas lejos de la pregunta")
    texto: str


class Metricas(BaseModel):
    """Tiempos y consumo de una consulta (observabilidad basica)."""

    recuperacion_s: float = Field(ge=0)
    generacion_s: float = Field(ge=0)
    total_s: float = Field(ge=0)
    intentos: int = Field(ge=0, description="Llamadas al modelo (1 = salio bien a la primera)")
    tokens_entrada: int = Field(ge=0)
    tokens_salida: int = Field(ge=0)


class RespuestaRAG(BaseModel):
    """Lo que devuelve `get_rag_response`: la respuesta MAS las fuentes que la respaldan."""

    pregunta: str
    respuesta: str
    encontrada: bool
    fuentes: list[FragmentoRecuperado] = Field(
        default_factory=list, description="Solo los fragmentos que el modelo cito"
    )
    recuperados: list[FragmentoRecuperado] = Field(
        default_factory=list, description="Todos los fragmentos que trajo el retriever (top_k)"
    )
    modelo_llm: str
    modelo_embeddings: str
    metricas: Metricas | None = None
