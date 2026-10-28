"""Tests de los contratos Pydantic (schemas.py). Sin red ni API keys."""
import pytest
from pydantic import ValidationError

from schemas import (
    MENSAJE_NO_SE,
    FragmentoRecuperado,
    Metricas,
    RespuestaLLM,
    RespuestaRAG,
)


def fragmento(**kw) -> FragmentoRecuperado:
    base = dict(referencia=1, archivo="a.md", seccion="1. Alcance", chunk_id="abc", distancia=0.2, texto="texto")
    return FragmentoRecuperado(**{**base, **kw})


# ------------------------------------------------------------------ RespuestaLLM
def test_respuesta_con_fuente_es_valida():
    r = RespuestaLLM(respuesta="Son 4 horas hábiles.", encontrada=True, referencias=[1, 3])
    assert r.referencias == [1, 3]


def test_referencias_repetidas_se_eliminan_conservando_el_orden():
    r = RespuestaLLM(respuesta="ok", encontrada=True, referencias=[2, 1, 2, 1])
    assert r.referencias == [2, 1]


def test_encontrada_sin_referencias_es_invalida():
    """Una respuesta sin fuente no es una respuesta 'grounded'."""
    with pytest.raises(ValidationError, match="al menos un fragmento"):
        RespuestaLLM(respuesta="Son 4 horas.", encontrada=True, referencias=[])


def test_no_encontrada_con_referencias_es_invalida():
    with pytest.raises(ValidationError, match="debe estar vacia"):
        RespuestaLLM(respuesta=MENSAJE_NO_SE, encontrada=False, referencias=[1])


def test_no_encontrada_debe_decir_no_lo_se():
    """Si dice 'no encontré' pero responde igual, el modelo se contradice: se rechaza."""
    with pytest.raises(ValidationError, match="No lo sé"):
        RespuestaLLM(respuesta="Probablemente unos 5 días.", encontrada=False, referencias=[])


@pytest.mark.parametrize("texto", [MENSAJE_NO_SE, "No lo se.", "  no lo sé, no figura", "NO LO SÉ"])
def test_no_lo_se_acepta_variantes_de_tilde_y_mayusculas(texto):
    assert RespuestaLLM(respuesta=texto, encontrada=False, referencias=[]).encontrada is False


@pytest.mark.parametrize("referencias", [[0], [-1], [1, 0]])
def test_las_referencias_empiezan_en_1(referencias):
    with pytest.raises(ValidationError, match="empiezan en 1"):
        RespuestaLLM(respuesta="ok", encontrada=True, referencias=referencias)


def test_respuesta_vacia_es_invalida():
    with pytest.raises(ValidationError):
        RespuestaLLM(respuesta="", encontrada=True, referencias=[1])


def test_el_modelo_no_puede_inventar_nombres_de_archivo():
    """El contrato solo admite NUMEROS de fragmento: un nombre de archivo no valida."""
    with pytest.raises(ValidationError):
        RespuestaLLM(respuesta="ok", encontrada=True, referencias=["politica.md"])


# ------------------------------------------------------------------ resto de los modelos
def test_distancia_negativa_es_invalida():
    with pytest.raises(ValidationError):
        fragmento(distancia=-0.1)


def test_referencia_de_fragmento_empieza_en_1():
    with pytest.raises(ValidationError):
        fragmento(referencia=0)


def test_metricas_no_admiten_negativos():
    with pytest.raises(ValidationError):
        Metricas(recuperacion_s=-1, generacion_s=0, total_s=0, intentos=1, tokens_entrada=0, tokens_salida=0)


def test_respuesta_rag_ida_y_vuelta_json():
    original = RespuestaRAG(
        pregunta="¿Plazo?",
        respuesta="4 horas.",
        encontrada=True,
        fuentes=[fragmento()],
        recuperados=[fragmento(), fragmento(referencia=2, archivo="b.md")],
        modelo_llm="claude-haiku-4-5",
        modelo_embeddings="gemini-embedding-001",
        metricas=Metricas(recuperacion_s=0.1, generacion_s=0.9, total_s=1.0, intentos=1, tokens_entrada=500, tokens_salida=40),
    )
    assert RespuestaRAG.model_validate_json(original.model_dump_json()) == original
