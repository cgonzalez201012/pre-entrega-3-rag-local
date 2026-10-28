"""Tests de embeddings.py. Corren sin API key y sin red."""
import math

import pytest

from embeddings import EmbeddingsSimulados, crear_embeddings, nombre_del_modelo
from errores import ConfiguracionError


def coseno(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # los vectores simulados ya estan normalizados


def test_simulados_son_deterministas():
    e1, e2 = EmbeddingsSimulados(), EmbeddingsSimulados()
    assert e1.embed_query("incidente critico") == e2.embed_query("incidente critico")


def test_simulados_estan_normalizados():
    v = EmbeddingsSimulados().embed_query("plazo de resolucion de incidentes")
    assert math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-9)


def test_simulados_ignoran_tildes_y_mayusculas():
    e = EmbeddingsSimulados()
    assert e.embed_query("Política de Calidad") == e.embed_query("politica de calidad")


def test_textos_parecidos_quedan_mas_cerca_que_textos_distintos():
    e = EmbeddingsSimulados()
    consulta = e.embed_query("plazo para resolver un incidente critico")
    relacionado = e.embed_documents(["Incidente critico: plazo de resolucion de 4 horas habiles"])[0]
    ajeno = e.embed_documents(["Los respaldos se conservan durante 35 dias en el centro de datos"])[0]
    assert coseno(consulta, relacionado) > coseno(consulta, ajeno)


def test_simulados_cuentan_las_llamadas():
    e = EmbeddingsSimulados()
    e.embed_documents(["a", "b", "c"])
    e.embed_query("q")
    assert (e.llamadas_documentos, e.llamadas_consultas) == (3, 1)


def test_crear_embeddings_simulado_no_necesita_clave(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(crear_embeddings(simulado=True), EmbeddingsSimulados)


def test_sin_clave_falla_con_un_mensaje_que_dice_como_arreglarlo(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfiguracionError) as info:
        crear_embeddings()
    assert "GOOGLE_API_KEY" in str(info.value)
    assert "aistudio.google.com" in str(info.value)


def test_con_clave_crea_el_cliente_de_gemini_sin_llamar_a_la_red(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-falsa-de-prueba")
    emb = crear_embeddings()
    assert emb.model == "gemini-embedding-001"
    assert nombre_del_modelo(emb) == "gemini-embedding-001"


def test_nombre_del_modelo_de_los_simulados():
    assert nombre_del_modelo(EmbeddingsSimulados()) == "simulado-bolsa-de-palabras"
