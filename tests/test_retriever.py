"""Tests del retriever (retriever.py). Sin API key: embeddings simulados + la base de verdad de /data."""
import asyncio
from pathlib import Path

import pytest

import config
import ingest
from embeddings import EmbeddingsSimulados
from errores import BaseVectorialVaciaError, EmbeddingsNoCoincidenError
from retriever import Recuperador, abrir_recuperador


@pytest.fixture
def base(tmp_path: Path):
    """Indexa los 4 documentos reales de /data con embeddings simulados (contador sin red)."""
    embeddings = EmbeddingsSimulados()
    carpeta = tmp_path / "vs"
    ingest.ingestar(config.DATA_DIR, carpeta, embeddings, contador=ingest.contar_tokens_aproximado)
    return embeddings, carpeta


# ------------------------------------------------------------------ k
@pytest.mark.parametrize("k", [3, 4, 5])
def test_k_dentro_del_rango_pedido_es_valido(base, k):
    embeddings, carpeta = base
    assert abrir_recuperador(embeddings, carpeta, k=k).k == k


@pytest.mark.parametrize("k", [0, 1, 2, 6, 10])
def test_k_fuera_del_rango_3_a_5_se_rechaza(base, k):
    embeddings, carpeta = base
    with pytest.raises(ValueError, match="entre 3 y 5"):
        abrir_recuperador(embeddings, carpeta, k=k)


# ------------------------------------------------------------------ recuperar
async def test_devuelve_k_fragmentos_numerados_y_ordenados_por_cercania(base):
    embeddings, carpeta = base
    r = abrir_recuperador(embeddings, carpeta, k=4)
    frags = await r.recuperar("¿Cuál es el plazo máximo para resolver un incidente crítico?")
    assert len(frags) == 4
    assert [f.referencia for f in frags] == [1, 2, 3, 4]
    distancias = [f.distancia for f in frags]
    assert distancias == sorted(distancias)
    assert all(d >= 0 for d in distancias)


async def test_cada_fragmento_trae_lo_necesario_para_citarlo(base):
    embeddings, carpeta = base
    frags = await abrir_recuperador(embeddings, carpeta).recuperar("incidente crítico plazo")
    for f in frags:
        assert f.archivo.endswith((".md", ".txt"))
        assert f.chunk_id and f.texto.strip()


@pytest.mark.parametrize(
    "pregunta, archivos_esperados",
    [
        ("¿Cuál es el plazo máximo para resolver un incidente crítico?", {"01-politica-calidad-de-datos.md", "02-roles-y-responsabilidades.md"}),
        ("¿Quién debe aprobar el alta de un artículo de marca propia?", {"03-procedimiento-maestro-de-articulos.md", "02-roles-y-responsabilidades.md"}),
        ("¿Cuántos años se conservan los datos transaccionales de ventas?", {"04-glosario-catalogo-y-clasificacion.md"}),
        ("¿Cuántos miembros hacen falta para que sesione el Comité de Gobierno de Datos?", {"02-roles-y-responsabilidades.md"}),
    ],
)
async def test_las_preguntas_con_respuesta_traen_el_documento_correcto(base, pregunta, archivos_esperados):
    """Evidencia de recuperacion: el documento que contiene la respuesta esta entre los top_k."""
    embeddings, carpeta = base
    frags = await abrir_recuperador(embeddings, carpeta, k=5).recuperar(pregunta)
    assert {f.archivo for f in frags} & archivos_esperados


async def test_usa_el_mismo_modelo_de_embeddings_para_consultar(base):
    embeddings, carpeta = base
    antes = embeddings.llamadas_consultas
    await abrir_recuperador(embeddings, carpeta).recuperar("política de calidad")
    assert embeddings.llamadas_consultas == antes + 1


@pytest.mark.parametrize("pregunta", ["", "   ", "\n"])
async def test_pregunta_vacia_se_rechaza_sin_llamar_al_proveedor(base, pregunta):
    embeddings, carpeta = base
    antes = embeddings.llamadas_consultas
    with pytest.raises(ValueError, match="vacia"):
        await abrir_recuperador(embeddings, carpeta).recuperar(pregunta)
    assert embeddings.llamadas_consultas == antes


async def test_consultas_en_paralelo_dan_lo_mismo_que_una_por_una(base):
    embeddings, carpeta = base
    r = abrir_recuperador(embeddings, carpeta)
    preguntas = ["incidente crítico", "marca propia", "datos de ventas", "quórum del comité"]
    en_serie = [await r.recuperar(p) for p in preguntas]
    en_paralelo = await asyncio.gather(*(r.recuperar(p) for p in preguntas))
    assert [[f.chunk_id for f in fs] for fs in en_serie] == [[f.chunk_id for f in fs] for fs in en_paralelo]


# ------------------------------------------------------------------ abrir_recuperador
def test_base_inexistente_pide_indexar_primero(tmp_path: Path):
    with pytest.raises(BaseVectorialVaciaError, match="python ingest.py"):
        abrir_recuperador(EmbeddingsSimulados(), tmp_path / "nada")


def test_modelo_de_embeddings_distinto_se_detecta(base):
    """Error #1 de la consigna: consultar con otro modelo que el de la indexacion devuelve ruido."""
    _, carpeta = base

    class OtroModelo(EmbeddingsSimulados):
        nombre_modelo = "otro-modelo"

    with pytest.raises(EmbeddingsNoCoincidenError):
        abrir_recuperador(OtroModelo(), carpeta)


def test_constructor_directo_tambien_valida_k(base):
    embeddings, carpeta = base
    db = ingest.abrir_coleccion(embeddings, carpeta, config.COLLECTION_NAME)
    with pytest.raises(ValueError):
        Recuperador(db, embeddings, k=9)
