"""Tests de main.py (CLI de prueba). Sin API keys ni red."""
import json

import pytest

import main
import rag_chain
from main import PRUEBAS, Prueba, evaluar
from schemas import MENSAJE_NO_SE, FragmentoRecuperado, RespuestaRAG


def frag(archivo: str, ref: int = 1) -> FragmentoRecuperado:
    return FragmentoRecuperado(referencia=ref, archivo=archivo, seccion="s", chunk_id="c", distancia=0.1, texto="t")


def rag(respuesta: str, encontrada: bool = True, fuentes: list[FragmentoRecuperado] | None = None) -> RespuestaRAG:
    return RespuestaRAG(
        pregunta="p", respuesta=respuesta, encontrada=encontrada, fuentes=fuentes or [],
        recuperados=fuentes or [], modelo_llm="m", modelo_embeddings="e",
    )


CON_RESPUESTA = Prueba("p", esperados=(r"4 horas",), archivos=frozenset({"01.md", "02.md"}))
TRAMPA = Prueba("p", es_trampa=True)


# ------------------------------------------------------------------ el set de pruebas
def test_hay_preguntas_con_respuesta_y_preguntas_trampa():
    """La consigna pide 2 tests: una pregunta que SI se responde y una trampa que NO."""
    assert sum(not p.es_trampa for p in PRUEBAS) >= 1
    assert sum(p.es_trampa for p in PRUEBAS) >= 1
    assert all(p.esperados and p.archivos for p in PRUEBAS if not p.es_trampa)


# ------------------------------------------------------------------ evaluar (el "juez" de las pruebas)
def test_respuesta_correcta_con_fuente_esperada_aprueba():
    ok, _ = evaluar(CON_RESPUESTA, rag("El plazo es de 4 horas hábiles.", True, [frag("01.md")]))
    assert ok


def test_dice_no_lo_se_cuando_la_respuesta_si_esta_falla():
    ok, motivo = evaluar(CON_RESPUESTA, rag(MENSAJE_NO_SE, False))
    assert not ok and "sí está en los documentos" in motivo


def test_respuesta_sin_el_dato_esperado_falla():
    ok, motivo = evaluar(CON_RESPUESTA, rag("Depende del caso.", True, [frag("01.md")]))
    assert not ok and "no contiene" in motivo


def test_respuesta_con_fuente_equivocada_falla():
    ok, motivo = evaluar(CON_RESPUESTA, rag("Son 4 horas.", True, [frag("04.md")]))
    assert not ok and "no citó ninguno" in motivo


def test_el_dato_esperado_se_busca_sin_distinguir_mayusculas():
    ok, _ = evaluar(CON_RESPUESTA, rag("SON 4 HORAS.", True, [frag("02.md")]))
    assert ok


def test_trampa_bien_respondida_aprueba():
    ok, _ = evaluar(TRAMPA, rag(MENSAJE_NO_SE, False))
    assert ok


@pytest.mark.parametrize(
    "r",
    [
        rag("El presupuesto es de 1 millón.", True, [frag("01.md")]),   # alucinó y citó
        rag("El presupuesto es de 1 millón.", True),                     # alucinó sin citar
        rag(MENSAJE_NO_SE, False, [frag("01.md")]),                      # dice no sé pero cita una fuente
    ],
    ids=["alucina-y-cita", "alucina", "no-se-con-fuente"],
)
def test_trampa_donde_el_sistema_inventa_falla(r):
    ok, _ = evaluar(TRAMPA, r)
    assert not ok


# ------------------------------------------------------------------ CLI de punta a punta (modo simulado)
def test_pruebas_en_modo_simulado_corren_todo_el_flujo(capsys):
    codigo = main.main(["--pruebas", "--simular"])
    salida = capsys.readouterr().out
    assert codigo == 0
    assert "PREGUNTAS CON RESPUESTA" in salida and "PREGUNTAS TRAMPA" in salida
    assert "MODO SIMULADO" in salida               # deja claro que no se midió la calidad
    assert f"{len(PRUEBAS)} consultas una tras otra" in salida


def test_pruebas_concurrentes_en_modo_simulado(capsys):
    assert main.main(["--pruebas", "--simular", "--concurrente"]) == 0
    assert "en paralelo" in capsys.readouterr().out


def test_una_pregunta_muestra_respuesta_fuentes_y_metricas(capsys):
    assert main.main(["¿Cuál es el plazo máximo para resolver un incidente crítico?", "--simular"]) == 0
    salida = capsys.readouterr().out
    for esperado in ("PREGUNTA :", "RESPUESTA:", "Fuentes citadas", "Recuperados (top_k=4)", "Métricas:"):
        assert esperado in salida


def test_json_es_una_respuesta_rag_valida(capsys):
    assert main.main(["¿Cuántos años se conservan los datos transaccionales de ventas?", "--simular", "--json"]) == 0
    r = RespuestaRAG.model_validate(json.loads(capsys.readouterr().out))
    assert len(r.recuperados) == 4 and r.metricas.intentos == 1


def test_sin_pregunta_ni_pruebas_pide_una_y_sale_con_codigo_2(capsys):
    assert main.main(["--simular"]) == 2
    assert "Escribí una pregunta" in capsys.readouterr().err


def test_sin_claves_el_error_es_claro_y_no_un_traceback(monkeypatch, capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)   # no leer el .env real
    rag_chain.get_default_sistema.cache_clear()
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    try:
        assert main.main(["hola"]) == 1
    finally:
        rag_chain.get_default_sistema.cache_clear()
    err = capsys.readouterr().err
    assert err.startswith("ERROR:") and "GOOGLE_API_KEY" in err and "Traceback" not in err
