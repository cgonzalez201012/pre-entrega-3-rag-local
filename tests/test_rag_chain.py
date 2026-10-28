"""Tests de la cadena RAG (rag_chain.py). Sin API keys ni red: LLM de guion + embeddings simulados."""
import asyncio
from pathlib import Path

import anthropic
import httpx
import pytest
from google.genai import errors as gerr
from langchain_core.messages import HumanMessage, SystemMessage

import config
import ingest
import rag_chain
from embeddings import EmbeddingsSimulados
from errores import ConfiguracionError, ProveedorError, RespuestaInvalidaError
from rag_chain import SistemaRAG, build_chain, build_model, formatear_contexto, get_rag_response
from retriever import abrir_recuperador
from schemas import MENSAJE_NO_SE, FragmentoRecuperado
from fakes import TOKENS_ENTRADA, TOKENS_SALIDA, EmbeddingsConFallas, LLMGuion, json_respuesta

PREGUNTA = "¿Cuál es el plazo máximo para resolver un incidente crítico?"
OK = json_respuesta("El plazo máximo es de 4 horas hábiles.", True, [1])


@pytest.fixture(scope="module")
def carpeta_base(tmp_path_factory) -> Path:
    """La base se indexa una sola vez para todo el modulo (los tests solo la leen)."""
    carpeta = tmp_path_factory.mktemp("vs")
    ingest.ingestar(config.DATA_DIR, carpeta, EmbeddingsSimulados(), contador=ingest.contar_tokens_aproximado)
    return carpeta


def armar(carpeta: Path, guion: list, *, embeddings=None, k: int = 4, max_intentos: int = 3, **kw_llm):
    """Devuelve (sistema, llm) con todo el cableado real y solo el LLM / embeddings falsos."""
    embeddings = embeddings or EmbeddingsSimulados()
    llm = LLMGuion(guion=guion, **kw_llm)
    sistema = SistemaRAG(
        abrir_recuperador(embeddings, carpeta, k=k),
        build_chain(llm, max_intentos=max_intentos, esperar_entre_intentos=False),
        modelo_llm="guion",
        modelo_embeddings="simulado",
        max_intentos=max_intentos,
        esperar_entre_intentos=False,
    )
    return sistema, llm


def error_conexion() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))


def error_http(clase, estado: int):
    respuesta = httpx.Response(estado, request=httpx.Request("POST", "https://api.anthropic.com"))
    return clase(message=f"http {estado}", response=respuesta, body=None)


# ------------------------------------------------------------------ contexto y prompt
def test_formatear_contexto_numera_los_fragmentos():
    frags = [
        FragmentoRecuperado(referencia=1, archivo="a.md", seccion="1. Alcance", chunk_id="x", distancia=0.1, texto="uno"),
        FragmentoRecuperado(referencia=2, archivo="b.md", seccion="", chunk_id="y", distancia=0.2, texto="dos"),
    ]
    assert formatear_contexto(frags) == "[1] a.md · 1. Alcance\nuno\n\n[2] b.md\ndos"


async def test_el_prompt_lleva_la_regla_no_lo_se_y_el_contexto_numerado(carpeta_base):
    sistema, llm = armar(carpeta_base, [OK])
    r = await sistema.responder(PREGUNTA)

    mensajes = llm.mensajes_recibidos[0]
    sistema_msg = next(m for m in mensajes if isinstance(m, SystemMessage)).content
    humano_msg = next(m for m in mensajes if isinstance(m, HumanMessage)).content

    assert MENSAJE_NO_SE in sistema_msg and "SOLO" in sistema_msg      # filtro de veracidad
    assert "referencias" in sistema_msg                                  # instrucciones de formato (parser)
    assert f"PREGUNTA: {PREGUNTA}" in humano_msg
    for f in r.recuperados:                                              # todo lo recuperado llega al modelo, numerado
        assert f"[{f.referencia}] {f.archivo}" in humano_msg
        assert f.texto in humano_msg


# ------------------------------------------------------------------ camino feliz
async def test_respuesta_con_fuentes_y_metricas(carpeta_base):
    sistema, llm = armar(carpeta_base, [json_respuesta("4 horas hábiles.", True, [2])])
    r = await sistema.responder(f"  {PREGUNTA}  ")

    assert r.pregunta == PREGUNTA                       # se normaliza
    assert r.respuesta == "4 horas hábiles." and r.encontrada
    assert len(r.recuperados) == 4
    assert [f.referencia for f in r.fuentes] == [2]     # fuentes = solo lo que el modelo citó
    assert r.fuentes[0] == r.recuperados[1]
    assert r.modelo_llm == "guion" and r.modelo_embeddings == "simulado"
    m = r.metricas
    assert m.intentos == 1 and llm.llamadas == 1
    assert (m.tokens_entrada, m.tokens_salida) == (TOKENS_ENTRADA, TOKENS_SALIDA)
    assert m.total_s >= m.recuperacion_s >= 0 and m.generacion_s >= 0


async def test_varias_fuentes_se_conservan_en_el_orden_del_retriever(carpeta_base):
    sistema, _ = armar(carpeta_base, [json_respuesta("ok", True, [3, 1])])
    r = await sistema.responder(PREGUNTA)
    assert [f.referencia for f in r.fuentes] == [1, 3]


async def test_get_rag_response_usa_el_sistema_indicado(carpeta_base):
    sistema, llm = armar(carpeta_base, [OK])
    r = await get_rag_response(PREGUNTA, sistema)
    assert r.encontrada and llm.llamadas == 1


# ------------------------------------------------------------------ "No lo sé"
async def test_pregunta_sin_respuesta_en_los_documentos(carpeta_base):
    sistema, _ = armar(carpeta_base, [json_respuesta(MENSAJE_NO_SE, False, [])])
    r = await sistema.responder("¿Cuál es el presupuesto anual del área?")
    assert r.encontrada is False
    assert r.respuesta.startswith("No lo sé")
    assert r.fuentes == []                 # no cita fuentes para una respuesta que no existe
    assert len(r.recuperados) == 4         # pero queda registro de qué se miró


async def test_un_no_lo_se_incoherente_se_reintenta(carpeta_base):
    """Dice 'no encontré' pero igual responde con datos y cita: se rechaza y se pide de nuevo."""
    incoherente = json_respuesta("Son 10 días.", False, [1])
    sistema, llm = armar(carpeta_base, [incoherente, json_respuesta(MENSAJE_NO_SE, False, [])])
    r = await sistema.responder("¿Cuál es el presupuesto anual?")
    assert llm.llamadas == 2 and r.encontrada is False and r.metricas.intentos == 2


# ------------------------------------------------------------------ reintentos
@pytest.mark.parametrize(
    "mala",
    [
        "esto no es JSON",
        '{"respuesta": "falta lo demas"}',
        json_respuesta("sin fuente", True, []),
        json_respuesta("cita un fragmento que no existe", True, [9]),
    ],
    ids=["no-json", "campos-faltantes", "sin-referencias", "cita-inexistente"],
)
async def test_respuesta_invalida_se_reintenta_y_se_cuenta(carpeta_base, mala):
    sistema, llm = armar(carpeta_base, [mala, OK])
    r = await sistema.responder(PREGUNTA)
    assert r.respuesta.startswith("El plazo") and llm.llamadas == 2
    assert r.metricas.intentos == 2
    # cada reintento se paga: los tokens suman TODOS los intentos
    assert r.metricas.tokens_entrada == 2 * TOKENS_ENTRADA and r.metricas.tokens_salida == 2 * TOKENS_SALIDA


async def test_la_cita_a_un_fragmento_inexistente_nunca_llega_al_usuario(carpeta_base):
    """Caso [9] con 4 fragmentos: Pydantic valida la forma, el codigo valida la verdad."""
    sistema, llm = armar(carpeta_base, [json_respuesta("inventada", True, [9])], max_intentos=3)
    with pytest.raises(RespuestaInvalidaError, match=r"\[9\].*solo recibio 4"):
        await sistema.responder(PREGUNTA)
    assert llm.llamadas == 3


@pytest.mark.parametrize("max_intentos", [1, 2, 4])
async def test_se_respeta_el_maximo_de_intentos(carpeta_base, max_intentos):
    sistema, llm = armar(carpeta_base, ["basura"], max_intentos=max_intentos)
    with pytest.raises(RespuestaInvalidaError, match=f"tras {max_intentos} intento"):
        await sistema.responder(PREGUNTA)
    assert llm.llamadas == max_intentos


async def test_error_de_red_transitorio_se_reintenta(carpeta_base):
    sistema, llm = armar(carpeta_base, [error_conexion(), error_http(anthropic.RateLimitError, 429), OK])
    r = await sistema.responder(PREGUNTA)
    assert llm.llamadas == 3 and r.metricas.intentos == 3 and r.encontrada


async def test_red_caida_de_forma_persistente_da_error_claro(carpeta_base):
    sistema, llm = armar(carpeta_base, [error_conexion()])
    with pytest.raises(ProveedorError, match="APIConnectionError"):
        await sistema.responder(PREGUNTA)
    assert llm.llamadas == 3


@pytest.mark.parametrize("clase, estado", [(anthropic.AuthenticationError, 401), (anthropic.PermissionDeniedError, 403)])
async def test_clave_invalida_no_se_reintenta(carpeta_base, clase, estado):
    """Reintentar con una clave mala solo gasta tiempo: falla de inmediato y dice qué revisar."""
    sistema, llm = armar(carpeta_base, [error_http(clase, estado)])
    with pytest.raises(ProveedorError, match="ANTHROPIC_API_KEY"):
        await sistema.responder(PREGUNTA)
    assert llm.llamadas == 1


# ------------------------------------------------------------------ errores de Gemini (embeddings)
def gemini_error(clase, codigo: int):
    return clase(codigo, {"error": {"message": "x", "status": "ESTADO"}})


async def test_cuota_gratuita_de_gemini_da_mensaje_claro_y_no_se_reintenta(carpeta_base):
    emb = EmbeddingsConFallas(gemini_error(gerr.ClientError, 429), fallas=99)
    sistema, llm = armar(carpeta_base, [OK], embeddings=emb)
    with pytest.raises(ProveedorError, match="limite de uso del plan gratuito"):
        await sistema.responder(PREGUNTA)
    assert emb.intentos_consulta == 1 and llm.llamadas == 0     # ni siquiera se llamo a Claude


async def test_clave_de_gemini_rechazada_apunta_al_env(carpeta_base):
    emb = EmbeddingsConFallas(gemini_error(gerr.ClientError, 400), fallas=99)
    sistema, _ = armar(carpeta_base, [OK], embeddings=emb)
    with pytest.raises(ProveedorError, match="GOOGLE_API_KEY"):
        await sistema.responder(PREGUNTA)


async def test_error_5xx_de_gemini_se_reintenta(carpeta_base):
    emb = EmbeddingsConFallas(gemini_error(gerr.ServerError, 503), fallas=2)
    sistema, _ = armar(carpeta_base, [OK], embeddings=emb)
    r = await sistema.responder(PREGUNTA)
    assert emb.intentos_consulta == 3 and r.encontrada


async def test_gemini_caido_agota_los_reintentos(carpeta_base):
    emb = EmbeddingsConFallas(gemini_error(gerr.ServerError, 503), fallas=99)
    sistema, llm = armar(carpeta_base, [OK], embeddings=emb)
    with pytest.raises(ProveedorError, match="ServerError"):
        await sistema.responder(PREGUNTA)
    assert emb.intentos_consulta == 3 and llm.llamadas == 0


# ------------------------------------------------------------------ entradas inválidas
@pytest.mark.parametrize("pregunta", ["", "   ", "\n\t"])
async def test_pregunta_vacia_falla_antes_de_gastar_llamadas(carpeta_base, pregunta):
    emb = EmbeddingsSimulados()
    sistema, llm = armar(carpeta_base, [OK], embeddings=emb)
    antes = emb.llamadas_consultas
    with pytest.raises(ValueError, match="vacia"):
        await sistema.responder(pregunta)
    assert llm.llamadas == 0 and emb.llamadas_consultas == antes


# ------------------------------------------------------------------ concurrencia
async def test_consultas_con_gather_corren_a_la_vez(carpeta_base):
    """Las 5 llamadas al modelo estan EN VUELO al mismo tiempo (si fueran secuenciales, la barrera vence)."""
    sistema, llm = armar(carpeta_base, [OK], esperar_n=5)
    preguntas = [f"{PREGUNTA} (variante {i})" for i in range(5)]
    resultados = await asyncio.gather(*(sistema.responder(p) for p in preguntas))
    assert llm.max_en_vuelo == 5
    assert [r.pregunta for r in resultados] == preguntas          # cada respuesta corresponde a su pregunta
    assert all(r.metricas.intentos == 1 for r in resultados)      # y las metricas no se mezclan entre consultas


async def test_en_serie_nunca_hay_mas_de_una_llamada_a_la_vez(carpeta_base):
    sistema, llm = armar(carpeta_base, [OK])
    for _ in range(3):
        await sistema.responder(PREGUNTA)
    assert llm.max_en_vuelo == 1


async def test_las_metricas_de_consultas_concurrentes_no_se_contaminan(carpeta_base):
    """Una consulta con reintento no debe sumar sus intentos/tokens a las consultas vecinas."""
    # Con un guion unico y concurrencia el orden no es determinista; se usan dos sistemas con el mismo handler-por-consulta.
    bueno, _ = armar(carpeta_base, [OK])
    con_reintento, _ = armar(carpeta_base, ["basura", OK])
    a, b = await asyncio.gather(bueno.responder(PREGUNTA), con_reintento.responder(PREGUNTA))
    assert (a.metricas.intentos, b.metricas.intentos) == (1, 2)


# ------------------------------------------------------------------ construccion del sistema real
def test_build_model_sin_clave_explica_que_hacer(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfiguracionError, match="ANTHROPIC_API_KEY"):
        build_model()


def test_build_model_configura_claude_como_se_espera(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa-de-prueba")
    modelo = build_model()
    assert modelo.model == config.LLM_MODEL
    assert modelo.max_retries == 0            # la unica politica de reintentos es la de la cadena
    assert modelo.temperature is None         # no se fuerza temperature (igual que en la pre-entrega 2)


@pytest.fixture
def sin_cache(monkeypatch):
    """`get_default_sistema` se cachea; los tests lo limpian y evitan leer el .env real del proyecto."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    rag_chain.get_default_sistema.cache_clear()
    yield
    rag_chain.get_default_sistema.cache_clear()


def test_sistema_por_defecto_sin_claves_falla_con_mensaje_claro(sin_cache, monkeypatch):
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfiguracionError, match="GOOGLE_API_KEY"):
        rag_chain.get_default_sistema()


def test_sistema_por_defecto_lee_la_configuracion_del_entorno(sin_cache, monkeypatch, carpeta_base):
    monkeypatch.setattr(rag_chain, "crear_embeddings", lambda: EmbeddingsSimulados())
    monkeypatch.setattr(config, "PERSIST_DIR", carpeta_base)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa-de-prueba")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    sistema = rag_chain.get_default_sistema()
    assert sistema.recuperador.k == 3
    assert sistema.modelo_llm == "claude-sonnet-5-5"
    assert rag_chain.get_default_sistema() is sistema        # se crea una sola vez
