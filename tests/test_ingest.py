"""Tests de ingest.py. Corren sin API key y sin red (embeddings simulados + contador de palabras)."""
import sys
import types
from pathlib import Path

import pytest

import config
import ingest
from embeddings import EmbeddingsSimulados
from errores import EmbeddingsNoCoincidenError, IngestaError

TAM, SOLAPE = 30, 5  # chunking chico para que los documentos de prueba sean cortos


def contar_palabras(texto: str) -> int:
    return len(texto.split())


def seccion(numero: int, tema: str) -> str:
    """Una seccion de ~25 palabras: dos de ellas juntas superan TAM, asi que quedan separadas."""
    cuerpo = " ".join(f"{tema}{i}" for i in range(1, 23))
    return f"## {numero}. {tema.capitalize()}\n\n{cuerpo}\n"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    carpeta = tmp_path / "data"
    carpeta.mkdir()
    (carpeta / "a.md").write_text("# Documento A\n\n" + seccion(1, "alfa") + "\n" + seccion(2, "beta"), encoding="utf-8")
    (carpeta / "b.md").write_text("# Documento B\n\n" + seccion(1, "gamma"), encoding="utf-8")
    return carpeta


def correr(data: Path, tmp_path: Path, embeddings=None, **kw):
    return ingest.ingestar(
        data, tmp_path / "vs", embeddings or EmbeddingsSimulados(),
        chunk_size=TAM, chunk_overlap=SOLAPE, contador=contar_palabras, **kw,
    )


# ------------------------------------------------------------------ limpieza y carga
def test_limpiar_texto_normaliza_saltos_y_espacios():
    sucio = "Linea uno   \r\n\r\n\r\n\r\nLinea dos\r\n\n"
    assert ingest.limpiar_texto(sucio) == "Linea uno\n\nLinea dos"


def test_cargar_documentos_lee_md_y_txt_y_ignora_otros(tmp_path: Path):
    (tmp_path / "uno.md").write_text("# Uno\n\ntexto", encoding="utf-8")
    (tmp_path / "dos.txt").write_text("texto plano", encoding="utf-8")
    (tmp_path / "imagen.png").write_bytes(b"\x89PNG")
    docs = ingest.cargar_documentos(tmp_path)
    assert [d.metadata["source"] for d in docs] == ["dos.txt", "uno.md"]


def test_carpeta_inexistente(tmp_path: Path):
    with pytest.raises(IngestaError, match="No existe la carpeta"):
        ingest.cargar_documentos(tmp_path / "no-existe")


def test_carpeta_sin_documentos(tmp_path: Path):
    with pytest.raises(IngestaError, match="No hay archivos"):
        ingest.cargar_documentos(tmp_path)


def test_archivo_que_no_es_utf8(tmp_path: Path):
    (tmp_path / "malo.md").write_bytes("canción".encode("latin-1"))
    with pytest.raises(IngestaError, match="UTF-8"):
        ingest.cargar_documentos(tmp_path)


def test_todos_vacios(tmp_path: Path):
    (tmp_path / "vacio.md").write_text("  \n\n ", encoding="utf-8")
    with pytest.raises(IngestaError, match="vacios"):
        ingest.cargar_documentos(tmp_path)


# ------------------------------------------------------------------ chunking
def test_overlap_mayor_o_igual_al_tamano_es_un_error():
    with pytest.raises(ValueError):
        ingest.crear_splitter(contar_palabras, chunk_size=10, chunk_overlap=10)


def test_fragmentos_no_superan_el_tamano_y_tienen_metadatos(data: Path):
    fr = ingest.fragmentar(ingest.cargar_documentos(data), contar_palabras, TAM, SOLAPE)
    assert len(fr) >= 3
    for f in fr:
        assert contar_palabras(f.page_content) <= TAM
        assert {"source", "chunk_id", "chunk_index", "seccion", "tokens", "start_index"} <= set(f.metadata)
        assert f.metadata["tokens"] == contar_palabras(f.page_content)


DOC = "# Doc\n\n## 1. Uno\n\naaa\n\n## 2. Dos\n\nbbb\n\n## 3. Tres\n\nccc\n\n## 4. Cuatro\n\nddd\n"


def _tramo(desde: str, hasta: str) -> tuple[int, int]:
    """Posicion (inicio, fin) del tramo del DOC que va de `desde` hasta `hasta` inclusive."""
    inicio = DOC.index(desde)
    return inicio, DOC.index(hasta) + len(hasta)


def test_fragmento_dentro_de_una_seccion_cita_solo_esa_seccion():
    assert ingest._seccion_del_fragmento(DOC, *_tramo("aaa", "aaa")) == "1. Uno"


def test_fragmento_que_cruza_secciones_las_cita_todas_en_orden():
    """El caso real: el plazo de 4 horas estaba en la seccion 4 pero el fragmento arrancaba en la 3."""
    assert ingest._seccion_del_fragmento(DOC, *_tramo("aaa", "bbb")) == "1. Uno / 2. Dos"


def test_si_arranca_en_el_titulo_del_documento_se_citan_los_subtitulos():
    assert ingest._seccion_del_fragmento(DOC, 0, DOC.index("bbb")) == "1. Uno / 2. Dos"


def test_se_limita_la_cantidad_de_titulos_de_la_cita():
    assert ingest._seccion_del_fragmento(DOC, *_tramo("aaa", "ddd")) == "1. Uno / 2. Dos / 3. Tres / ..."


def test_documento_sin_subtitulos_cita_su_titulo():
    assert ingest._seccion_del_fragmento("# Solo titulo\n\ntexto", 0, 20) == "Solo titulo"


def test_texto_antes_de_cualquier_titulo_no_tiene_seccion():
    assert ingest._seccion_del_fragmento("texto suelto\n\n## 1. Uno\n\nx", 0, 5) == ""


def test_la_seccion_del_fragmento_es_el_titulo_donde_empieza(data: Path):
    fr = ingest.fragmentar(ingest.cargar_documentos(data), contar_palabras, TAM, SOLAPE)
    secciones = {(f.metadata["source"], f.metadata["seccion"]) for f in fr}
    assert ("a.md", "1. Alfa") in secciones
    assert ("a.md", "2. Beta") in secciones
    assert ("b.md", "1. Gamma") in secciones


def test_los_ids_son_deterministas_y_unicos(data: Path):
    docs = ingest.cargar_documentos(data)
    ids1 = [f.metadata["chunk_id"] for f in ingest.fragmentar(docs, contar_palabras, TAM, SOLAPE)]
    ids2 = [f.metadata["chunk_id"] for f in ingest.fragmentar(docs, contar_palabras, TAM, SOLAPE)]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)


def test_el_id_depende_del_archivo_y_del_texto():
    base = ingest.id_de_fragmento("a.md", "hola")
    assert base == ingest.id_de_fragmento("a.md", "hola")
    assert base != ingest.id_de_fragmento("b.md", "hola")
    assert base != ingest.id_de_fragmento("a.md", "hola!")


# ------------------------------------------------------------------ contador de tokens
def test_sin_tiktoken_se_usa_la_aproximacion(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # fuerza ImportError al importar
    contar, descripcion = ingest.crear_contador_de_tokens()
    assert contar is ingest.contar_tokens_aproximado
    assert "aproximado" in descripcion
    assert contar("a" * 35) == 10  # 35 caracteres / 3.5


def test_con_tiktoken_se_mide_con_el_vocabulario(monkeypatch):
    class Encoding:
        def encode(self, texto, disallowed_special=()):
            return texto.split()

    falso = types.SimpleNamespace(get_encoding=lambda nombre: Encoding())
    monkeypatch.setitem(sys.modules, "tiktoken", falso)
    contar, descripcion = ingest.crear_contador_de_tokens()
    assert contar("uno dos tres") == 3
    assert descripcion == f"tiktoken/{config.TOKEN_ENCODING}"


# ------------------------------------------------------------------ ingesta y persistencia
def test_primera_ingesta_embebe_todo(data: Path, tmp_path: Path):
    r = correr(data, tmp_path)
    assert r.archivos == 2
    assert r.nuevos == r.fragmentos > 0
    assert (r.sin_cambios, r.eliminados) == (0, 0)
    assert not r.base_al_dia
    assert r.tokens_max <= TAM


def test_reindexar_sin_cambios_no_llama_a_la_api_ni_duplica(data: Path, tmp_path: Path):
    emb = EmbeddingsSimulados()
    primera = correr(data, tmp_path, emb)
    llamadas = emb.llamadas_documentos
    segunda = correr(data, tmp_path, emb)

    assert segunda.base_al_dia
    assert (segunda.nuevos, segunda.sin_cambios) == (0, primera.fragmentos)
    assert emb.llamadas_documentos == llamadas  # no se re-embebio nada


def test_la_base_persiste_en_disco(data: Path, tmp_path: Path):
    r = correr(data, tmp_path)
    reabierta = ingest.abrir_coleccion(EmbeddingsSimulados(), tmp_path / "vs")
    assert len(reabierta.get(include=[])["ids"]) == r.fragmentos


def test_si_cambia_un_archivo_solo_se_reindexan_sus_fragmentos(data: Path, tmp_path: Path):
    primera = correr(data, tmp_path)
    (data / "a.md").write_text(
        "# Documento A\n\n" + seccion(1, "alfa") + "\n" + seccion(2, "delta"), encoding="utf-8"
    )
    segunda = correr(data, tmp_path)

    assert segunda.nuevos == 1 and segunda.eliminados == 1  # solo la seccion "beta" -> "delta"
    assert segunda.fragmentos == primera.fragmentos
    db = ingest.abrir_coleccion(EmbeddingsSimulados(), tmp_path / "vs")
    assert len(db.get(include=[])["ids"]) == segunda.fragmentos  # sin duplicados ni sobrantes


def test_si_se_quita_un_archivo_se_borran_sus_fragmentos(data: Path, tmp_path: Path):
    primera = correr(data, tmp_path)
    (data / "b.md").unlink()
    segunda = correr(data, tmp_path)

    assert segunda.eliminados > 0
    assert segunda.fragmentos == primera.fragmentos - segunda.eliminados
    db = ingest.abrir_coleccion(EmbeddingsSimulados(), tmp_path / "vs")
    fuentes = {m["source"] for m in db.get(include=["metadatas"])["metadatas"]}
    assert fuentes == {"a.md"}


def test_reset_reconstruye_desde_cero(data: Path, tmp_path: Path):
    emb = EmbeddingsSimulados()
    primera = correr(data, tmp_path, emb)
    tercera = correr(data, tmp_path, emb, reset=True)
    assert tercera.nuevos == primera.fragmentos  # todo se vuelve a embeber


# ------------------------------------------------------------------ embeddings que no coinciden
class OtroModelo(EmbeddingsSimulados):
    nombre_modelo = "otro-modelo-de-embeddings"


def test_abrir_la_base_con_otro_modelo_de_embeddings_falla(data: Path, tmp_path: Path):
    correr(data, tmp_path, EmbeddingsSimulados())
    with pytest.raises(EmbeddingsNoCoincidenError) as info:
        correr(data, tmp_path, OtroModelo())
    assert "--reset" in str(info.value)


def test_con_reset_se_puede_cambiar_de_modelo(data: Path, tmp_path: Path):
    correr(data, tmp_path, EmbeddingsSimulados())
    r = correr(data, tmp_path, OtroModelo(), reset=True)
    assert r.modelo_embeddings == "otro-modelo-de-embeddings"
    assert r.nuevos == r.fragmentos


# ------------------------------------------------------------------ dataset real y CLI
def test_el_dataset_de_ejemplo_cumple_lo_que_pide_la_consigna(tmp_path: Path):
    r = ingest.ingestar(
        config.DATA_DIR, tmp_path / "vs", EmbeddingsSimulados(), contador=ingest.contar_tokens_aproximado
    )
    assert 3 <= r.archivos <= 6        # "3 o 4 archivos de texto sobre un tema"
    assert r.fragmentos >= 8           # suficientes para que top_k de 3 a 5 tenga sentido
    assert r.tokens_max <= config.CHUNK_SIZE_TOKENS


def test_cli_ok_y_cli_con_error(data: Path, tmp_path: Path, capsys):
    ok = ingest.main(["--simular", "--data", str(data), "--persist-dir", str(tmp_path / "vs")])
    salida = capsys.readouterr().out
    assert ok == 0 and "Fragmentos vigentes" in salida

    mal = ingest.main(["--simular", "--data", str(tmp_path / "no-existe"), "--persist-dir", str(tmp_path / "vs2")])
    assert mal == 1
    assert "No existe la carpeta" in capsys.readouterr().err
