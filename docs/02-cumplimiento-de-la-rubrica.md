# Cumplimiento de la rúbrica

Para cada criterio de evaluación indico **dónde está en el código**, **qué test lo protege** y **qué evidencia real tengo**. Todo lo marcado como evidencia está en [`evidencia/`](evidencia/) y salió de corridas reales con Gemini y Claude.

Cómo verificarlo en 1 minuto, sin API keys:

```bash
pip install -r requirements.txt
python -m pytest -q          # 127 passed
python main.py --pruebas --simular   # el flujo completo sin claves (no mide calidad)
```

---

## 1. Chunking e ingesta (25 %)

| Requisito de la consigna | Dónde | Test / evidencia |
|---|---|---|
| Leer `.txt` / `.md` de `/data` | `ingest.cargar_documentos` | `test_cargar_documentos_lee_md_y_txt_y_ignora_otros`, `test_carpeta_inexistente`, `test_carpeta_sin_documentos`, `test_archivo_que_no_es_utf8`, `test_todos_vacios` |
| Preprocesamiento | `ingest.limpiar_texto` | `test_limpiar_texto_normaliza_saltos_y_espacios` |
| `RecursiveCharacterTextSplitter`, 500 tokens, 50 de overlap, **medido en tokens** | `ingest.crear_splitter` (`length_function` = contador de `tiktoken`); valores en `config.py` | `test_fragmentos_no_superan_el_tamano_y_tienen_metadatos`, `test_con_tiktoken_se_mide_con_el_vocabulario`, `test_overlap_mayor_o_igual_al_tamano_es_un_error`, `test_el_dataset_de_ejemplo_cumple_lo_que_pide_la_consigna`. Real: 17 fragmentos, 61–492 tokens, promedio 367,5, `tiktoken/cl100k_base` → [`ingesta-segunda-corrida.txt`](evidencia/ingesta-segunda-corrida.txt) |
| Base **persistente** en ChromaDB local | `ingest.abrir_coleccion` (`chromadb.PersistentClient`, espacio coseno) | `test_la_base_persiste_en_disco` |
| **Verificar si la base ya existe** antes de reindexar | `ingest.ingestar`: compara los IDs guardados con los vigentes y embebe solo los nuevos | `test_reindexar_sin_cambios_no_llama_a_la_api_ni_duplica`. Real: 2.ª corrida → *0 nuevos, 17 sin cambios* |
| **IDs deterministas** | `ingest.id_de_fragmento` = `sha256(archivo + texto)[:16]` | `test_los_ids_son_deterministas_y_unicos`, `test_el_id_depende_del_archivo_y_del_texto` |
| Metadatos para citar la fuente | `source`, `seccion`, `chunk_id`, `chunk_index`, `tokens`, `start_index` (`ingest.fragmentar`) | `test_fragmento_que_cruza_secciones_las_cita_todas_en_orden` y los otros 6 de sección |
| Cambios en los documentos | un archivo cambiado reemplaza solo sus fragmentos; uno quitado, los borra | `test_si_cambia_un_archivo_solo_se_reindexan_sus_fragmentos`, `test_si_se_quita_un_archivo_se_borran_sus_fragmentos`, `test_reset_reconstruye_desde_cero` |

**Extra:** la `seccion` lista todos los títulos que toca el fragmento. Lo descubrí en una corrida real: el plazo de 4 horas estaba en la sección 4, pero el fragmento empezaba en la 3 y la cita decía solo "3. Medición y reporte".

**Sobre "mínimo 500 tokens":** la consigna pide "mínimo 500 tokens con 50 de overlap". Interpreté `chunk_size=500` y `chunk_overlap=50`: 500 es el tamaño objetivo y también el tope, y el último fragmento de cada documento puede ser más chico. Está explicado en el README y se cambia en un solo lugar (`CHUNK_SIZE_TOKENS` en `config.py`).

## 2. Arquitectura asíncrona y recuperación (35 %)

| Requisito | Dónde | Test / evidencia |
|---|---|---|
| Retriever por similitud con `top_k` entre 3 y 5 | `retriever.Recuperador` (valida 3 ≤ k ≤ 5; por defecto 4) | `test_k_dentro_del_rango_pedido_es_valido`, `test_k_fuera_del_rango_3_a_5_se_rechaza`, `test_devuelve_k_fragmentos_numerados_y_ordenados_por_cercania` |
| **El mismo modelo de embeddings** para indexar y consultar (error #1) | el nombre del modelo se guarda en la colección y `abrir_coleccion` (ingesta **y** consulta) lanza `EmbeddingsNoCoincidenError` si no coincide | `test_abrir_la_base_con_otro_modelo_de_embeddings_falla`, `test_modelo_de_embeddings_distinto_se_detecta`, `test_usa_el_mismo_modelo_de_embeddings_para_consultar` |
| Recupera lo correcto | — | `test_las_preguntas_con_respuesta_traen_el_documento_correcto` (4 preguntas). Real: una pregunta **sin palabras en común** con el documento igual trajo el fragmento correcto (prueba nº 5) |
| Cadena **LCEL** con prompt "filtro de veracidad" | `rag_chain.build_chain` (`contexto → prompt → ChatAnthropic → PydanticOutputParser → verificación`); prompt en `prompts.py` | `test_el_prompt_lleva_la_regla_no_lo_se_y_el_contexto_numerado` |
| `get_rag_response(query)` **asíncrona** | `rag_chain.get_rag_response` → `SistemaRAG.responder`: `await aembed_query`, búsqueda en hilo con `asyncio.to_thread`, `await chain.ainvoke` | `test_get_rag_response_usa_el_sistema_indicado` |
| Concurrencia real | `asyncio.gather` en `main.py --pruebas --concurrente` | `test_consultas_con_gather_corren_a_la_vez` (una barrera exige que las 5 llamadas estén **a la vez** en vuelo; con consultas en serie fallaría por *timeout*; no depende de tiempos), `test_en_serie_nunca_hay_mas_de_una_llamada_a_la_vez`, `test_consultas_en_paralelo_dan_lo_mismo_que_una_por_una`. Real: **2,25 s en paralelo contra 14,83 s en serie** ([paralelo](evidencia/pruebas-reales-en-paralelo.txt) / [serie](evidencia/pruebas-reales-en-serie.txt)) |

## 3. Validación con Pydantic y esquemas (20 %)

| Requisito | Dónde | Test / evidencia |
|---|---|---|
| Salida parseada con **`PydanticOutputParser`** | `rag_chain.build_chain` | `test_respuesta_invalida_se_reintenta_y_se_cuenta[no-json / campos-faltantes]` |
| Modelo con **texto + referencias** | `schemas.RespuestaLLM` (lo que devuelve Claude) y `schemas.RespuestaRAG` (lo que devuelve el sistema: respuesta, `fuentes` con archivo y sección, `recuperados`, `metricas`) | `test_respuesta_rag_ida_y_vuelta_json`, `test_respuesta_con_fuentes_y_metricas`. Real: [`ejemplo-respuesta.json`](evidencia/ejemplo-respuesta.json) |
| Coherencia interna | validadores: `encontrada=true` exige referencias; `encontrada=false` prohíbe referencias y exige empezar con "No lo sé"; referencias ≥ 1 y sin repetidos; el modelo **no puede escribir nombres de archivo**, solo números | los 18 tests de `tests/test_schemas.py` |
| La fuente citada **existe de verdad** | `rag_chain._verificar_referencias`: si el modelo cita `[9]` habiendo recibido 4, se rechaza y se reintenta | `test_la_cita_a_un_fragmento_inexistente_nunca_llega_al_usuario`. *Pydantic valida la forma; el código valida la verdad.* |
| Las fuentes son solo lo citado | `fuentes` = fragmentos citados; `recuperados` = todo lo que trajo el retriever | `test_varias_fuentes_se_conservan_en_el_orden_del_retriever`, `test_pregunta_sin_respuesta_en_los_documentos` |

## 4. Robustez y evidencia de pruebas (20 %)

| Requisito | Dónde | Test / evidencia |
|---|---|---|
| **Dos pruebas pedidas**: pregunta con respuesta y pregunta trampa | `main.py --pruebas`: **5 con respuesta** + **5 trampa** (presupuesto, teletrabajo, sucursales, sueldo —tema cercano—, e intento de manipulación), cada una con veredicto automático | `test_main.py` (16 tests, incluido que el "juez" no se deja engañar). Real: **10/10** → [`pruebas-reales-en-serie.txt`](evidencia/pruebas-reales-en-serie.txt) |
| No alucinar | prompt + contrato (`encontrada=false` ⇒ sin fuentes) | las 5 trampas reales dijeron *"No lo sé"* sin citar fuentes |
| Reintentos con criterio | `.with_retry` (máx. 3, backoff con jitter): JSON inválido, contrato incumplido, cita inexistente, red / 429 / 5xx de Anthropic, 5xx de Gemini | `test_respuesta_invalida_se_reintenta_y_se_cuenta`, `test_error_de_red_transitorio_se_reintenta`, `test_error_5xx_de_gemini_se_reintenta`, `test_se_respeta_el_maximo_de_intentos` |
| **No** reintentar lo que no se arregla esperando | clave inválida (401/403) y el 429 de Gemini | `test_clave_invalida_no_se_reintenta`, `test_cuota_gratuita_de_gemini_da_mensaje_claro_y_no_se_reintenta`. Real: [`clave-invalida.txt`](evidencia/clave-invalida.txt) |
| Errores accionables | `errores.py`; `rag_chain._traducir_error`; el CLI muestra `ERROR: …` sin traza | `test_sin_claves_el_error_es_claro_y_no_un_traceback`, `test_sin_clave_falla_con_un_mensaje_que_dice_como_arreglarlo`, `test_base_inexistente_pide_indexar_primero` |
| Entradas inválidas | pregunta vacía falla **antes** de gastar llamadas | `test_pregunta_vacia_falla_antes_de_gastar_llamadas`, `test_pregunta_vacia_se_rechaza_sin_llamar_al_proveedor` |
| Observabilidad | `Metricas`: tiempos de recuperación y generación, **intentos** y **tokens de todos los intentos**; por consulta, sin mezclarse entre consultas concurrentes | `test_respuesta_invalida_se_reintenta_y_se_cuenta`, `test_las_metricas_de_consultas_concurrentes_no_se_contaminan` |
| **Seguridad de las credenciales** | las claves viven en el `.env` (en `.gitignore`); `.env.example` solo trae los nombres, con valores vacíos; si falta una clave, el error dice cuál y no muestra una traza | `git ls-files` no lista ningún `.env`; `test_sin_claves_el_error_es_claro_y_no_un_traceback` |
| Tests que corren **sin red ni claves** | embeddings y LLM simulados (`embeddings.py`, `llm_simulado.py`) y un LLM de guion en los tests | **127 passed** (Python 3.12, 3.13 y 3.14) |

### Cómo comprobé que los tests protegen de verdad
Probé la suite contra **23 errores introducidos a propósito** en el código (no verificar citas inexistentes, devolver todo lo recuperado como "fuentes", un solo intento, reintentar la clave inválida, reintentar el 429 de Gemini, contar solo los tokens del último intento, serializar las consultas, quitar la regla "No lo sé" del prompt, desactivar la validación de `top_k`, invertir el orden de los fragmentos, etc.): **22 fueron detectados**. El restante (quitar la validación de pregunta vacía en `SistemaRAG.responder`) es equivalente: el `Recuperador` valida lo mismo, así que el comportamiento no cambia.

Además, los tests encontraron un error real mientras los escribía: `OutputParserException` y `ValidationError` heredan de `ValueError`, y un `except ValueError` los interceptaba antes de convertirlos en un error propio. Está corregido y protegido por `test_se_respeta_el_maximo_de_intentos`.

## Límites conocidos
- Medí la calidad de las respuestas con **10 preguntas** reales sobre un dataset chico y sintético; no es una evaluación estadística.
- El LLM y los embeddings simulados prueban la lógica, no la calidad.
- La redacción de las respuestas varía entre corridas (no fijo `temperature`); los datos y las fuentes no.
- No hay umbral de distancia para decir "No lo sé", y es deliberado (ver el README: una trampa quedó a 0,285 y una respuesta válida a 0,311).
