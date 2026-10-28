"""Errores propios del sistema RAG: cada uno dice QUE fallo y COMO arreglarlo."""


class RagError(Exception):
    """Base de todos los errores del sistema."""


class ConfiguracionError(RagError):
    """Falta una clave o una configuracion necesaria (por ejemplo GOOGLE_API_KEY)."""


class IngestaError(RagError):
    """No se pudieron leer o procesar los documentos de origen."""


class EmbeddingsNoCoincidenError(RagError):
    """La base fue creada con otro modelo de embeddings que el que se usa ahora.

    Mezclar modelos hace que las distancias no tengan sentido (resultados aleatorios).
    """


class BaseVectorialVaciaError(RagError):
    """La base vectorial no tiene fragmentos: hay que indexar los documentos primero."""


class RespuestaInvalidaError(RagError):
    """El modelo respondio, pero NO cumple el contrato (JSON roto, cita un fragmento que no existe...).

    Es un error REINTENTABLE: otra llamada puede devolver una respuesta correcta.
    """


class ProveedorError(RagError):
    """Falla de un proveedor (Gemini o Claude) que no se arregla reintentando
    (clave invalida, cuota agotada) o que siguio fallando tras agotar los reintentos."""
