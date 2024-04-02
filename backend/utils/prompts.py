from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType



############# ToDo ############################
"""add query gen prompt here. Now it's in rag_session.py"""

#########################################



############# CUSTOM PROMPTS ############################

CUSTOM_REFINE_PROMPT_TMPL = (
    "La consulta original es la siguiente: {query_str}\n"
    "Hemos proporcionado una respuesta existente: {existing_answer}\n"
    "Tenemos la oportunidad de refinar la respuesta existente "
    "(solo si es necesario) con un poco más de contexto a continuación.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Dado el nuevo contexto, refine la respuesta original para responder mejor "
    "a la consulta. "
    "Si el contexto no es útil, devuelva la respuesta original.\n"
    "Respuesta Refinada: "
)
CUSTOM_REFINE_PROMPT = PromptTemplate(
    CUSTOM_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)



CUSTOM_TEXT_QA_PROMPT_TMPL = (
    "La información de contexto está a continuación.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Sos un experto en derecho y leyes argentinas, enfocado en responder preguntas sobre el Decreto de Necesidad y Urgencia (DNU) del 20-12-2023 del presidente Javier Milei.\n"
    "Utilizá siempre el contexto proporcionado para dar una respuesta INTEGRAL, INFORMATIVA, PRECISA y OBJETIVA a la pregunta del usuario. No debes emitir opinión.\n"
    "SIEMPRE responder en castellano. Si la información es insuficiente para responder la pregunta, responde sinceramente que no sabes; no intentes inventar una respuesta ni utilizar conocimiento previos.\n"
    "Debes utilizar la fecha de emisión del contexto para dar formato a tu respuesta. Tu respuesta debe explicar la legislación ANTES del DNU del 20-12-2023, y cómo serán las cosas DESPUÉS del DNU del 20-12-2023.\n"
    "Si resulta útil, tu respuesta incluirá la sección ANTES DEL DNU 20-12-2023 y la sección DESPUES DEL DNU 20-12-2023.\n"
    "Pregunta: {query_str}\n"
    "Respuesta: "
)
CUSTOM_TEXT_QA_PROMPT = PromptTemplate(
    CUSTOM_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)










############# DEFAULT llama_index PROMPTS ######################

DEFAULT_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)
