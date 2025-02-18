# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: nlp
#     language: python
#     name: python3
# ---

# %% [markdown]
# In this notebook we will explore the implementation of the [Instructor](https://github.com/jxnl/instructor) (which uses Pydantic under the hood). The objective of implementing this library is to have more control over the not-so-reliable behaviour of [OpenAI's LLM JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode). 
# We will begin by setting a baseline of how the JSON mode behaves without `Instructor`, and then compare it when `Instructor` is added. 
#
# The task of the LLM is to extract information from an user query in JSON format. 
#

# %% [markdown]
# Baseline

# %%
queries = [
    # queries about specific articles
    "que dice el articulo 95 del DNU?", 
    "explicame el art 90",
    "cuales son todos los articulos del Titulo III?", 
    "cuales son todos los articulos del Titulo 3?", 
    "que dicen los articulos 30 y 35?",
    "cuales son los contenidos del Titulo I y el Titulo II?", 
    "cuales son los contenidos del Capitulo I?",
    "cuales son los contenidos del Capitulo 1?",
    "cuales son todos los articulos del Capitulo I del Titulo XII?", 
    "cuales son todos los articulos del Capitulo 1 del Titulo 12?", 
]


import re
import json
#from openai import OpenAI
import openai
from typing import Dict, Optional

def extract_json(assistant_msg: str) -> Optional[Dict[str, str]]:
    """
    Extracts a JSON object from a string.

    Args:
        assistant_msg: A string that contains a JSON object.

    Returns:
        A dictionary that represents the JSON object if it exists, None otherwise.
    """
    # Using regular expression to find the JSON part in the string
    json_match = re.search(r'\{.*\}', assistant_msg, re.DOTALL)

    if json_match:
        json_part = json_match.group(0)
        try:
            # Converting the JSON string into a Python dictionary
            json_data = json.loads(json_part)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            json_data = None
    else:
        print("Error parsing JSON: No valid JSON found in the string")
        json_data = None
    
    return json_data

def get_legal_info(query: str, model: str = "gpt-3.5-turbo-1106") -> Optional[Dict[str, str]]:
    """
    Retrieves legal information based on a user's query.

    Args:
        query: A string that represents the user's query.
        model: A string that represents the model to be used.

    Returns:
        A dictionary that represents the legal information if it exists, None otherwise.
    """
    client = openai.OpenAI()
    conversation = []

    interpreter_prompt = """
    Tu objetivo es interpretar una consulta sobre informacion legal de la argentina y rellenar los campos de un archivo JSON. A continuacion un EJEMPLO de como se veria un archivo JSON correctamente completado: 
    {
        'titulo': Titulo II, 
        'capitulo': Capitulo I, 
        'articulo': Articulo 12, 
    }

    Ten en cuenta que no todos los campos tiene que tener un valor especifico, es decir, alguno de los campos 'titulo', 'capitulo', o 'articulo' puede ser estar VACIO si la consulta no hace referencia a ese campo. Por EJEMPLO:

    {
        'titulo': Titulo I, 
        'capitulo': , 
        'articulo': Articulo 1, 
    }

    Ten en cuenta que los valores de los campos tienen que ser el nombre del campo del JSON mas caracteres númericos, ya sea con números romanos o con números arábigos. EJEMPLO: el campo 'titulo' siempre debe contener la palabra 'Titulo' seguida de un espacio y un número romano o arábigo. Por EJEMPLO, 'Titulo I' o 'Titulo 1'.
    Por EJEMPLO, si la consulta es "cuales son todos los articulos sobre la reforma del estado", el JSON deberia tener todos los campos VACIOS, porque si bien se nombra la palabra "articulo", no se menciona ningun numero. 

    Si la consulta refiere a dos Articulos, Titulos or Capitulos distintos (e.g., "que dicen los articulos 60 y 78?") el JSON deberá incluir estos dos separados por una coma. POR EJEMPLO:
    {
        'titulo': , 
        'capitulo': , 
        'articulo': Articulo 60, Articulo 78, 
    }

    Nota: NUNCA COMPLETES LOS CAMPOS DEL JSON CON UNA PALABRA QUE NO SEA "titulo", "capitulo", o "articulo", seguidos por un numbero arabigo o romano.

    Por favor, no uses la informacion de los EJEMPLOS, es solo para que entiendas el formato. La informacion que contiene es ficticia, NO LA USES.
    Manten siempre el formato propuesto en los EJEMPLOS.
    No hagas asumpciones acerca de ninguno de los campos. Completa los campos solo si lo consula lo incluye. Siempre devuelve todos los campos del JSON, por mas que ALGUNOS ESTEN VACIOS. 
    

    RECUERDA, NO COMPLETAR CAMPOS DEL JSON QUE NO ESTEN EN LA CONSULTA. PIENSA CRITICAMENTE Y PASO POR PASO ANTES DE COMPLETAR CADA CAMPO.
    """

    system_interpretative = [
        {"role": "system", "content": f" {interpreter_prompt}"}
    ]

    # Append to interpretative conversation:
    conversation.append({"role": "user", "content": query})

    interpretative_completion = client.chat.completions.create(
        model=model,
        messages=system_interpretative + conversation,
        response_format={ "type": "json_object" }
    )

    output_json = extract_json(interpretative_completion.choices[0].message.content)

    return output_json
