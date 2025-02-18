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
# ToDo:
# - Improve all LLM calls with Pydantic and Instructor. 
# - Implement reproducible LLM outputs https://platform.openai.com/docs/guides/text-generation/reproducible-outputs

# %% [markdown]
# This notebook will inspect everything related to vectors --> vectors by itself, vectorstores, docstores, linear adapters, embedding models, etc
#

# %% [markdown]
# First, we will get one chunk of text that we care about and will build its vector representation

# %%
import json

with open('../data/LaLeyDeMilei-raw/decreto_flat.json') as f:
    data = json.load(f)

# %% [markdown]
# We can see that now we have an object with all the chunks of text from the DNU

# %%
data

# %% [markdown]
# We will just extract the first one and turn it into a vector with a custom embedding model

# %%
# Get the first key-value pair as a new dictionary
first_item = {list(data.keys())[0]: data[list(data.keys())[0]]}
first_item

# %% [markdown]
# Turn it into a vector

# %%
# Turn it into a vector:
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn')

# get data as str
first_chunk = first_item.get(list(first_item.keys())[0])

# encode it
first_vector = model.encode(first_chunk)

# %%
len(first_vector)

# %% [markdown]
# Ok, now we have a vector of length 768, which is the "length_embedding" of the model used (https://huggingface.co/dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn)
#
# Now we will wrap this into a function and do it for all the data. 

# %%
from typing import Dict, Any
from sentence_transformers import SentenceTransformer

def encode_values(original_dict: Dict[str, str], model: SentenceTransformer) -> Dict[str, Any]:
    """
    Encodes the values of a dictionary using a model.

    Args:
        original_dict: A dictionary where each value is a string.
        model: A SentenceTransformer model used for encoding.

    Returns:
        A new dictionary where each key is the same as in the original dictionary,
        and each value is the result of encoding the corresponding value in the original dictionary using the model.
    """
    return {key: model.encode(value) for key, value in original_dict.items()}



# %%
model = SentenceTransformer('dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn')

vectorstore = encode_values(data, model)

# %%
vectorstore

# %% [markdown]
# And now we will compress these 768-dimensional vectors into 2-d vectors to visualize them. The objective of this is to see if this embedding model can create vectors that are easily separated. 

# %%
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from typing import Dict, Any
import numpy as np

def plot_vectors(vectors: Dict[str, Any]) -> None:
    """
    Creates a t-SNE plot of vectors with Plotly. The function separates vectors into two categories based on their keys:
    those that start with 'query:' and those that do not. The two categories are plotted separately. Vectors whose keys
    start with 'query:' are plotted with a black star marker, while the rest are plotted with a point marker.

    Args:
        vectors: A dictionary where each key is a string and each value is a vector. The keys that start with 'query:'
        are considered as query vectors and are plotted differently from the rest.

    Returns:
        None. The function displays a plot.

    """

    # Get the list of vectors and keys
    vectors_list = list(vectors.values())
    keys_list = list(vectors.keys())

    # Convert the list of vectors to a numpy array
    vectors_array = np.array(vectors_list)

    # Create a t-SNE object
    tsne = TSNE(n_components=2, random_state=0)

    # Perform t-SNE
    vectors_2d = tsne.fit_transform(vectors_array)

    # Separate 2D coordinates for queries and non-queries
    query_vectors_2d = np.array([vec for key, vec in zip(keys_list, vectors_2d) if key.startswith('query:')])
    non_query_vectors_2d = np.array([vec for key, vec in zip(keys_list, vectors_2d) if not key.startswith('query:')])
    query_keys = [key for key in keys_list if key.startswith('query:')]
    non_query_keys = [key for key in keys_list if not key.startswith('query:')]

    # Create the plot for non-query vectors
    fig = go.Figure(data=go.Scatter(
        x=non_query_vectors_2d[:, 0],
        y=non_query_vectors_2d[:, 1],
        mode='markers',
        text=non_query_keys,  # This line sets the hover text
        marker=dict(
            size=8,
            color=non_query_vectors_2d[:, 0],  # Set color equal to x
            colorscale='Viridis',  # One of plotly colorscales
            showscale=False
        )
    ))

    # Add the plot for query vectors, if there are any
    if len(query_vectors_2d) > 0:
        fig.add_trace(go.Scatter(
            x=query_vectors_2d[:, 0],
            y=query_vectors_2d[:, 1],
            mode='markers',
            text=query_keys,  # This line sets the hover text
            marker=dict(
                size=8,
                color='black',  # Set color to black
                symbol='star'  # Set marker symbol to star
            )
        ))

    # Set the title and labels
    fig.update_layout(title=f"t-SNE plot of vectors",
                      xaxis=dict(title='Dimension 1'),
                      yaxis=dict(title='Dimension 2'))

    # Show the plot
    fig.show()


# add optional arg "plot_query_retrieval_similarity" (yes/no), and `similarity_measure` (that can be cosine, euclidean or?). If yes, plot a line between the stars and top 5 retrieved vectors


# %%
plot_vectors(vectorstore)

# %% [markdown]
# NOTES:
# - the 2-d projection looks very well separated, but this does not NECESSARILY mean that the 768-d is also very well separated. Find a way to compute average distance between n-dimensional vectors (that works well on ~700-d vectors), and use a clustering algorithm for the 700-d vectors to see if they cluster as well as the 2-d vectors. 
# - t-sne is being used with defaults, there's a lot of hyperparams to tune that will yield different behaviors in the projection. Keep this is mind. For more info, check: https://distill.pub/2016/misread-tsne/
# - This vector separation can be improved (although it looks very good) by fine-tuning a linear adapter on top of the embedding model (or just fine-tuning the embedding model by itself, but this is not the best for deplyment). Check: https://blog.llamaindex.ai/fine-tuning-a-linear-adapter-for-any-embedding-model-8dd0a142d383    and 
# https://blog.llamaindex.ai/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971
#
#
# ToDo: use ada instead of local embedding model and compare

# %% [markdown]
# Now we will also do t-sne on the chunks we used for the rag we wanted to launch into prod but didn't really work. 

# %%
# read json as dict
filepath_custom = '../data/LaLeyDeMilei/decreto_chunks_flat.json'
data_custom = json.load(open(filepath_custom))

# encode vectors
model = SentenceTransformer('dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn')
vectorstore_custom = encode_values(data_custom, model)




# %%
plot_vectors(vectorstore_custom)

# %% [markdown]
# We can see that with the chunking we had before, the vectors are very mixed in themes and in general not very well separated. !!!!!!!!! 

# %% [markdown]
# This is clearly one reason why our retrieval sucks. But the other way that I can think of now is the query vectors. Because say our vectors in the vectorstore are properly distanced. But what about the query vector? is it close to the correct retrieval vectors? is it close to any vector? This may not be the case at all, and may be because a query made by an user, say "que va a pasar con los alquileres", is written in a way that is very different to the content of a retrieval vector, e.g., 
#     "'Título XVI: REGISTRO AUTOMOTOR (Decreto - Ley N° 6582/58 ratificado por la Ley N° 14.467 (t.o. 1997) y sus modificatorias). Articulo 362: Sustitúyese el párrafo primero del artículo 23 del Decreto-Ley N° 6582/58 ratificado por la Ley N° 14.467 (t.o. 1997) y sus modificatorias por el siguiente:\n\n"ARTÍCULO 23 ...."
#
# So let's define a few queries, encode them into vector form, and project them to this 2-d space

# %%
queries = [
    # queries about topics
    'que va a pasar con las farmacias?', 
    'quiero saber todos los cambios que va a haber en el sector de deportes?', 
    'quiero saber que cambios va a haber con el titulo automotor',
    'que va a pasar con aerolineas argentinas',
    "cuales son todos los articulos sobre la reforma del estado?",

    # queries about specific articles
    "que dice el articulo 95 del DNU?", # note that here there's the articulo 95 of the dnu, and also the articulo 95 de la ley N° 22.415 that is mentioned in articulo 108 of the DNU
    #"explicame el art 90",
    #"cuales son todos los articulos del Titulo III?", # This should retrieve all the articles from Titulo II
    #"cuales son todos los articulos del Titulo 3?", 
    #"que dicen los articulos 30 y 35?",
    #"cuales son los contenidos del Titulo I y el Titulo II?", 
    #"cuales son los contenidos del Capitulo I?",
    #"cuales son los contenidos del Capitulo 1?",
    #"cuales son todos los articulos del Capitulo I del Titulo XII?", 
    #"cuales son todos los articulos del Capitulo 1 del Titulo 12?", 
]



# %%
from typing import List

def list_to_dict(input_list: List[str]) -> Dict[str, str]:
    """
    Transforms a list into a dictionary where both the key and the value are the same for each element in the list.
    The key is prepended with "query:".

    Args:
        input_list: A list of strings.

    Returns:
        A dictionary where the key is each element in the list prepended with "query:", and the value is the same element.
    """
    return {f"query: {item}": item for item in input_list}



# %%
queries_dict = list_to_dict(queries)
queries_dict

# %% [markdown]
# Now we will encode this queries, and project them with t-sne

# %%
model = SentenceTransformer('dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn')
vectorized_queries = encode_values(queries_dict, model)
vectorized_queries

# %%
# add vector queries to vectorstores
import copy

#vectorstore_with_queries = copy.deepcopy(vectorstore)
vectorstore_with_queries = {**vectorstore, **vectorized_queries}

#vectorstore_custom_with_queries = copy.deepcopy(vectorstore_custom)
vectorstore_custom_with_queries = {**vectorstore_custom, **vectorized_queries}


vectorstore.keys(), vectorstore_with_queries.keys(), vectorstore_custom.keys(), vectorstore_custom_with_queries.keys()



# %%

# %% [markdown]
# project old chunks

# %%
# project with t-sne


plot_vectors(vectorstore_custom_with_queries)

# %% [markdown]
# project new chunks

# %%
plot_vectors(vectorstore_with_queries)

# %% [markdown]
# We can see that the queries are closer to the correct retrieved chunks with the new way of chunking. We can also see that the queries that ask for the content of a given article fail to be close to the correct chunk. This will need a vector search not by a distance metric but by the content of the key (i.e., if the query asks for article 5, find the vector that has a key with the str "articulo 5" (when doing this, be careful with upper/lower case and tildes))

# %% [markdown]
# An important thing to note is that we are looking at the distance between query and retrieval vectors in the projected 2-d space. But this is not what happens when doing rag. Instead, the un-projected 768-d vectors are retrieved based on the distance with the 768-d representation of the query, where the distance can be cosine, euclidean, etc. 
#
# So to check this, we will retrieve the top 5 chunks for each query based on cosine similarity (other distance metrics to be tested) and then will plot lines in the t-sne plot. 

# %% [markdown]
# Doing the cosine similarity: 
#
#     Remember that the cosine similarity ranges from -1 (meaning vectors are diametrically opposed) to 1 (meaning they are identical). A value of 0 indicates orthogonality or decorrelation, while in-between values indicate intermediate similarity or dissimilarity. For text matching, the attribute vectors are often binary, so cosine similarity can take on values between 0 and 1, and is equivalent to the Jaccard coefficient.

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional



from sklearn.metrics.pairwise import cosine_similarity

def query_similarity(vectors: Dict[str, np.ndarray], top_k: Optional[int] = None) -> Dict[str, List[Tuple[str, float]]]:
    """
    Computes the cosine similarity between each 'query:' vector and all other vectors.

    Args:
        vectors: A dictionary where each key is a string and each value is a vector.
        top_k: An optional integer specifying the number of top vectors to return for each query. If None, all vectors are returned.

    Returns:
        A dictionary where each key is a 'query:' key from the input dictionary, and each value is a list of tuples.
        Each tuple contains a non-'query:' key from the input dictionary and the cosine similarity between the 'query:'
        vector and the non-'query:' vector. The list is sorted in descending order of similarity.
    """
    # Separate the vectors into query vectors and non-query vectors
    query_vectors = {k: v for k, v in vectors.items() if k.startswith('query:')}
    non_query_vectors = {k: v for k, v in vectors.items() if not k.startswith('query:')}

    # Initialize the result dictionary
    result = {}

    # Compute the cosine similarity between each 'query:' vector and all non-'query:' vectors
    for query_key, query_vector in query_vectors.items():
        similarities = []
        for non_query_key, non_query_vector in non_query_vectors.items():
            similarity = cosine_similarity(query_vector.reshape(1, -1), non_query_vector.reshape(1, -1))
            similarities.append((non_query_key, similarity[0][0]))
        
        # Sort the similarities in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # If top_k is specified, only keep the top_k similarities
        if top_k is not None:
            similarities = similarities[:top_k]

        # Add the similarities to the result dictionary
        result[query_key] = similarities

    return result


# %%
similarity_results = query_similarity(vectors = vectorstore_with_queries, top_k=10)

# %% [markdown]
# Now we will modify the `plot_vectors` function to compute the query similarity and plot it with dashed lines

# %%
from sklearn.metrics.pairwise import cosine_similarity
from typing import Callable

def plot_vectors(vectors: Dict[str, Any], query_similarity: Optional[Callable] = None, top_k: Optional[int] = None) -> None:
    """
    Creates a t-SNE plot of vectors with Plotly. The function separates vectors into two categories based on their keys:
    those that start with 'query:' and those that do not. The two categories are plotted separately. Vectors whose keys
    start with 'query:' are plotted with a black star marker, while the rest are plotted with a point marker.

    Args:
        vectors: A dictionary where each key is a string and each value is a vector. The keys that start with 'query:'
        are considered as query vectors and are plotted differently from the rest.
        query_similarity: An optional function that computes the cosine similarity between each 'query:' vector and all other vectors.
        top_k: An optional integer specifying the number of top vectors to return for each query. If None, all vectors are returned.

    Returns:
        None. The function displays a plot.

    """
    # Get the list of vectors and keys
    vectors_list = list(vectors.values())
    keys_list = list(vectors.keys())

    # Convert the list of vectors to a numpy array
    vectors_array = np.array(vectors_list)

    # Create a t-SNE object
    tsne = TSNE(n_components=2, random_state=0)

    # Perform t-SNE
    vectors_2d = tsne.fit_transform(vectors_array)

    # Separate 2D coordinates for queries and non-queries
    query_vectors_2d = np.array([vec for key, vec in zip(keys_list, vectors_2d) if key.startswith('query:')])
    non_query_vectors_2d = np.array([vec for key, vec in zip(keys_list, vectors_2d) if not key.startswith('query:')])
    query_keys = [key for key in keys_list if key.startswith('query:')]
    non_query_keys = [key for key in keys_list if not key.startswith('query:')]

    # Create the plot for non-query vectors
    fig = go.Figure(data=go.Scatter(
        x=non_query_vectors_2d[:, 0],
        y=non_query_vectors_2d[:, 1],
        mode='markers',
        text=non_query_keys,  # This line sets the hover text
        marker=dict(
            size=8,
            color=non_query_vectors_2d[:, 0],  # Set color equal to x
            colorscale='Viridis',  # One of plotly colorscales
            showscale=False
        )
    ))

    # Add the plot for query vectors, if there are any
    if len(query_vectors_2d) > 0:
        fig.add_trace(go.Scatter(
            x=query_vectors_2d[:, 0],
            y=query_vectors_2d[:, 1],
            mode='markers',
            text=query_keys,  # This line sets the hover text
            marker=dict(
                size=8,
                color='black',  # Set color to black
                symbol='star'  # Set marker symbol to star
            )
        ))

    # If query_similarity and top_k are specified, compute the cosine similarity and add dashed lines
    if query_similarity is not None and top_k is not None:
        # Compute the cosine similarity between each 'query:' vector and all non-'query:' vectors
        similarities = query_similarity(vectors, top_k)

        # Add dashed lines for the top_k similarities
        for query_key, top_vectors in similarities.items():
            query_vector_2d = vectors_2d[keys_list.index(query_key)]
            for non_query_key, _ in top_vectors:
                non_query_vector_2d = vectors_2d[keys_list.index(non_query_key)]
                fig.add_trace(go.Scatter(
                    x=[query_vector_2d[0], non_query_vector_2d[0]],
                    y=[query_vector_2d[1], non_query_vector_2d[1]],
                    mode='lines',
                    line=dict(color='black', width=1, dash='dash'),
                    opacity=0.5
                ))

    # Set the title and labels
    fig.update_layout(title=f"t-SNE plot of vectors",
                      xaxis=dict(title='Dimension 1'),
                      yaxis=dict(title='Dimension 2'))

    # Show the plot
    fig.show()


# %% [markdown]
# plot new chunks

# %%
plot_vectors(vectorstore_with_queries, query_similarity, top_k=5)

# %% [markdown]
# Note that in the cases where the query vector is close to the correct retrieval vectors (e.g., "quiero saber que cambios va a haber con el titulo automotor"), the cosine similarity matches pretty well the 2-d spatial representation made by t-sne. 
# But in the cases where the query vector is NOT close to the correct retrieval vectors (e.g., "explicame el articulo 90" or "cuales son todos los articulos del titulo III"), the cosine similarity does not always correspond with the 2-d spatial representation (i.e., the query vector is connected by a dashed line to a retrieval vector that is very far away). 
#

# %% [markdown]
# plot old chunks

# %%
plot_vectors(vectorstore_custom_with_queries, query_similarity, top_k=5)

# %% [markdown]
# In the case of the old chunks, most of the things suck :/

# %% [markdown]
#

# %% [markdown]
# ### Let's now build a retrieval engine based not on cosine similarity but string matching. 
#
# First we will extract the strings using openai llms that work with json. 

# %%
specific_queries = queries[5:]
specific_queries

# %%
import re
import json
from openai import OpenAI
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
    client = OpenAI()
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


# %%
query = "que dice el articulo 870"
model = "gpt-3.5-turbo-1106"
output_json = get_legal_info(query, model)
print(output_json)

# %%
for query in specific_queries: 
    output = get_legal_info(query, model)
    print(f"Query: {query}")
    print(f"Output: {output}")
    print("\n")

# %% [markdown]
# We can see that the llm works very well. Nonetheless, we will build a guardrail where if the value of the field "titulo", "capitulo", or "articulo" doesn't contain a roman or arabic number, it re-calls the llm saying that it made a mistake (explaining the mistake) and asks it to re-fill the json correctly. (check llm guardrails repo, although I think this repo is more for the content of the llm response) ---> ToDo:
#
# Note: This guardrail is sort of redundant because now we create a function that uses an llm that uses zero-shot prompting to determine if the json extractor llm should be run. 

# %% [markdown]
# Now we will build the llm that will determine if the json-extractor-llm should be run. 

# %%
queries


# %%
def get_query_specificity(query: str, model: str = "gpt-3.5-turbo-1106") -> Optional[Dict[str, str]]:
    """
    Determines if a query is asking for a specific "titulo", "capitulo", or "articulo".

    Args:
        query: A string that represents the user's query.
        model: A string that represents the model to be used.

    Returns:
        A dictionary that represents the query specificity if it exists, None otherwise.
    """
    client = OpenAI()
    conversation = []

    interpreter_prompt = """
    Tu objetivo es determinar si una consulta es sobre algo general dentro de la ley argentina o si esta preguntando especificamente por un "titulo", "capitulo", o "articulo" en particular. Rellena los campos de un archivo JSON con 'Si' si la consulta está pidiendo ese campo, o 'No' si no lo está. 
    Cuando la consulta es sobre algo especifico de un "titulo", "capitulo", o "articulo", la consulta INCLUYE CARACTERES NUMERICOS ROMANOS O ARABIGOS que indican a que "titulo", "capitulo", o "articulo" se refiere. 

    A continuación EJEMPLOS de como se veria un archivo JSON correctamente completado dadas ciertas consultas particulares donde el JSON respuesta es 'No': 
    
    CONSULTAS: 
        - 'quiero saber que cambios va a haber con el titulo automotor'
        - 'que va a pasar con aerolineas argentinas'
        - 'cuales son todos los articulos sobre la reforma del estado?'

    JSON CORRECTO A TODAS ESTAS CONSULTAS:
    {
        'pregunta_especifica': 'No', 
    }
    
    PRESTA ATENCION A LA ULTIMA CONSULTA del caso 'No': "cuales son todos los articulos sobre la reforma del estado". Si bien la consulta incluye la palabra "articulo", no incluye ningun numero que indique a que articulo se refiere. Por lo tanto, el JSON respuesta es 'No'.


    A continuación EJEMPLOS de como se veria un archivo JSON correctamente completado dadas ciertas consultas particulares donde el JSON respuesta es 'Si': 
    CONSULTAS: 
        - 'cuales son todos los articulos del Titulo III?',
        - 'cuales son todos los articulos del Titulo 3?',
        - 'que dice el articulo 870'

    JSON CORRECTO A TODAS ESTAS CONSULTAS:
    {
        'pregunta_especifica': 'Si', 
    }

    PRESTA ATENCION A LA INCLUSION DE NUMEROS ROMANOS O ARABIGOS en la consulta. 


    Por favor, no uses la informacion de los EJEMPLOS, es solo para que entiendas el formato. La informacion que contiene es ficticia, NO LA USES.
    Manten siempre el formato propuesto en los EJEMPLOS.
    No hagas asumpciones sobre el campo 'pregunta_especifica', solo completa el campo con 'Si' o 'No' basado en PENSAMIENTO CRITICO y PASO A PASO sobre la consulta. 

    QUERY:
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

# %%


client = OpenAI()
conversation = []

interpreter_prompt = """
Tu objetivo es determinar si una consulta es sobre algo general dentro de la ley argentina o si esta preguntando especificamente por un "titulo", "capitulo", o "articulo" en particular. Rellena los campos de un archivo JSON con 'Si' si la consulta está pidiendo ese campo, o 'No' si no lo está. 
Cuando la consulta es sobre algo especifico de un "titulo", "capitulo", o "articulo", la consulta INCLUYE CARACTERES NUMERICOS ROMANOS O ARABIGOS que indican a que "titulo", "capitulo", o "articulo" se refiere. 

A continuación EJEMPLOS de como se veria un archivo JSON correctamente completado dadas ciertas consultas particulares donde el JSON respuesta es 'No': 

CONSULTAS: 
    - 'quiero saber que cambios va a haber con el titulo automotor'
    - 'que va a pasar con aerolineas argentinas'
    - 'cuales son todos los articulos sobre la reforma del estado?'

JSON CORRECTO A TODAS ESTAS CONSULTAS:
{
    'pregunta_especifica': 'No', 
}

PRESTA ATENCION A LA ULTIMA CONSULTA del caso 'No': "cuales son todos los articulos sobre la reforma del estado". Si bien la consulta incluye la palabra "articulo", no incluye ningun numero que indique a que articulo se refiere. Por lo tanto, el JSON respuesta es 'No'.


A continuación EJEMPLOS de como se veria un archivo JSON correctamente completado dadas ciertas consultas particulares donde el JSON respuesta es 'Si': 
CONSULTAS: 
    - 'cuales son todos los articulos del Titulo III?',
    - 'cuales son todos los articulos del Titulo 3?',
    - 'que dice el articulo 870'

JSON CORRECTO A TODAS ESTAS CONSULTAS:
{
    'pregunta_especifica': 'Si', 
}

PRESTA ATENCION A LA INCLUSION DE NUMEROS ROMANOS O ARABIGOS en la consulta. 


Por favor, no uses la informacion de los EJEMPLOS, es solo para que entiendas el formato. La informacion que contiene es ficticia, NO LA USES.
Manten siempre el formato propuesto en los EJEMPLOS.
No hagas asumpciones sobre el campo 'pregunta_especifica', solo completa el campo con 'Si' o 'No' basado en PENSAMIENTO CRITICO y PASO A PASO sobre la consulta. 

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

#output_json = extract_json(interpretative_completion.choices[0].message.content)
interpretative_completion.choices[0].message.content


# %%
query = "quiero saber todos los cambios que va a haber en el sector de deportes?"
model = "gpt-3.5-turbo-1106"
output_json = get_query_specificity(query, model)
print(output_json)

# %%
for query in queries: 
    output = get_query_specificity(query, model)
    print(f"Query: {query}")
    print(f"Output: {output}")
    print("\n")

# %% [markdown]
# NOTE:
#     - Since we are doint zero-shot prompting, the testing will not be very generalizable. We need to build more queries to test how well this works. Nonetheless, we are leaving 2 queries for each case (yes/no) that are not used for zero-shot prompting, thus these results are generalizable. 

# %% [markdown]
# Now we will pipe the `get_query_specificity()` function with the `get_legal_info()` function. 

# %%
for query in queries:
    output = get_query_specificity(query, model)
    if output['pregunta_especifica'] == 'Si': 
        print("------------------")
        print(f"Query: {query}")
        print("La pregunta es ESPECÍFICA, lanzando llm que extrae campos JSON...")
        output_json = get_legal_info(query, model)
        print("El JSON extraido es:")
        print(output_json)
    if output['pregunta_especifica'] == 'No': 
        print("------------------")
        print(f"Query: {query}")
        print("La pregunta es GENERAL, no se lanza el `get_legal_info` llm.\n Iniciando el vector retrieval por cosine similarity...")
        # vector retrieval done downstream

# %% [markdown]
# Note that when the question is about two articles, a single json field is completed ('articulo': 'Articulo 30 y Articulo 35'). This should either be handled with the LLM (to make two json fields in this case), or with the fuzzy search. 

# %% [markdown]
# Now we will make a function that does some fuzzy search to retrieve the correct vectors

# %%
import re
from typing import Dict, List

def strip_tildes(old: str) -> str:
    """
    Removes common tildes from characters, lower form.
    """

    new = old.lower()
    new = re.sub(r'[àáâãäå]', 'a', new)
    new = re.sub(r'[èéêë]', 'e', new)
    new = re.sub(r'[ìíîï]', 'i', new)
    new = re.sub(r'[òóôõö]', 'o', new)
    new = re.sub(r'[ùúûü]', 'u', new)
    return new

def find_matching_keys(json_fields: Dict[str, str], legal_data: Dict[str, Any]) -> List[str]:
    """
    Performs a case-insensitive, accent-insensitive string matching search through the keys of a dictionary.

    Args:
        json_fields: A dictionary that represents the JSON fields.
        legal_data: A dictionary that represents the legal data.

    Returns:
        A list of keys from the legal data dictionary that match the fields in the JSON object.
    """
    matching_keys = []

    for key in legal_data.keys():
        if not key.startswith('query:'):
            stripped_key = strip_tildes(key)
            for field_name, field_value in json_fields.items():
                if field_value:  # if field_value is not empty
                    stripped_field_value = strip_tildes(field_value)
                    # Check if the field is in the key, but also if it's followed by a space or the end of the string
                    if re.search(f"{re.escape(stripped_field_value)}($|\s)", stripped_key):
                        matching_keys.append(key)

    return matching_keys


# %%
def strip_tildes(old: str) -> str:
    """
    Removes common tildes from characters, lower form.
    """

    new = old.lower()
    new = re.sub(r'[àáâãäå]', 'a', new)
    new = re.sub(r'[èéêë]', 'e', new)
    new = re.sub(r'[ìíîï]', 'i', new)
    new = re.sub(r'[òóôõö]', 'o', new)
    new = re.sub(r'[ùúûü]', 'u', new)
    return new


def roman_to_arabic(roman: str) -> str:
    """
    Converts a Roman numeral to an Arabic numeral.

    Args:
        roman: A string that represents a Roman numeral.

    Returns:
        A string that represents an Arabic numeral.
    """
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    arabic = 0
    for i in range(len(roman)):
        if i > 0 and roman_numerals[roman[i]] > roman_numerals[roman[i - 1]]:
            arabic += roman_numerals[roman[i]] - 2 * roman_numerals[roman[i - 1]]
        else:
            arabic += roman_numerals[roman[i]]
    return str(arabic)

def arabic_to_roman(arabic: int) -> str:
    """
    Converts an Arabic numeral to a Roman numeral.

    Args:
        arabic: An integer that represents an Arabic numeral.

    Returns:
        A string that represents a Roman numeral.
    """
    arabic_to_roman_numerals = {1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL', 50: 'L', 90: 'XC', 100: 'C', 400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'}
    roman = ''
    for key in sorted(arabic_to_roman_numerals.keys(), reverse=True):
        while arabic >= key:
            roman += arabic_to_roman_numerals[key]
            arabic -= key
    return roman

def find_matching_keys(json_fields: Dict[str, str], legal_data: Dict[str, Any]) -> List[str]:
    """
    Performs a case-insensitive, accent-insensitive string matching search through the keys of a dictionary.

    Args:
        json_fields: A dictionary that represents the JSON fields.
        legal_data: A dictionary that represents the legal data.

    Returns:
        A list of keys from the legal data dictionary that match the fields in the JSON object.
    """
    matching_keys = []

    for key in legal_data.keys():
        if not key.startswith('query:'):
            stripped_key = strip_tildes(key)
            for field_name, field_value in json_fields.items():
                if field_value:  # if field_value is not empty
                    stripped_field_values = [strip_tildes(value.strip()) for value in field_value.split(',')]
                    for stripped_field_value in stripped_field_values:
                        # Convert Roman numerals to Arabic numerals in the 'titulo' field
                        if field_name == 'titulo' and stripped_field_value.isalpha():
                            stripped_field_value = roman_to_arabic(stripped_field_value)
                        # Check if the field is in the key, but also if it's followed by a space or the end of the string
                        if re.search(f"{re.escape(stripped_field_value)}($|\s)", stripped_key):
                            matching_keys.append(key)

    return matching_keys


# %%
def find_matching_keys(json_fields: Dict[str, str], legal_data: Dict[str, Any]) -> List[str]:
    """
    Performs a case-insensitive, accent-insensitive string matching search through the keys of a dictionary.

    Args:
        json_fields: A dictionary that represents the JSON fields.
        legal_data: A dictionary that represents the legal data.

    Returns:
        A list of keys from the legal data dictionary that match the fields in the JSON object.
    """
    matching_keys = []

    for key in legal_data.keys():
        if not key.startswith('query:'):
            stripped_key = strip_tildes(key)
            for field_name, field_value in json_fields.items():
                if field_value:  # if field_value is not empty
                    stripped_field_values = [strip_tildes(value.strip()) for value in field_value.split(',')]
                    for stripped_field_value in stripped_field_values:
                        # Convert Roman numerals to Arabic numerals in the 'titulo' field
                        if field_name == 'titulo' and stripped_field_value.isalpha():
                            stripped_field_value = roman_to_arabic(stripped_field_value)
                        # Check if the field is in the key, but also if it's followed by a space or the end of the string
                        if re.search(f"{re.escape(stripped_field_value)}", stripped_key):
                            matching_keys.append(key)

    return matching_keys


# %%

def find_matching_keys(json_fields: Dict[str, str], legal_data: Dict[str, Any]) -> List[str]:
    """
    Performs a case-insensitive, accent-insensitive string matching search through the keys of a dictionary.

    Args:
        json_fields: A dictionary that represents the JSON fields.
        legal_data: A dictionary that represents the legal data.

    Returns:
        A list of keys from the legal data dictionary that match the fields in the JSON object.
    """
    matching_keys = []

    for key in legal_data.keys():
        if not key.startswith('query:'):
            stripped_key = strip_tildes(key)
            for field_name, field_value in json_fields.items():
                if field_value:  # if field_value is not empty
                    stripped_field_values = [strip_tildes(value.strip()) for value in field_value.split(',')]
                    for stripped_field_value in stripped_field_values:
                        # Convert Roman numerals to Arabic numerals in the 'titulo' field
                        if field_name == 'titulo' and stripped_field_value.isalpha():
                            stripped_field_value = roman_to_arabic(stripped_field_value)
                        # Check if the field is in the key, but also if it's followed by a space or the end of the string
                        if re.search(f"{re.escape(stripped_field_value)}", stripped_key):
                            matching_keys.append(key)

    return matching_keys


# %%
def find_matching_keys(json_fields: Dict[str, str], legal_data: Dict[str, Any]) -> List[str]:
    """
    Performs a case-insensitive, accent-insensitive string matching search through the keys of a dictionary.

    Args:
        json_fields: A dictionary that represents the JSON fields.
        legal_data: A dictionary that represents the legal data.

    Returns:
        A list of keys from the legal data dictionary that match the fields in the JSON object.
    """
    matching_keys = []

    for key in legal_data.keys():
        if not key.startswith('query:'):
            stripped_key = strip_tildes(key)
            for field_name, field_value in json_fields.items():
                if field_value and isinstance(field_value, str):  # if field_value is not empty and is a string
                    stripped_field_values = [strip_tildes(value.strip()) for value in field_value.split(',')]
                    for stripped_field_value in stripped_field_values:
                        # Convert Arabic numerals to Roman numerals in the 'titulo' field
                        if field_name == 'titulo' and stripped_field_value.isdigit():
                            stripped_field_value = arabic_to_roman(int(stripped_field_value))
                        # Check if the field is in the key, but also if it's followed by a space or the end of the string
                        if re.search(f"{re.escape(stripped_field_value)}", stripped_key):
                            matching_keys.append(key)

    return matching_keys


# %%
for field_name, field_value in output_json.items():
    print(f"field name is {field_name}")
    print(f"field value is {field_value}")

# %%
matching_keys = find_matching_keys(output_json, vectorstore_with_queries)
print(output_json), print(matching_keys)

# %%
queries

# %%
for query in queries:
#for query in queries[5:]:
    output = get_query_specificity(query, model)
    if output['pregunta_especifica'] == 'Si': 
        print("------------------")
        print(f"Query: {query}")
        print("La pregunta es ESPECÍFICA, lanzando llm que extrae campos JSON...")
        output_json = get_legal_info(query, model)
        print("El JSON extraido es:")
        print(output_json)
        matching_keys = find_matching_keys(output_json, vectorstore_with_queries)
        print(f"Las keys de los datos legales que matchean con el JSON extraido son: {matching_keys}")
        print(f"Iniciando el vector retrieval por fuzzy search...")
    if output['pregunta_especifica'] == 'No': 
        print("------------------")
        print(f"Query: {query}")
        print("La pregunta es GENERAL, no se lanza el `get_legal_info` llm.\n Iniciando el vector retrieval por cosine similarity...")
        # vector retrieval done downstream


# %% [markdown]
#

# %% [markdown]
# This works pretty well, nonetheless we will run some tests. 
#
# Note: maybe we could use an llm to replace the fuzzy search or Maybe do mapping between the original non-flattened format and the flattened one to solve the fuzzy search?

# %% [markdown]
# To ensure that the fuzzy search function is robust and generalizable, let's consider some edge cases and write tests for them.
#
# Here are some potential edge cases:
#
# 1. The JSON fields are empty.
# 2. The JSON fields contain only spaces.
# 3. The JSON fields contain special characters.
# 4. The JSON fields contain numbers. # check
# 5. The legal data dictionary is empty.
# 6. The legal data dictionary contains keys that don't match the expected format.

# %%
def test_find_matching_keys():
    # Test when JSON fields are empty
    json_fields = {'titulo': '', 'capitulo': '', 'articulo': ''}
    legal_data = {'Título I: Articulo 1': 'content1', 'Título II: Articulo 2': 'content2'}
    assert find_matching_keys(json_fields, legal_data) == []

    # Test when JSON fields contain only spaces
    json_fields = {'titulo': ' ', 'capitulo': ' ', 'articulo': ' '}
    assert find_matching_keys(json_fields, legal_data) == []

    # Test when JSON fields contain special characters
    json_fields = {'titulo': 'Título $', 'capitulo': 'Capítulo &', 'articulo': 'Articulo #'}
    assert find_matching_keys(json_fields, legal_data) == []

    """ToDo: CHECK THIS ONE"""
    # Test when JSON fields contain numbers
    json_fields = {'titulo': 'Título 1', 'capitulo': 'Capítulo 2', 'articulo': 'Articulo 3'}
    assert find_matching_keys(json_fields, legal_data) == []
    """END CHECK"""

    # Test when legal data dictionary is empty
    json_fields = {'titulo': 'Título I', 'capitulo': '', 'articulo': 'Articulo 1'}
    legal_data = {}
    assert find_matching_keys(json_fields, legal_data) == []

    # Test when legal data dictionary contains keys that don't match the expected format
    json_fields = {'titulo': 'Título I', 'capitulo': '', 'articulo': 'Articulo 1'}
    legal_data = {'Key 1': 'content1', 'Key 2': 'content2'}
    assert find_matching_keys(json_fields, legal_data) == []

    # Test when there's a match
    json_fields = {'titulo': 'Título I', 'capitulo': '', 'articulo': 'Articulo 1'}
    legal_data = {'Título I: Articulo 1': 'content1', 'Título II: Articulo 2': 'content2'}
    assert find_matching_keys(json_fields, legal_data) == ['Título I: Articulo 1']

    # ToDo: Test when it is asked for 2+ specific documents


    print("All tests passed!")


test_find_matching_keys()

# %% [markdown]
# Ok, now that `find_matching_keys()` is sortof tested, we will include it in the cosineSimilarity-fuzzySearch-retrieval-pipeline. 

# %%
for query in queries:
    output = get_query_specificity(query, model)
    if output['pregunta_especifica'] == 'Si': 
        print("------------------")
        print(f"Query: {query}")
        print("La pregunta es ESPECÍFICA, lanzando llm que extrae campos JSON...")
        output_json = get_legal_info(query, model)
        print("El JSON extraido es:")
        print(output_json)
        print("Iniciando vector fuzzy search...")
        matching_keys = find_matching_keys(output_json, vectorstore_with_queries)
        print("Los documentos que matchean son:")
        print(matching_keys)
    if output['pregunta_especifica'] == 'No': 
        print("------------------")
        print(f"Query: {query}")
        print("La pregunta es GENERAL, no se lanza el `get_legal_info` llm.")
        print("Iniciando el vector retrieval por cosine similarity...")
        # vector retrieval done downstream


# %% [markdown]
# Ok, now we will merge everything. That is, we will merge the vector retriever based on cosine similarity for general questions with the vector retriever based on fuzzy search for specific questions. Then, we will evaluate the retrieval by looking at it in the 2-d t-sne projected space. 
#

# %% [markdown]
# roman_to_arabic() screwd everything up. CHECK --> Specify to cursor all the ways in which the json could be filled and then ask it to make a robust fuzzy search. 
#

# %% [markdown]
# ToDo: check if tildes have to be handled in LLM prompt (but remember ended up removing all tildes from the keys of the legal data. )

# %% [markdown]
# To define: If the query is specific, we do a fuzzy search. But should we do some cosine similarity with a low `top_k` to see if it adds any relevant information? (although we know that the vector similarity for these types of queries is very bad as of now.)

# %%

# %% [markdown]
# Doing a comprehensive fuzzy search is quite problematic. 
# We will instead ask an LLM to extract a json with the keys of each key of the legal data (instead of doing fuzzy search), and we will bind them

# %%
