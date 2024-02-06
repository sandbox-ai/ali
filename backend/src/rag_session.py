import os
import json
from typing import Dict, Any, Union, Tuple, Optional, List
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from openai import OpenAI
import time
import logging
import uuid
from src.utils import *
import copy

"""
system will fail if: 
- retrieved chunks (with different lengths) exceed context length. Need to handle this (something like retrieve chunks until max n_chars is reached)
- raw prompt (without a template) that we are using now is not the best. Check justicio and the result of the prompt template of llama_index, AND CHECK OPENAI GUIDES ON PROMPTS
https://platform.openai.com/docs/guides/prompt-engineering
There's also going to be a problem with citations that are not used (retrieved chunks that the llm won't use either because it's not useful or because the attention is lacking). maybe add numbers to each retrieved chunk and ask another llm to tell which ones were used by the answer-llm and use that to filter out the unused citations?


Prompt Stuff: 
- "explicas de donde viene la info"????

ToDo:
- use a logger? is this needed or just helpful? justicio did it and pablito did something with logger but I don't know what. 
- determine top_k based on the top number of queries and the average number of articles per title or chapter. 
- make the embedding of the vectorstore run only if vectorstore is not locally (where it's supposed to be)
- instructor with refine to include modified laws
- Use FAISS to speed up cosine similarity through the vectorstore. "Efficient Data Structures: Use efficient data structures like KD-trees or approximate nearest neighbor (ANN) algorithms (e.g., FAISS, Annoy) for faster similarity searches. These can significantly reduce the search space and time."
"""


def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print(f"{method.__name__} took {end_time - start_time} seconds to execute")
        return result

    return timed


class DataLoader:
    @staticmethod
    def load_json(filepath: str) -> Dict:
        with open(filepath, "r") as file:
            return json.load(file)


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_text(
        self, data: Union[str, Dict[str, str]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(data, dict):
            return {key: self.model.encode(value) for key, value in data.items()}
            logging.info(f"Embedding {len(data)} documents")
        elif isinstance(data, str):
            return self.model.encode(data)
            logging.info(f"Embedding {len(data)} documents")
        else:
            raise ValueError("Input must be either str or dict")


class NdarrayEncoder(json.JSONEncoder):
    """
    Converts ndarrays in dictionary to lists.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)


class NdarrayDecoder:
    def __call__(self, dct):
        """
        Converts lists in the dictionary back to numpy ndarrays.
        """
        for key, value in dct.items():
            if isinstance(value, list):
                dct[key] = np.array(value)
        return dct


class VectorStoreManager:
    @staticmethod
    def save_vectorstore(filepath: str, vectorstore: Dict[str, np.ndarray]):
        with open(filepath, "w") as file:
            json.dump(vectorstore, file, cls=NdarrayEncoder)
        logging.info(f"Vectorstore saved to {filepath}")

    @staticmethod
    def read_vectorstore(filepath: str) -> Dict[str, np.ndarray]:
        ndarray_decoder = NdarrayDecoder()

        logging.info(f"Reading vectorstore from {filepath}")
        with open(filepath, "r") as file:
            return json.load(file, object_hook=ndarray_decoder)


class PipelineTester:
    @staticmethod
    # Check that encoding/decoding of vectorstore's ndarrays is lossless
    def verify_data_integrity(original_data: Dict[str, np.ndarray]) -> None:
        """
        Tests the integrity of data through encoding to JSON and decoding back to a dictionary.

        This function encodes a dictionary (where values are numpy ndarrays) into a JSON string,
        then decodes this JSON string back into a dictionary. It checks that the keys match between
        the original and decoded dictionaries and that the values (ndarrays) are equal in type and content.

        Args:
            original_data: A dictionary with string keys and numpy ndarray values.

        Raises:
            AssertionError: If the keys do not match between the original and decoded dictionaries,
                            or if the ndarray values are not equal in type and content.
        """
        # Encode to JSON string
        encoded_data = json.dumps(original_data, cls=NdarrayEncoder)

        # Decode back to dictionary
        ndarray_decoder = NdarrayDecoder()
        decoded_data = json.loads(encoded_data, object_hook=ndarray_decoder)

        # Check if keys match
        assert set(original_data.keys()) == set(
            decoded_data.keys()
        ), "Keys do not match."
        # Check if values match in type and content
        for key in original_data:
            assert isinstance(
                decoded_data[key], np.ndarray
            ), f"Value for {key} is not an ndarray."
            np.testing.assert_array_equal(
                original_data[key],
                decoded_data[key],
                err_msg=f"Arrays for {key} do not match.",
            )

        print("Test passed: No modification in data during encoding and decoding.")

    @staticmethod
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
        query_vectors_2d = np.array(
            [vec for key, vec in zip(keys_list, vectors_2d) if key.startswith("query:")]
        )
        non_query_vectors_2d = np.array(
            [
                vec
                for key, vec in zip(keys_list, vectors_2d)
                if not key.startswith("query:")
            ]
        )
        query_keys = [key for key in keys_list if key.startswith("query:")]
        non_query_keys = [key for key in keys_list if not key.startswith("query:")]

        # Create the plot for non-query vectors
        fig = go.Figure(
            data=go.Scatter(
                x=non_query_vectors_2d[:, 0],
                y=non_query_vectors_2d[:, 1],
                mode="markers",
                text=non_query_keys,  # This line sets the hover text
                marker=dict(
                    size=8,
                    color=non_query_vectors_2d[:, 0],  # Set color equal to x
                    colorscale="Viridis",  # One of plotly colorscales
                    showscale=False,
                ),
            )
        )

        # Add the plot for query vectors, if there are any
        if len(query_vectors_2d) > 0:
            fig.add_trace(
                go.Scatter(
                    x=query_vectors_2d[:, 0],
                    y=query_vectors_2d[:, 1],
                    mode="markers",
                    text=query_keys,  # This line sets the hover text
                    marker=dict(
                        size=8,
                        color="black",  # Set color to black
                        symbol="star",  # Set marker symbol to star
                    ),
                )
            )

        # Set the title and labels
        fig.update_layout(
            title=f"t-SNE plot of vectors",
            xaxis=dict(title="Dimension 1"),
            yaxis=dict(title="Dimension 2"),
        )

        # Show the plot
        fig.show()

        fig.write_html("t-SNE_plot.html", auto_open=True)


class QueryEngine:
    def __init__(
        self,
        vectorstore: Dict[str, np.ndarray],
        embedder: SentenceTransformer,
        legal_docs: Dict[str, str],
        top_k: Optional[int] = 5,
    ):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.legal_docs = legal_docs
        self.top_k = top_k
        # Precompute norms of vectors to speed up vector search
        self.vector_norms = {
            key: np.linalg.norm(vector) for key, vector in vectorstore.items()
        }

    @timeit
    def query_similarity(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        if top_k is None:
            top_k = self.top_k
        query_vector = self.embedder.embed_text(query)

        docs_with_scores = {}
        # Compute the cosine similarity between the query vector and all vectors in the vectorstore
        for key, vector in self.vectorstore.items():
            similarity = cosine_similarity(
                query_vector.reshape(1, -1), vector.reshape(1, -1)
            )
            docs_with_scores[key] = similarity[0][0]

        # Sort the similarities in descending order
        sorted_docs_with_scores = sorted(
            docs_with_scores.items(), key=lambda x: x[1], reverse=True
        )

        top_k_docs = dict(sorted_docs_with_scores[:top_k])

        # Select matching legal docs
        matching_docs = {
            key: self.legal_docs[key]
            for key in top_k_docs.keys()
            if key in self.legal_docs
        }

        logging.info("Computing vectors similarities")

        return top_k_docs, matching_docs

    @timeit
    def fast_query_similarity(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        if top_k is None:
            top_k = self.top_k
        query_vector = self.embedder.embed_text(query).reshape(1, -1)
        vector_matrix = np.array(list(self.vectorstore.values()))
        query_norm = np.linalg.norm(query_vector)

        # Compute cosine similarity using dot product and precomputed norms
        dot_products = np.dot(vector_matrix, query_vector.T).flatten()
        norms = np.array(list(self.vector_norms.values()))
        similarities = dot_products / (norms * query_norm)

        # Get top_k results
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        top_k_keys = np.array(list(self.vectorstore.keys()))[top_k_indices]
        top_k_docs = {key: similarities[idx] for idx, key in enumerate(top_k_keys)}
        matching_docs = {
            key: self.legal_docs[key] for key in top_k_keys if key in self.legal_docs
        }

        logging.info("Computing vectors similarities")

        return top_k_docs, matching_docs

    @timeit
    def generate_llm_response(
        self,
        query: str,
        client,
        model_name: str = "gpt-4-0125-preview",
        temperature: float = 0,
        max_tokens: int = 2000,
        streaming: bool = True,
        top_k_docs: Optional[Dict[str, float]] = None,
        matching_docs: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Generates a response from a legal assistant model using OpenAI's GPT.

        Args:
            query: The user's query to respond to.
            client: An instance of the OpenAI client.
            model_name: The name of the model to use for generating responses.
            temperature: The temperature to use for the response generation.
            max_tokens: The maximum number of tokens to generate.
            streaming: Whether to stream the response or not.

        Returns:
            None. Prints the response directly if streaming is True, otherwise returns the response.
        """
        if top_k_docs is None or matching_docs is None:
            print("No top_k_docs or matching_docs provided. Computing them now.")
            top_k_docs, matching_docs = self.query_similarity(query)

        # Preprocess the context from matching_docs
        context_preprocessed = {
            matching_docs[key]: top_k_docs[key] for key in matching_docs.keys()
        }

        # Construct the messages for the chat
        messages = [
            {
                "role": "system",
                "content": "Sos un asistente legal útil que reponde SIN EMITIR JUICIO DE SI UN CAMBIO ES BUENO O MALO, MEJOR O PEOR. Al dar tu respuesta, tenes que tener en cuenta y utilizar el contexto proporcionado para dar una respuesta INTEGRAL, INFORMATIVA, PRECISA y OBJETIVA a la pregunta del usuario.",
            },
            {"role": "system", "content": "A continuación se proporciona el contexto:"},
            {"role": "system", "content": str(context_preprocessed)},
            {
                "role": "system",
                "content": "A continuación se proporciona la pregunta del usuario:",
            },
            {"role": "user", "content": query},
        ]

        # Generate the response using OpenAI
        if not streaming:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return completion.choices[0].message.content
        else:
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")

    @timeit
    def generate_metadata_from_key(self, key: str) -> dict:
        """
        Extracts metadata from a given key string. The metadata includes 'documento',
        'titulo', 'capitulo', and 'articulo', derived from the structure of the key.

        Parameters:
        - key (str): The key string from which to extract metadata. It is expected to
        contain sections separated by periods and specific markers like "Titulo",
        "Capitulo", and "Articulo".

        Returns:
        - dict: A dictionary containing the extracted metadata. The dictionary has the
        following keys: 'documento', 'titulo', 'capitulo', 'articulo', 'start_idx',
        and 'end_idx'. The 'start_idx' and 'end_idx' are included for compatibility
        but are set to 0 as they are ignored in this context.
        """
        # Initialize metadata dictionary with default values
        metadata = {
            'documento': '',
            'titulo': '',
            'capitulo': '',
            'articulo': '',
            'start_idx': 0,  
            'end_idx': 0     
        }
        
        # Extract 'documento' from the beginning until the first period
        documento_end_idx = key.find(".")
        if documento_end_idx != -1:
            metadata['documento'] = key[:documento_end_idx]
        
        # Find indices for 'titulo', 'capitulo', and 'articulo'
        titulo_start_idx = key.find("Titulo")
        capitulo_start_idx = key.find("Capitulo")
        articulo_start_idx = key.find("Articulo")
        
        # Determine the end index for 'titulo' based on the presence of 'capitulo' or 'articulo'
        titulo_end_idx = min(
            [idx for idx in [capitulo_start_idx, articulo_start_idx, len(key)] if idx != -1]
        )
        
        if titulo_start_idx != -1 and titulo_end_idx != -1:
            metadata['titulo'] = key[titulo_start_idx:titulo_end_idx].strip(" .")
        
        # Extract 'capitulo' if present
        if capitulo_start_idx != -1:
            capitulo_end_idx = articulo_start_idx if articulo_start_idx != -1 else len(key)
            metadata['capitulo'] = key[capitulo_start_idx:capitulo_end_idx].strip(" .")
        
        # Extract 'articulo'
        if articulo_start_idx != -1:
            metadata['articulo'] = key[articulo_start_idx:]
        
        return metadata


    @timeit
    def generate_complete_citations_dict(self, matching_docs: Dict[str, str], top_k_docs: Dict[str, float]) -> Dict[str, Dict]:
        """
        Generates a dictionary of dictionaries, each containing text, score, and metadata extracted from the keys of matching_docs and the scores from top_k_docs. Each entry is uniquely identified by a randomly generated UUID.

        Parameters:
        - matching_docs (Dict[str, str]): A dictionary where keys are document identifiers and values are the text of the documents.
        - top_k_docs (Dict[str, float]): A dictionary where keys are document identifiers and values are the scores of the documents.

        Returns:
        - Dict[str, Dict]: A dictionary where each key is a UUID string and each value is a dictionary containing 'text', 'score', 'metadata', 'start_char_idx', and 'end_char_idx'.
        """
        complete_dicts = {}

        for key, text in matching_docs.items():
            metadata = self.generate_metadata_from_key(key)
            score = top_k_docs.get(key, 0)  # Default score to 0 if key not found in top_k_docs
            item_id = str(uuid.uuid4())  # Generate a random UUID
            complete_dicts[item_id] = {
                'text': text,
                'score': score,
                'metadata': metadata,
                'start_char_idx': None,  
                'end_char_idx': None     
            }

        return complete_dicts
    
    # NEEDS REWRITING. Esta dos veces la func para tenerla afuera de la clase.
def generate_metadata_from_key(key: str) -> dict:
    """
    Extracts metadata from a given key string. The metadata includes 'documento',
    'titulo', 'capitulo', and 'articulo', derived from the structure of the key.

    Parameters:
    - key (str): The key string from which to extract metadata. It is expected to
    contain sections separated by periods and specific markers like "Titulo",
    "Capitulo", and "Articulo".

    Returns:
    - dict: A dictionary containing the extracted metadata. The dictionary has the
    following keys: 'documento', 'titulo', 'capitulo', 'articulo', 'start_idx',
    and 'end_idx'. The 'start_idx' and 'end_idx' are included for compatibility
    but are set to 0 as they are ignored in this context.
    """
    # Initialize metadata dictionary with default values
    metadata = {
        'documento': '',
        'titulo': '',
        'capitulo': '',
        'articulo': '',
        'start_idx': 0,  
        'end_idx': 0     
    }
        
    # Extract 'documento' from the beginning until the first period
    documento_end_idx = key.find(".")
    if documento_end_idx != -1:
        metadata['documento'] = key[:documento_end_idx]
        
    # Find indices for 'titulo', 'capitulo', and 'articulo'
    titulo_start_idx = key.find("Titulo")
    capitulo_start_idx = key.find("Capitulo")
    articulo_start_idx = key.find("Articulo")
        
    # Determine the end index for 'titulo' based on the presence of 'capitulo' or 'articulo'
    titulo_end_idx = min(
        [idx for idx in [capitulo_start_idx, articulo_start_idx, len(key)] if idx != -1]
    )
        
    if titulo_start_idx != -1 and titulo_end_idx != -1:
        metadata['titulo'] = key[titulo_start_idx:titulo_end_idx].strip(" .")
        
    # Extract 'capitulo' if present
    if capitulo_start_idx != -1:
        capitulo_end_idx = articulo_start_idx if articulo_start_idx != -1 else len(key)
        metadata['capitulo'] = key[capitulo_start_idx:capitulo_end_idx].strip(" .")
        
    # Extract 'articulo'
    if articulo_start_idx != -1:
        metadata['articulo'] = key[articulo_start_idx:]
        
    return metadata


if __name__ == "__main__":
    """   
    ========================================
    ||            example run             ||
    ========================================   
    """

    file_path_dnu = "./data/LaLeyDeMilei-raw/decreto_flat.json"
    file_path_dnu_unpreppended = (
        "./data/LaLeyDeMilei-raw/decreto_flat_unpreppended.json"
    )
    file_path_vectorstore = "./data/dnu_vectorstore.json"

    query = "que va a pasar con los impuestos de los autos"

    data_loader = DataLoader()
    dnu = data_loader.load_json("./data/LaLeyDeMilei-raw/decreto_flat.json")
    dnu_unpreppended = data_loader.load_json(
        "./data/LaLeyDeMilei-raw/decreto_flat_unpreppended.json"
    )

    embedder = Embedder("dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn")

    if not os.path.exists(file_path_vectorstore):
        vectorstore = embedder.embed_text(dnu)
        VectorStoreManager.save_vectorstore(file_path_vectorstore, vectorstore)
    else:
        vectorstore = VectorStoreManager.read_vectorstore(file_path_vectorstore)

    query_vector = embedder.embed_text(query)

    # Initialize the QueryEngine with necessary parameters
    query_engine = QueryEngine(vectorstore, embedder, legal_docs=dnu, top_k=5)

    # Use the query_similarity method to find documents similar to the query
    top_k_docs, matching_docs = query_engine.query_similarity(
        query="que va a pasar con los impuestos de autos"
    )

    # Now, use the generate_llm_response method to generate a response based on the query and matching documents
    text = query_engine.generate_llm_response(
        query="que va a pasar con los impuestos de autos",
        client=OpenAI(),
        model_name= 'gpt-3.5-turbo-0125', #'gpt-4-1106-preview'# <-- ~15s, #'gpt-4' # <--- ~8s, #"gpt-4-0125-preview",# <--- slow AF, ~27s
        temperature=0,
        max_tokens=1000, # 2000
        streaming=False,
        top_k_docs=top_k_docs,
        matching_docs=matching_docs,
    )

    print(text)

    citations = query_engine.generate_complete_citations_dict(matching_docs, top_k_docs)

    print(citations)

    """   
    ========================================
    ||          END example run           ||
    ========================================   
    """


    """   
    ========================================
    ||       generate citations data      ||
    ========================================   
    """
    file_path_dnu = "./data/LaLeyDeMilei-raw/decreto_flat.json"
    data_loader = DataLoader()
    dnu = data_loader.load_json("./data/LaLeyDeMilei-raw/decreto_flat.json")
    dnu_unpreppended = data_loader.load_json("./data/LaLeyDeMilei-raw/decreto_flat_unpreppended.json")

    # generate citations
    def generate_citation_data(legal_doc) -> Dict[str, Dict[str, Dict]]:
        """
        UPDATE DOCSTRING
        Generates a dictionary of dictionaries, each containing text, score, and metadata extracted from the keys of matching_docs and the scores from top_k_docs. Each entry is uniquely identified by a randomly generated UUID.

        Parameters:
        - matching_docs (Dict[str, str]): A dictionary where keys are document identifiers and values are the text of the documents.
        - top_k_docs (Dict[str, float]): A dictionary where keys are document identifiers and values are the scores of the documents.

        Returns:
        - Dict[str, Dict]: A dictionary where each key is a UUID string and each value is a dictionary containing 'text', 'score', 'metadata', 'start_char_idx', and 'end_char_idx'.
        """

        dnu_metadata = {
            key: {
                str(uuid.uuid4()): {
                    'text': value,
                    'score': 0, 
                    'metadata': generate_metadata_from_key(key),
                    'start_char_idx': None,
                    'end_char_idx': None, 
                }
            } for key, value in dnu.items()
        }

        return dnu_metadata


    dnu_metadata = generate_citation_data(dnu)
    dnu_metadata

    with open('./data/dnu_metadata.json', 'w') as f:
        json.dump(dnu_metadata, f)

    with open('./data/dnu_metadata.json', 'r') as f:
        dnu_metadata = json.load(f)




    def get_cached_citations(with_score: bool = True)


    


    import copy
    # Initialize citations as an empty dict
    citations = {}
    # Iterate over top_k_docs to filter, flatten, and update scores
    for key in top_k_docs:
        if key in dnu_metadata:
            for sub_key, sub_value in dnu_metadata[key].items():
                # Deep copy the sub_value to ensure original data is not modified
                citations[sub_key] = copy.deepcopy(sub_value)
                # Update the score for the flattened entry
                citations[sub_key]['score'] = top_k_docs[key]
    


    head_dict(citations, 2)
    head_dict(dnu_metadata, 2)