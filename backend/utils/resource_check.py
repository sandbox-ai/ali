import os
import psutil
from sentence_transformers import SentenceTransformer, util

"""
This script measures the memory usage of a SentenceTransformer model during loading and inference. 

The memory usage is measured in two stages:
1. Loading the model: The script calculates the memory used to load a SentenceTransformer model.
2. Inference: The script calculates the memory used to perform inference using the loaded model.

The memory usage is measured in megabytes (MB) using the psutil library.

The script uses a pre-trained SentenceTransformer model to encode a corpus of text and a query. It then performs a semantic search on the corpus using the query and prints the top 2 results.

This script is useful for understanding the memory footprint of SentenceTransformer models during loading and inference.

ToDO: 
- use llama_index embeddings instead of SentenceTransformer: 
    -   from llama_index.embeddings import HuggingFaceEmbedding
"""

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # convert bytes to megabytes

# Before loading the model
mem_before = get_memory_usage()

# Load your model
model = SentenceTransformer('dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn')

# After loading the model
mem_after = get_memory_usage()

print(f'Memory used to load model (in MB): {mem_after - mem_before}')

# Some examples that may contain information that is relevant to your question
corpus = [
    "Napoleón I Bonaparte (Ajaccio, 15 de agosto de 1769-Longwood, 5 de mayo de 1821) fue un militar y estadista francés, general republicano durante la Revolución francesa y el Directorio, y artífice del golpe de Estado del 18 de brumario que lo convirtió en primer cónsul (Premier Consul) de la República el 11 de noviembre de 1799.",
    "Luis XVI de Francia (en francés: Louis XVI; Versalles, 23 de agosto de 1754 – París, 21 de enero de 1793) fue rey de Francia y de Navarra4 entre 1774 y 1789, copríncipe de Andorra entre 1774 y 1793, y rey de los franceses3 entre 1789 y 1792.2 Fue el último monarca antes de la caída de la monarquía por la Revolución Francesa, así como el último que ejerció sus poderes de monarca absoluto.",
    "Felipe VI de España (Madrid, 30 de enero de 1968) es el actual rey de España, título por el que ostenta la jefatura del Estado y el mando supremo de las Fuerzas Armadas, desde el 19 de junio de 2014, fecha en que ascendió al trono por la abdicación de su padre, el rey Juan Carlos I.",
    "Lionel Andrés Messi Cuccittini (Rosario, 24 de junio de 1987), conocido como Leo Messi, es un futbolista argentino que juega como delantero o centrocampista. Jugador histórico del Fútbol Club Barcelona, al que estuvo ligado veinte años, desde 2021 integra el plantel del Paris Saint-Germain de la Ligue 1 de Francia. Es también internacional con la selección de Argentina, equipo del que es capitán."
]

# Your question
query = "Listar aquellos personajes que tuvieron poder en Francia"

# Before running the inference
mem_before_inference = get_memory_usage()

# Encode corpus and query
corpus_embeddings = model.encode(corpus)
query_embedding = model.encode(query)

# Get the 2 best results on the corpus options
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=2)[0]
for hit in hits:
    print(f"corpus_id: {hit['corpus_id']}, score: {hit['score']}, text: {corpus[hit['corpus_id']][0:100]}...")

# After running the inference
mem_after_inference = get_memory_usage()

print(f'Memory used for inference (in MB): {mem_after_inference - mem_before_inference}')