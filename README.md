<h1 align="center"><strong>ALI</strong></h1>
<p align="center"> Asistente Legal Inteligente </p>

<div align="center">

![](https://img.shields.io/badge/huggingface-transformers-blue)
![](https://img.shields.io/badge/python-3.10-blue)
![](https://img.shields.io/badge/Angular-DD0031?style=flat&logo=Angular)
![](https://img.shields.io/badge/GenAI-blue)
![](https://img.shields.io/badge/Made%20with-Love-red)

</div>                 


<p align="center">
    <img src="./ali_logo.png" alt="readme_image" style="width:220px;height:220px;" />
</p>

This repo contains the code and instructions to set up ALI, a web UI and back-end pipeline for Retrieval Augmented Generation around legal documents.
You can try ALI [here](https://ali.sandboxai.ar/) .

## Table of Contents
- [Installation](#installation)
- [Description](#description)
  - [Overview](#overview)
  - [Technical information](#technical-information)
    - [General](#general)
    - [On LLM frameworks](#on-llm-frameworks)
    - [Main challenges encountered](#main-challenges-encountered)
    - [Improvements over baseline RAG.](#improvements-over-baseline-rag)
    - [Dataset](#dataset)
    - [Testing](#testing)
- [Acknowledgementes](#acknowledgements)
- [Contributing](#contributing)
- [Sponsoring the project](#sponsoring-the-project)
- [License](#license)

### Installation

Clone the repo, create a new environment and install backend and frontend requirements:
```bash
git clone https://github.com/sandbox-ai/ali.git
cd ali
conda create --name ali
conda activate ali
cd backend
pip install -r requirements.txt
cd ../frontend
npm install 
```

**Usage**

Launch the backend in one terminal:
```bash
conda activate ali
export OPENAI_API_KEY=<your-key-here>
# or if you have your own LLM backend
export OPENAI_API_BASE=<custom-endpoint>
cd backend
python api.py
```

Open a new terminal and launch the frontend: 
```bash
cd frontend
ng serve
```

Also see [`backend/README.md`](backend/README.md) and [`frontend/README.md`](frontend/README.md)    


## Description
### Overview

The complexity of legal documents and legalese terminology presents a barrier to most of the population, who won't able to interact and understand their own legal system unless aided by a professional in the field.

To help close this gap in the Spanish speaking world, we built the Asistente Legal Inteligente (or ALI). It uses Retrieval Augmented Generation ([RAG](https://arxiv.org/abs/2005.11401)), i.e. searching a vectorstore given an user query and formulating an answer with a Large Language Model (LLM), and a custom dataset that lays a RAG optimized structure. 

The result is a grounded and comprehensive assistant that can answer questions about general legislation and specific legal situations.

### Technical information
#### General
The user query is embedded using a custom Spanish [embedding model](https://huggingface.co/dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn), and then used to search for the best matching legal documents with cosine-similarity. To formulate the answer, an LLM hosted with an [OpenAI compatible API endpoint](https://platform.openai.com/docs/api-reference) is queried with a custom prompt and the relevant documents. 

This technique has ample room for improvements. See our roadmap on [RAG improvements](#improvements-over-baseline-rag). 
You can check out the RAG system written from scratch in [`src/rag_session.py`](src/rag_session.py)   

#### On LLM frameworks
We've tested both [llama_index](https://github.com/run-llama/llama_index) and [langchain](https://github.com/langchain-ai/langchain), but found them too restrictive and in the end more cumbersome than developing our own pipeline over [transformers](https://huggingface.co/docs/transformers/en/index), enabling finer control and suprevision. 

#### Main challenges encountered
The main problem we encountered with a RAG pipeline over Argentinian legal data was the embedding of the information. This problem has two parts: 

1. Embedding model:

    We tested OpenAI's embedding model (ada-002), and found that it wasn't great at distancing different legal topics and clustering similar ones (and it wasn't great in general for Spanish). Thus we opted for the custom embedding [model](https://huggingface.co/dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn) described earlier.

    **Notes:** Newer models that are topping the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) are worth testing. New OpenAI's [embbeding models](https://openai.com/blog/new-embedding-models-and-api-updates?ref=blog.salesforceairesearch.com)  have been released and should be tested. 

2. Legal Document Chunking/Parsing
    
    We found that naively chunking legal documents with a regular document chunker was sub-par, trimming and leaving out vital contextual information. We decided to define each legal article as our atomic unit and embbed it as a whole. 

When we tried to embed full articles as they were, distancing and clustering of vectors wasn't great. The trick that did it was to prepend all the contextual metadata to each article BEFORE embedding. So, each article to be embbeded would look like:
```
"Decreto de Necesidad y Urgencia N° DNU-2023-70-APN-PTE. Fecha 20-12-2023. Titulo II: DESREGULACIÓN ECONÓMICA. Capitulo II: Tarjetas de crédito (Ley N° 25.065). Articulo 15: La entidad emisora deberá obligatoriamente dar a conocer el público la [...]"
```
instead of just 
```
"Articulo 15: La entidad emisora deberá obligatoriamente dar a conocer el público la [...]".
``` 


All the results we found in relation to the embedding models were a direct conclusion of plotting the resulting embedding vectors with the dimensionality reduction technique [t-SNE: t-distributed Stochastic Neighbor Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). Note that there are alternatives such as [UMAP: Universal Manifold Approximation & Projection](https://github.com/lmcinnes/umap)

#### Improvements over baseline RAG. 
There are many improvements to be made over this baseline RAG. The following is a non-exhaustive list: 

- Query rewriting (hard without proper context of argentinean law, either parametric or non parametric), potentially connected with a Fusion retriever. 
- Document reranking
- Hybrid Search (keyword-based search algorithms combined with vector search)
- Linear adapter for the embedding model
- Fine-tuning over each legal domain
- Adding an agentic layer capable of handling complex questions through multiple hops of reasoning and retrieval. 


#### Dataset
For testing purposes, we include in the current repo a processed version of the [DNU 70/2023](https://www.argentina.gob.ar/normativa/nacional/decreto-70-2023-395521/texto), an extensive executive order that modifies a wide array of laws.

In order to create a vector database of the whole Argentinian law, we can look into the Boletín Oficial, where everything about legislation is published.

We have built [a tool to scrap](https://github.com/sandbox-ai/Boletin-Oficial-Argentina) the whole Boletín Oficial into a dataset. You can also find the result uploaded and up-to-date on [Huggingface](https://huggingface.co/datasets/marianbasti/boletin-oficial-argentina).

This raw dataset must be parsed into the format described earlier (prepending contextual metadata). Given the inconsistent and unpredictable formatting of the documents and texts, there is no simple programmatic parsing to automate the process. We found that various NLP techniques are useful in automating this task (prompting LLMs, sentence-transformers, NER). 

#### Testing
[Ragas](https://github.com/explodinggradients/ragas) is a framework to evaluate RAG pipelines that could be used to test ALI. It is important to acknowledge the costs using a paid API (we tested this with a relatively small document and GPT-4 and spent 40 USD in half an hour!)    

## Acknowledgements
Huge thanks to the [Justicio](https://github.com/bukosabino/justicio) team from Spain, who gave us a lot of tips and shared their embedding model with us
Definetly go check their project and talk to the creators!

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for a detailed explanation of branch/commit naming conventions

## Who we are
We are a group of Argentinian developers named [`sandbox.ai`](https://sandbox-ai.github.io/).
You can find us on [Twitter](https://twitter.com/sandboxaiorg) and [LinkedIn](https://www.linkedin.com/in/sandboxai-org-b0a7842b4/).
You can also contact us directly at `sandboxai.org@proton.me`.

## Sponsoring the project
As of now, ALI is running on AWS using our OpenAI API key. Both things are expensive and as of now these costs are being covered by the sandbox.ai devs. 
If you find this project useful and would like to sponsor us (either by covering the AWS costs or the OpenAI api key costs), please contact us at `sandboxai.org@proton.me`.

## License
GPL













