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


# Mission
Our mission is to democratize the access to legal information in Argentina. Legal information is [abundant](https://www.boletinoficial.gob.ar/) in Argentina, but it's very hard to search and digest for citizens that do not work in the legal domain. 
Our hope is that one day, if you have a legal query, you can ask ALI (el Asistente Legal Inteligente) and it will reply with all the information that you need, in an understandable way. 
This of course is not a replacement for an actual lawyer, who is authorized to take legal actions, but it will help every citizen be aware of important things such as his/her rights in different situations, what the law says about a given topic, or what changes a new law introduces and how it affects him/her. 

# Who are We
We are just a group of Argentinean developers, named `sandbox.ai`. As of now, this is our temporary [placeholder website](https://sandbox-ai.github.io/). Our full website is currently under construction and will be available soon. Same thing with our social media, it's there but it's not fully fledged yet: [Twitter](https://twitter.com/sandboxaiorg), [LinkedIn](https://www.linkedin.com/in/sandboxai-org-b0a7842b4/).
You can also contact us directly via email at `sandboxai.org@proton.me`.


## Table of Contents
- [Table of Contents](#table-of-contents)
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Folder structure](#folder-structure)
    - [Backend]()
    - [Frontend]()
- [Contributing](#contributing)
- [License](#license)



# Description
## How ALI works
### Overview
The approach we took with ALI is a Retrieval Augmented Generation ([RAG](https://arxiv.org/abs/2005.11401)) system. That is: given an user query, the system searches for the relevant information and hands this information to a Large Language Model (LLM), whom will use that information to answer the query. 

It's important to note that RAG is some sort of trick or cheat. Ideally, we would like our LLM (or Mixture of LLMs) to "know" everything about the Argentinean law. This means that we would have to find a way to compress/encode all the legal information into the parameters of the model (often called "parametric memory"), so that when the LLM is prompted to answer the query, this legal information gets decompressed/decoded to generate an answer. These methods exist (e.g., [fine-tuning]() and [LoRA](https://arxiv.org/abs/2106.09685)), but as of now we these methods are not robust enough to rely solely on them (plus they can be expensive), thus we rely on RAG (often called "non-parametric memory"). If we could reliably teach our models the information that we need them to know (that is, every piece of useful information goes into parametric memory), then we wouldn't need things like RAG. 

One can think of storing information in model parameters as reading a book and remembering it, whereas RAG is more akin to having a book that you can check its pages whenever you need some of the information. 


### Technical information
#### General
As of now, the RAG implemented in ALI is as simple as it can be: 
We embed the user queries using a custom [embedding model](https://huggingface.co/dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn) (which we host in AWS), we search for the best matching legal documents using cosine-similarity, and then ask the LLM to answer the user query using the best matching documents. 
This is far from something advanced and performant. To see our roadmap on RAG improvements, see #Improvements. 
The RAG system is written from scratch in python, see `src/rag_session.py`

#### On LLM frameworks
We tried using both [llama_index](https://github.com/run-llama/llama_index) and [langchain](https://github.com/langchain-ai/langchain), but we found them too restrictive and in the end more cumbersome than writing everything from scratch, which allowed us to have finer control and find where the system was failing. 

Looking ahead, we are more keen on writing the whole pipeline with [DSPy](https://github.com/stanfordnlp/dspy), which has the benefits of a pure python approach plus the benefit of writing custom loss functions to optimize the prompts and model parameters. 

#### Main problems we encountered
The main problem we encountered with a RAG over argentinean legal data was the embedding of the information. This problem has two parts: 

1. Embedding model
    We tried using OpenAI's embedding model (i.e., ada-002), but it wasn't great at distancing different legal topics and clustering similar ones (and it wasn't great in general for Spanish text). Thus we tried the custom embedding [model](https://huggingface.co/dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn) described earlier. 
    Notes: OpenAI has release new [embbeding models](https://openai.com/blog/new-embedding-models-and-api-updates?ref=blog.salesforceairesearch.com) like `text-embedding-3` that we haven't tried, and Cohere seems to be releasing great embedding models (both performant and 100x cheapear than OpenAI's) that are on the top of the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). See [cohere-embed-multilingual-v3.0](https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0).

2. Legal Document Chunking/Parsing
    - We found that just chunking legal documents with a regular document chunker wasn't great, it was hard to compute a good embedding vector with a chunk that is missing a lot of contextual information (e.g., a chunk can say something like: ""). Thus we decided to define each legal article as our atomic unit and embbed it as a whole. 
    - When we tried to embedd whole articles as they were, distancing and clustering of vectors wasn't great. The trick was to preppend all the legal metadata to each legal article BEFORE embedding it. That is, each article to be embbeded looks like: "Decreto de Necesidad y Urgencia N° DNU-2023-70-APN-PTE. Fecha 20-12-2023. Titulo II: DESREGULACIÓN ECONÓMICA. Capitulo II: Tarjetas de crédito (Ley N° 25.065). Articulo 15: La entidad emisora deberá obligatoriamente dar a conocer el público la [...]" instead of just "Articulo 15: La entidad emisora deberá obligatoriamente dar a conocer el público la [...]". 


Note that all the results that we found in relation to the embedding models were a direct conclusion of plotting the resulting embedding vectors with the dimensionality reduction technique [t-SNE: t-distributed Stochastic Neighbor Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). Note that there are alternatives such as [UMAP: Universal Manifold Approximation & Projection](https://github.com/lmcinnes/umap)





### Improvements over baseline RAG. 
There are many improvements to be made over this baseline RAG. 

- Query rewriting (hard without proper context of argentinean law, either parametric or non parametric) potentially with a Fusion retriever. 
- Document reranking
- Hybrid Search (keyword-based search algorithms combined with vector search)
- Document reranking
- Linear adapting for the embedding model
- Fine-tuning over each legal domain (think MoE over each legal subject, such as civil, penal, etc)
- Adding an agentic layer capable of handling complex questions through multiple hops of reasoning and retrieval. 











### Data prep
We have scraped the whole Boletín Oficial. You can find the code to do so [here]()
We need to parse all that data in the way we explained earlier (preppending metadata). This needs to be done with an LLM, since using a hard coded python function seems unfeasible (lots of edge cases that are hard to handle and predict). 

# Hosting of custom LLMs
Since fine-tuning/QLoRAing models will likely improve the performance of the RAG system, the question of where to host those model arises (since hosting them in something like AWS is prohibitively, at least without lots of optimizations). The provider with the best latency and cheaper price for the time being seems to be [Together.AI](https://www.together.ai/), but we haven't tried it yet. 




### Hosting ALI on AWS
ALI is being hostsed on EC2




### Frontend
Angular





# Local LLMs
Ollama
LocalAI



### Running LLMs on laptops


### Running LLMs on smartphones
https://llm.mlc.ai/docs/

https://github.com/mlc-ai/mlc-llm

https://github.com/LostRuins/koboldcpp?tab=readme-ov-file#compiling-on-android-termux-installation

https://github.com/UbiquitousLearning/mLLM

https://scifilogic.com/best-apps-to-run-llm-on-your-smartphone-locally/#:~:text=Apps%20to%20Run%20LLM%20on%20Your%20Smartphone%20Locally,Soon%29%20UI%3A%20The%20best%20among%20the%20four.%20




### Long-term vision
We are hoping to be able to run ALI locally smartphones. 
For one reason, that seems to be the most efficient way (why use cloud when lot's of smartphone compute is idle?), but most importantly, that will give a lot of agency to each citizen. We develop it, they own it. We are all free.
Plus this saves a lot on compute costs (what is the cheapest way to serve 47 million potential users? Locally of course!)
This space is still green, there's a lot of work going on. 

One of the best shots might be to use RNNs instead of Transformers. 
One example of this is the [RWKV Language Model](https://wiki.rwkv.com/), which reduces compute requirements by a potential 100x and scales linearly with context length, among other things. 

https://llm.mlc.ai/docs/

https://github.com/mlc-ai/mlc-llm

https://github.com/LostRuins/koboldcpp?tab=readme-ov-file#compiling-on-android-termux-installation

https://github.com/UbiquitousLearning/mLLM

https://scifilogic.com/best-apps-to-run-llm-on-your-smartphone-locally/#:~:text=Apps%20to%20Run%20LLM%20on%20Your%20Smartphone%20Locally,Soon%29%20UI%3A%20The%20best%20among%20the%20four.%20








# Thanks you notes
Huge thanks to the [Justicio](https://github.com/bukosabino/justicio) team from Spain, who gave us a lot of tips and shared their embedding model with us
Definetly go check their project and talk to the creators!

# Proyectos similares en español
- [Justicio](https://github.com/bukosabino/justicio) - España 


# STUFF TO ALOCATE

[Mamba](https://arxiv.org/abs/2312.00752) and you can find the repo [here](https://github.com/spaceLabLLM/mamba)
Mamba seems to be in its very early stages so although its worth to keep an eye on its development, it doesn't seem its going to be usable anytime soon. 

[RWKV](https://arxiv.org/abs/2305.13048) and you can find the repo [here]().

The RWKV project is very developed (CAMBIAR ESTE PHRASING) and has a hugging face [integration](https://huggingface.co/docs/transformers/model_doc/rwkv)




# ((((Teaching)))) LLMs 
putting all the information inside the context windows
compressing/encoding the information in the form of model parameters (either fine-tuning or LoRA and its variants)




# Long-term strategy












to install 





Create new environment, install backend and frontend requirements
```bash
 
conda create --name <env-name>
conda activate <env-name>
cd backend
pip install -r requirements.txt
cd ../frontend
npm install 
```

to use:

Launch the backend in one terminal
```bash
conda activate <envname>
export OPENAI_API_KEY=<your-key-here>
cd backend
python api.py
```

Open a new terminal and launch the frontend: 
```bash
cd frontend
ng serve
```

















                  ___                 _  _                    ___  ___ 
                 / __| __ _  _ _   __| || |__  ___ __ __     /   \|_ _|
                 \__ \/ _` || ' \ / _` ||  _ \/ _ \\ \ /  _  | - | | | 
                 |___/\__/_||_||_|\__/_||____/\___//_\_\ (_) |_|_||___|          


## RAG your way through Argentina's legal documents







### Improving the rag pipeline. 
As of now, the rag pipeline is quite simple. We need to add query rewriting, llm guard, doc reranking, and so on. 





### Testing
We need to define at least 20 questions that we want to test.

We can test with [ragas](https://github.com/explodinggradients/ragas) but this must be done with a local LLM otherwise our wallets gonna burn (we tested this with a relatively small document and gpt4 and spent 40 U$$ in half an hour)
