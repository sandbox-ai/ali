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


## Mission
Our mission is to democratize access to legal information in Argentina, fostering a more informed society. While legal information is abundant, it remains challenging for citizens outside the legal domain to efficiently search and comprehend our legal system. Until recently, there existed an upper limit on how informed a society could be, consequently constraining its development. This upper boundary, however, is not static - it is defined by the current technology available to that society. As groundbreaking new technologies emerge, they effectively raise this ceiling, allowing for previously unattainable levels of progress. Throughout history, we have witnessed two pivotal revolutions in information technology – the invention of the printing press and the advent of the internet – each propelling society forward by expanding the boundaries of its potential development. Now, large language models (LLMs) present us with a new pivotal moment, one where we can access all the information we need in an effective and efficient way.

At the core of our vision is ALI (el Asistente Legal Inteligente), an intelligent legal assistant that acts as a bridge between the complexities of the legal system and the needs of everyday citizens. Through natural language interaction, ALI will demystify intricate legal concepts, providing clear explanations of rights, regulations, and the consequences of emerging legislation. While not a substitute for professional legal counsel, ALI will equip citizens with a deeper understanding of the laws that shape their lives.

We recognize that building a tool as transformative as ALI requires a collaborative and transparent approach. This initiative must be driven by the needs and perspectives of the community it aims to serve. By fostering an open dialogue and actively involving diverse stakeholders – from legal experts to civic organizations and engaged citizens – we can ensure that ALI accurately reflects the multifaceted realities of Argentina's legal landscape. Transparency will be paramount, as we strive to build trust and accountability through every stage of development and deployment.

Ultimately, our goal is to create a tool that belongs to the people, one that empowers individuals with the knowledge and understanding to navigate the legal system with confidence and agency. By embracing a community-driven, open approach, we can collectively unlock the full potential of this technology, shaping a future where access to legal information is truly democratized.

You can try ALI at [https://chat.omnibus.com.ar/](https://chat.omnibus.com.ar/).

## Who are We
We are just a group of Argentinean developers, named `sandbox.ai`. As of now, this is our temporary [placeholder website](https://sandbox-ai.github.io/). Our full website is currently under construction and will be available soon. Same thing with our social media, it's there but it's not fully fledged yet: [Twitter](https://twitter.com/sandboxaiorg), [LinkedIn](https://www.linkedin.com/in/sandboxai-org-b0a7842b4/).
You can also contact us directly via email at `sandboxai.org@proton.me`.


## Table of Contents
- [Mission](#mission)
- [Who are We](#who-are-we)
- [Table of Contents](#table-of-contents)
- [Description](#description)
  - [Overview](#overview)
  - [Technical information](#technical-information)
    - [General](#general)
    - [On LLM frameworks](#on-llm-frameworks)
    - [Main problems we encountered](#main-problems-we-encountered)
    - [Improvements over baseline RAG.](#improvements-over-baseline-rag)
    - [Data prep](#data-prep)
    - [Hosting of custom LLMs](#hosting-of-custom-llms)
    - [Running ALI locally](#running-ali-locally)
    - [Using Local LLMs instead of cloud-served ones (openai, anthropic, etc)](#using-local-llms-instead-of-cloud-served-ones-openai-anthropic-etc)
    - [Running ALI on a smartphone!?](#running-ali-on-a-smartphone)
    - [Testing](#testing)
- [Long-term vision](#long-term-vision)
- [Thank you notes](#thank-you-notes)
- [Sponsoring the project](#sponsoring-the-project)


## Description
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

Looking ahead, we are more keen on writing the whole pipeline with [DSPy](https://github.com/stanfordnlp/dspy).   

#### Main problems we encountered
The main problem we encountered with a RAG over argentinean legal data was the embedding of the information. This problem has two parts: 

1. Embedding model
    We tried using OpenAI's embedding model (i.e., ada-002), but it wasn't great at distancing different legal topics and clustering similar ones (and it wasn't great in general for Spanish text). Thus we tried the custom embedding [model](https://huggingface.co/dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn) described earlier. 
    Notes: OpenAI has release new [embbeding models](https://openai.com/blog/new-embedding-models-and-api-updates?ref=blog.salesforceairesearch.com) like `text-embedding-3` that we haven't tried, and Cohere seems to be releasing great embedding models (both performant and 100x cheapear than OpenAI's) that are on the top of the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). See [cohere-embed-multilingual-v3.0](https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0).

2. Legal Document Chunking/Parsing
    - We found that just chunking legal documents with a regular document chunker wasn't great, it was hard to compute a good embedding vector with a chunk that is missing a lot of contextual information (e.g., a chunk can say something like: ""). Thus we decided to define each legal article as our atomic unit and embbed it as a whole. 
    - When we tried to embedd whole articles as they were, distancing and clustering of vectors wasn't great. The trick was to preppend all the legal metadata to each legal article BEFORE embedding it. That is, each article to be embbeded looks like: "Decreto de Necesidad y Urgencia N° DNU-2023-70-APN-PTE. Fecha 20-12-2023. Titulo II: DESREGULACIÓN ECONÓMICA. Capitulo II: Tarjetas de crédito (Ley N° 25.065). Articulo 15: La entidad emisora deberá obligatoriamente dar a conocer el público la [...]" instead of just "Articulo 15: La entidad emisora deberá obligatoriamente dar a conocer el público la [...]". 


Note that all the results that we found in relation to the embedding models were a direct conclusion of plotting the resulting embedding vectors with the dimensionality reduction technique [t-SNE: t-distributed Stochastic Neighbor Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). Note that there are alternatives such as [UMAP: Universal Manifold Approximation & Projection](https://github.com/lmcinnes/umap)





#### Improvements over baseline RAG. 
There are many improvements to be made over this baseline RAG. The following is a non-exhaustive list: 

- Query rewriting (hard without proper context of argentinean law, either parametric or non parametric), potentially connected with a Fusion retriever. 
- Document reranking
- Hybrid Search (keyword-based search algorithms combined with vector search)
- Document reranking
- Linear adapter for the embedding model
- Fine-tuning over each legal domain (think MoE over each legal subject, such as civil, penal, etc)
- Adding an agentic layer capable of handling complex questions through multiple hops of reasoning and retrieval. 


#### Data prep
We have scraped the whole Boletín Oficial. You can find the code to do so [here](https://github.com/sandbox-ai/Boletin-Oficial-Argentina)
We need to parse all that data in the way we explained earlier (preppending metadata). This needs to be done with an LLM, since using a hard coded python function seems unfeasible (lots of edge cases that are hard to handle and predict). 

#### Hosting of custom LLMs
Since fine-tuning/QLoRAing models will likely improve the performance of the RAG system, the question of where to host those model arises (since hosting them in something like AWS is prohibitively, at least without lots of optimizations). The provider with the best latency and cheaper price for the time being seems to be [Together.AI](https://www.together.ai/), but we haven't tried it yet. 

#### Running ALI locally

**Installation**

Create new environment, install backend and frontend requirements
```bash
 
conda create --name <env-name>
conda activate <env-name>
cd backend
pip install -r requirements.txt
cd ../frontend
npm install 
```

**Usage**

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

Also see `backend/README.md` and `frontend/README.md`    


#### Using Local LLMs instead of cloud-served ones (openai, anthropic, etc)
We haven't developed this yet, but we believe the best frameworks to do this are: 
- [Ollama](https://github.com/ollama/ollama)
- [LocalAI](https://github.com/mudler/LocalAI)   


#### Running ALI on a smartphone!?
This space is VERY green yet, but we are excited about this possibility and is part of our long-term vision. 
A few repos that are doing this and we've been keeping an eye on are: 

- [MLC LLM](https://github.com/mlc-ai/mlc-llm)
- [mllm](https://github.com/UbiquitousLearning/mLLM)
- [maid](https://github.com/Mobile-Artificial-Intelligence/maid)
- [Jan](https://github.com/janhq/jan/issues/728)
- [koboldcpp](https://github.com/LostRuins/koboldcpp?tab=readme-ov-file#compiling-on-android-termux-installation)    

#### Testing
We plan to test ALI with [ragas](https://github.com/explodinggradients/ragas), but this must be done with a local LLM (we tested this with a relatively small document and gpt4 and spent 40 U$$ in half an hour)    

## Long-term vision
We are hoping to be able to run ALI locally smartphones. 
For one reason, that seems to be the most efficient way (why use cloud when lot's of smartphone compute is idle?), but most importantly, that will give a lot of agency to each citizen. We develop it, we all own it. We are all free.
Plus this saves a lot on compute costs (what is the cheapest way to serve 47 million potential users? Locally of course!)
This space is still green, there's a lot of work going on. 

One of the best shots might be to use RNNs instead of Transformers. 
One example of this is the [RWKV Language Model](https://wiki.rwkv.com/), which reduces compute requirements by a potential 100x and scales linearly with context length, among other things. 
The other one are the [Mamba](https://arxiv.org/abs/2312.00752) style LLMs, which are based in a state-space architecture, such as [Jamba](https://huggingface.co/ai21labs/Jamba-v0.1).


## Thank you notes
Huge thanks to the [Justicio](https://github.com/bukosabino/justicio) team from Spain, who gave us a lot of tips and shared their embedding model with us
Definetly go check their project and talk to the creators!

## Sponsoring the project
As of now, ALI is running on AWS using the OpenAI api key. Both things are expensive and as of now these costs are being covered by the sandbox.ai devs. 
If you would like to sponsor this project (either by covering the AWS costs or the OpenAI api key costs), please contact us at `sandboxai.org@proton.me`.















