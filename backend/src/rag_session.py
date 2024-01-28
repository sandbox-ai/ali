from llama_index import StorageContext, load_index_from_storage
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import QueryFusionRetriever
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings import OpenAIEmbedding
from llama_index.readers import JSONReader
from llama_index.schema import TextNode
from llama_index.llms import OpenAI
import src.custom_logging as logger
import logging
import json
import os
from src.prompts import *

class RAGSession:
    def __init__(self, config_filepath: str, use_custom_prompts: bool = False, text_qa_prompt = CUSTOM_TEXT_QA_PROMPT, refine_prompt = CUSTOM_REFINE_PROMPT):
        """
        Retrieval Augmented Generation (RAG) Session. Contains tools for:
            - Chunking and indexing of text documents
            - Loading and saving vectorstores
            - Retrieval of relevant chunks based on user queries
            - Generation of answers based on retrieved chunks

        Args:
            config_filepath: str -> Path to RAG Session configuration file.
        """
        """ --- Configuration ------------------------------------------------- """
        # Load session configuration:
        if os.path.exists(config_filepath):
            with open(config_filepath, 'r', encoding='utf-8') as file:
                config = json.load(file)
        else:
            logging.error(f"[ ERROR ] Configuration file not found at '{config_filepath}'.")
            exit()

        # Chunking:
        self.chunk_size = config['chunk_size']
        self.chunk_overlap = config['chunk_overlap']

        # Retrieval:
        self.n_queries = config['n_queries']
        self.ranking_mode = config['ranking_mode']
        self.vector_similarity_top_k = config['vector_similarity_top_k']
        self.fusion_top_k = config['fusion_top_k']

        # Paths:
        self.logging_dir = config['logging_dir']
        self.rawdata_filepath = config['rawdata_filepath']
        vectorstores_dir = config['vectorstores_dir']

        # Models:
        self.gpt_version = config['gpt_version']
        self.model_temperature = config['model_temperature']
        self.model_max_tokens = config['model_max_tokens']

        # Miscellaneous:
        self.autosetup = config['autosetup']
        self.session_name = config['session_name']
        self.bot_name = config['bot_name']

        """ --- Initialization ------------------------------------------------ """
        self.nodes = []
        self.vectorstore = None
        self.vector_retriever = None
        self.fusion_retriever = None
        self.query_engine = None

        # Create file for logging sessions:
        self.logging_filepath = logger.create_log_file(self.bot_name, config, os.path.join(self.logging_dir, self.session_name))

        # Define service context:
        if config['model_host'] == 'local':
            self.vectorstore_dir = os.path.join(
                vectorstores_dir,
                'local',
                config['local_embeddings_model'].split(sep='/')[-1],
                f"{self.session_name}"
            )
            self.embeddings_model = f"local:{config['local_embeddings_model']}"
            self.service_context = ServiceContext.from_defaults(
                embed_model=self.embeddings_model,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif config['model_host'] == 'openai':
            self.vectorstore_dir = os.path.join(
                vectorstores_dir,
                'openai',
                f"{self.session_name}"
            )
            self.embeddings_model = OpenAIEmbedding()
            self.service_context = ServiceContext.from_defaults(
                embed_model=self.embeddings_model,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError("[ ERROR ] 'model_host' must be either 'openai' or 'local'.")

        if self.autosetup:
            self.set_up()
        
        # Custom prompts
        if config['use_custom_prompts']:
            self.text_qa_template = text_qa_prompt
            self.refine_template = refine_prompt


        logging.info('Initialized Session.')

    """ === ACTIONS =============================================================================== """
    def set_up(self, save_vectorstore: bool = True, load_vectorstore: bool = True):
        """
        Sets up the RAG session.

        Args:
            save_vectorstore: bool -> If True, the generated vectorstore will be saved to storage. Default is True.
            load_vectorstore: bool -> If True, the function will try to load a vectorstore from storage. Default is True.

        Returns:
            None
        """
        if load_vectorstore:
            if os.path.exists(self.vectorstore_dir):
                # Load VectorStoreIndex from Storage:
                storage_context = StorageContext.from_defaults(persist_dir=self.vectorstore_dir)
                self.vectorstore = load_index_from_storage(storage_context, service_context=self.service_context)
                logging.info('Loaded vectorstore.')
            else:
                logging.warning('Vectorstore not found. Creating new one.')
                os.makedirs(self.vectorstore_dir)

                # Create nodes from raw data:
                self.create_nodes_from_preprocessed()

                # Create vector store index from nodes:
                self.create_vectorstore()

                if save_vectorstore:
                    # Save the index to the specified directory
                    self.vectorstore.storage_context.persist(persist_dir=self.vectorstore_dir)
        else:
            # Create nodes from raw data:
            self.create_nodes_from_preprocessed()

            # Create vector store index from nodes:
            self.create_vectorstore()

            if save_vectorstore:
                # Save the index to the specified directory
                self.vectorstore.storage_context.persist(persist_dir=self.vectorstore_dir)

        # Create retrievers:
        self.create_vector_retriever()
        self.create_fusion_retriever()

        # Create query engine:
        self.create_query_engine()

    def generate(self, user_query: str):
        # Save to log file:
        logger.save_user_message(user_query, self.logging_filepath)

        # Generate response:
        answer = self.query_engine.query(user_query)

        # Extract text and sources:
        text = answer.response
        metadata = answer.metadata
        source_nodes = answer.source_nodes
        citations = {}

        # Extract relevant data for citations:
        for source_node in source_nodes:
            node_id = source_node.node_id
            node_data = {
                'text': source_node.text,
                'score': source_node.score,
                'metadata': source_node.metadata,
                'start_char_idx': source_node.node.start_char_idx,
                'end_char_idx': source_node.node.end_char_idx,
            }
            citations[node_id] = node_data

        # Save to log file:
        logger.save_bot_message(text, bot_name=self.bot_name, filepath=self.logging_filepath, citations=citations)

        return text, citations

    """ === UTILS ================================================================================= """
    def create_nodes(self, levels_back: int = 0):
        """
        This function creates nodes from a JSON file using LlamaIndex's JSONReader and SimpleNodeParser.

        Args:
            levels_back: int -> The number of levels of the JSON hierarchy to go back when creating the TextNode objects. A value of 0 means that all levels will be included. Default is 0.

        Returns:
            None
        """
        # Create a Reader for the data's filetype:
        reader = JSONReader(levels_back=levels_back)

        # Read data into documents:
        documents = reader.load_data(self.rawdata_filepath)

        # Create a SimpleNodeParser instance:
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Parse the loaded data into TextNode objects:
        self.nodes = node_parser.get_nodes_from_documents(documents)

        logging.info('Created nodes.')

    def create_nodes_from_preprocessed(self):
        """
        This function creates nodes from preprocessed JSON files where each entry contains a 'text' and 'metadata' field.

        Returns:
            None
        """
        for source in os.listdir(self.rawdata_filepath):
            source_path = os.path.join(self.rawdata_filepath, source)
            with open(source_path, 'r', encoding='utf-8') as file:
                logging.info(f'Creating nodes from {source}...')
                dataset = json.load(file)

            for data in dataset:
                text = data['text']
                metadata = data['metadata']
                node = TextNode(text=text,
                                metadata=metadata,
                                excluded_llm_metadata_keys=['laws', 'start_idx', 'end_idx'],
                                excluded_embed_metadata_keys=['laws', 'start_idx', 'end_idx'],
                                metadata_template="{value}",
                                text_template="{metadata_str}\n\n{content}")
                self.nodes.append(node)

        logging.info('Created nodes.')

    def create_vectorstore(self):
        """
        Creates a VectorStoreIndex from the current list of TextNode objects.

        Args:
            model_host: str -> The embedding model to be used to create the Vectorstore. Must be either "openai" or "local". Default is "local".

        Returns:
            None
        """
        logging.info('Creating vectorstore...')

        # Create a VectorStoreIndex from the nodes using the ServiceContext
        self.vectorstore = VectorStoreIndex(self.nodes, service_context=self.service_context)

        logging.info('Created vectorstore.')

    def create_vector_retriever(self):
        self.vector_retriever = self.vectorstore.as_retriever(similarity_top_k=self.vector_similarity_top_k, service_context=self.service_context)
        logging.info('Created vector retriever.')

    def create_fusion_retriever(self):
        # Declare query generation prompt (the following was taken from the llama_index documentation and translated):
        QUERY_GEN_PROMPT = (
            "Eres un asistente útil especializado en derecho y leyes en la argentina que genera múltiples consultas de búsqueda basadas en una única"
            #"Eres un asistente útil especializado en derecho y leyes ARGENTINAS (especialmente el Decreto de Necesidad y Urgencia o DNU que anunció el presidente Javier Milei el 20-12-2023) que genera múltiples consultas de búsqueda basadas en una única"
            "consulta de entrada. Genera {num_queries} consultas de búsqueda, una en cada línea, "
            "relacionadas con la siguiente consulta de entrada:\n"
            "Consulta: {query}\n"
            "Consultas:\n"
        )

        # Set up fusion retriever:
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[self.vector_retriever],
            similarity_top_k=self.fusion_top_k,
            num_queries=self.n_queries,
            mode=self.ranking_mode,
            llm=OpenAI(model=self.gpt_version, temperature=self.model_temperature, max_tokens=self.model_max_tokens),
            use_async=True,
            verbose=True,
            query_gen_prompt=QUERY_GEN_PROMPT
        )

        logging.info('Created fusion retriever.')

    def create_query_engine(self):
        self.query_engine = RetrieverQueryEngine.from_args(
            self.fusion_retriever, 
            service_context=self.service_context, 
            text_qa_template=self.text_qa_template,
            refine_template=self.refine_template,
        )
        logging.info('Created query engine.')

