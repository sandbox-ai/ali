from src.rag_session import *
import src.custom_logging as logger
import traceback
import textwrap
import logging
import os
from dotenv import load_dotenv


__author__ = "SandboxAI Team"
__copyright__ = "Copyright 2023, Team Research"
__credits__ = ["SandboxAI"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "SanboxAI Team"
__email__ = "sandboxai <dot> org <at> proton <dot> me"
__status__ = "Development"


load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')


if __name__ == "__main__":
    # Set up logging level:
    logging.basicConfig(level=logging.INFO)

    logging_dir = "./logs/"
    bot_name = "ALI"
    session_name = "ALI-rag"
    file_path_vectorstore = "./data/dnu_vectorstore.json"

    logging_filepath = logger.create_log_file(bot_name, os.path.join(logging_dir, session_name))

    try:
        # Take user query:
        user_query = input("Pregunta sobre la legislación Argentina:")

        file_path_vectorstore = "./data/dnu_vectorstore.json"

        data_loader = DataLoader()
        dnu = data_loader.load_json("./data/ALI/decreto_flat.json")
        dnu_metadata = data_loader.load_json("./data/dnu_metadata.json")

        embedder = Embedder("dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn")

        if not os.path.exists(file_path_vectorstore):
            vectorstore = embedder.embed_text(dnu)
            VectorStoreManager.save_vectorstore(file_path_vectorstore, vectorstore)
        else:
            vectorstore = VectorStoreManager.read_vectorstore(file_path_vectorstore)

        query_vector = embedder.embed_text(user_query)

        # Initialize the QueryEngine with necessary parameters
        query_engine = QueryEngine(vectorstore, embedder, legal_docs=dnu, legal_metadata=dnu_metadata, top_k=5)

        # Use the query_similarity method to find documents similar to the query
        top_k_docs, matching_docs = query_engine.query_similarity(
            query=user_query
        )

        # Now, use the generate_llm_response method to generate a response based on the query and matching documents
        text = query_engine.generate_llm_response(
            query=user_query, 
            client=OpenAI(),
            model_name="gpt-4-0125-preview",
            temperature=0,
            max_tokens=1000, #2000,
            streaming=False, #True
            top_k_docs=top_k_docs,
            matching_docs=matching_docs,
        )

        citations = get_stored_citations(top_k_docs, dnu_metadata)

        # Print:
        print(text)
        for citation in citations.values():
            print(f"\n{citation['score']*100:.2f}% - {citation['text']}\n{citation['metadata']}, {citation['start_char_idx']}-{citation['end_char_idx']}")

    except Exception as e:
        # Extract error info:
        tb = traceback.extract_tb(e.__traceback__)
        filename, line, func, text = tb[-1]

        # Log the error
        error_message = textwrap.dedent(f"""\
            ========================================
            ||               ERROR                ||
            ========================================   
            File: {filename}
            Function name: {func}
            Line {line}: 
                {text}

            Error: {e}""")

        #logger.save_string(error_message, session.logging_filepath)
        raise
