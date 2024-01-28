from src.rag_session import RAGSession
import src.custom_logging as logger
import traceback
import textwrap
import logging
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

if __name__ == "__main__":
    # Set up logging level:
    logging.basicConfig(level=logging.INFO)

    # Path to session configuration file:
    config_filepath = r"./config.json"

    # Set up RAG session:
    session = RAGSession(config_filepath)

    try:
        # Set up session:
        session.set_up()

        # Take user query:
        user_query = input("Pregunta sobre el DNU del presidente Javier Milei: ")

        # Respond user query:
        text, citations = session.generate(user_query)

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

        logger.save_string(error_message, session.logging_filepath)
        raise
