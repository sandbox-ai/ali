# application.py
# -*- encoding: utf-8 -*-

from src.rag_session import *
import src.custom_logging as logger
import logging

from flask import Flask
from flask_cors import CORS
from flask import Blueprint, request, jsonify, current_app
from flask import Response

import sys
import argparse
import textwrap
import traceback

import multiprocessing
import http.server
import socketserver

# Set up logging level:
logging.basicConfig(level=logging.INFO)

# Set up Flask home Blueprint:
home = Blueprint('home_views', __name__)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="../frontend/dist/", **kwargs)


def flaskServer(ip='127.0.0.1', port=5000):
    app = create_application()
    logging.info("Flask serving...")
    app.run(port=port, debug=True, host=ip, use_reloader=False)


def httpServer():
    PORT = 4200
    logging.info("HTTPD serving... http://127.0.0.1:4200")
    with socketserver.TCPServer(("", PORT), Handler) as httpd_server:
        httpd_server.serve_forever()


def create_application():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(home)
    return app


################################################
# Heartbeat
################################################
@home.route("/api/heartbeat", methods=["GET"])
@home.route("/heartbeat", methods=["GET"])
def r_heartbeat():
    return jsonify({"heartbeat": "OK"})


################################################
# Question
################################################
@home.route("/api/question", methods=["POST"])
@home.route("/question", methods=["POST"])
def r_question():
    # Set up log file:
    logging_filepath = logger.create_log_file("La Ley de Milei", {}, os.path.join(logging_dir, "LaLeyDeMilei"))

    # Access query_engine from the Flask application context:
    query_engine = current_app.config['QUERY_ENGINE']

    try:
        # Get user query:
        user_query = request.get_json().get("question", "")
        logger.save_user_message(user_query, logging_filepath)

        # Use the query_similarity method to find chunks similar to the query:
        top_k_docs, matching_docs = query_engine.query_similarity(query=user_query)

        # WITHOUT STREAMING
        # Respond user query:
        text = query_engine.generate_llm_response(
            query=user_query,
            client=OpenAI(),
            model_name='gpt-3.5-turbo-0125',
            temperature=0,
            max_tokens=2000,
            streaming=False, #True, #False,
            top_k_docs=top_k_docs,
            matching_docs=matching_docs,
        )

        citations = get_stored_citations(top_k_docs, dnu_metadata)
        #citations = query_engine.generate_complete_citations_dict(matching_docs, top_k_docs)
        #citations = query_engine.get_stored_citations(top_k_docs, dnu_metadata)

        # Log bot response:
        logger.save_bot_message(text, logging_filepath, citations=citations)

        sources = []
        for citation in citations.values():
            metadata = citation['metadata']
            source_text = citation['text'].strip('\n')
            if metadata['documento'].lower() == 'decreto':
                source = f'{metadata["documento"]}\n{metadata["titulo"]}\n{metadata["capitulo"] + " - " if "capitulo" in metadata else ""}{metadata["articulo"]}\n\n"{source_text}"'
            else:
                source = f'{metadata["documento"]}\n\n"{source_text}"'
            sources.append(source)
            logging.info(source)

        return jsonify(answer=text, sources=sources, error="OK")

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

        logger.save_string(error_message, logging_filepath)
        raise

    # WITH STREAMING
    #def generate_stream():
        #for chunk in query_engine.generate_llm_response2(
            #query=user_query, 
            #client=OpenAI(),
            #model_name='gpt-3.5-turbo-0125', 
            #temperature=0,
            #max_tokens=2000,
            #streaming=True,
            #top_k_docs=top_k_docs,
            #matching_docs=matching_docs,
        #):
            ## Assuming each chunk is a string that needs to be JSON-formatted
            ##yield json.dumps({"text": chunk}) + "\n"  # NDJSON format
            ## test
            #yield chunk

    ### Return a streaming response with NDJSON content type
    #return Response(generate_stream(), content_type='application/x-ndjson')
    #return Response(generate_stream(), content_type='application/json')


if __name__ == '__main__':
    # Create parser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ip', action='store', default='127.0.0.1', help='IP address, just for vagrant')
    parser.add_argument('-p', '--port', action='store', default=5000, help='Listen port')
    parser.add_argument('-e', '--env', action='store', default='dev', help='Environment [dev, prod]')

    # Extract args from parser:
    args = parser.parse_args()
    ip = str(args.ip)
    port = str(args.port)
    env = str(args.env)

    # Set up logging file:
    logging_dir = "./logs/"

    # Load processed DNU:
    data_loader = DataLoader()
    dnu = data_loader.load_json("./data/LaLeyDeMilei-raw/decreto_flat.json")
    dnu_metadata = data_loader.load_json("./data/dnu_metadata.json")

    # Load embeddings model:
    embedder = Embedder("dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn")

    # Load / Create vectorstore:
    file_path_vectorstore = "./data/dnu_vectorstore.json"
    if os.path.exists(file_path_vectorstore):
        vectorstore = VectorStoreManager.read_vectorstore(file_path_vectorstore)
    else:
        vectorstore = embedder.embed_text(dnu)
        VectorStoreManager.save_vectorstore(file_path_vectorstore, vectorstore)

    # Initialize query engine:
    query_engine = QueryEngine(
        vectorstore,
        embedder,
        legal_docs=dnu,
        legal_metadata=dnu_metadata,
        top_k=5
    )

    # Create Flask app:
    app = create_application()
    # Load query engine into app:
    app.config['QUERY_ENGINE'] = query_engine

    # Check if environment is set to production:
    if env == 'prod':
        sys.stdout.flush()
        # Initialize a separate process for the Flask server:
        flask_proc = multiprocessing.Process(name='flask', target=flaskServer, kwargs={"ip": ip, "port": port})
        flask_proc.daemon = True    # Automatically terminate when the main process ends

        sys.stdout.flush()
        # Initialize a separate process for the HTTP server:
        httpd_proc = multiprocessing.Process(name='httpd', target=httpServer)
        httpd_proc.daemon = True    # Automatically terminate when the main process ends

        # Start parallel processes:
        flask_proc.start()
        httpd_proc.start()
        # Wait until manual termination or error:
        flask_proc.join()
        httpd_proc.join()
    else:
        app.run(port=int(port), debug=True, host=ip, use_reloader=False)


