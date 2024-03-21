# application.py
# -*- encoding: utf-8 -*-


__author__ = "SandboxAI Team"
__copyright__ = "Copyright 2023, Team Research"
__credits__ = ["SandboxAI"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "SanboxAI Team"
__email__ = "sandboxai <dot> org <at> proton <dot> me"
__status__ = "Development"


from src.rag_session import *
import src.custom_logging as logger
import logging

from flask import Flask
from flask_cors import CORS
from flask import Blueprint, request, jsonify
from flask import Response

import sys
import argparse
import textwrap
import traceback

import multiprocessing
import http.server
import socketserver


# Set up logging level
logging.basicConfig(level=logging.INFO)

# Path to session configuration file:
config_filepath = r"./config.json"

home = Blueprint('home_views', __name__)
query_engine = None
embedder = None
dnu_metadata = None


def flaskServer(ip='127.0.0.1', port=5000):
    app = create_application()
    logging.info("Flask serving...")
    app.run(port=port, debug=False, host=ip, use_reloader=False)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="../frontend/dist/", **kwargs)


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
# Initialize query_engine 
################################################
def initialize_query_engine():
    global query_engine
    global embedder
    global dnu_metadata

    file_path_dnu = "./data/LaLeyDeMilei-raw/decreto_flat.json"
    file_path_dnu_unpreppended = (
        "./data/LaLeyDeMilei-raw/decreto_flat_unpreppended.json"
    )
    file_path_vectorstore = "./data/dnu_vectorstore.json"

    data_loader = DataLoader()
    dnu = data_loader.load_json("./data/LaLeyDeMilei-raw/decreto_flat.json")
    dnu_unpreppended = data_loader.load_json(
        "./data/LaLeyDeMilei-raw/decreto_flat_unpreppended.json"
    )
        
    dnu_metadata = data_loader.load_json("./data/dnu_metadata.json")

    if not os.path.exists(file_path_vectorstore):
        vectorstore = embedder.embed_text(dnu)
        VectorStoreManager.save_vectorstore(file_path_vectorstore, vectorstore)
    else:
        vectorstore = VectorStoreManager.read_vectorstore(file_path_vectorstore)

    embedder = Embedder("dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn")

    # Initialize the QueryEngine with necessary parameters
    query_engine = QueryEngine(vectorstore, embedder, legal_docs=dnu, legal_metadata=dnu_metadata, top_k=5)


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
# def r_question(embedder = embedder, query_engine = query_engine):
def r_question():
    json_result = request.get_json()
    user_query = json_result.get("question", "")

    logging.info(f"Param received")
    logging.info(f"Question : {user_query}")

    # Use the query_similarity method to find documents similar to the query
    top_k_docs, matching_docs = query_engine.query_similarity(
        query=user_query
    )

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
    # citations = query_engine.generate_complete_citations_dict(matching_docs, top_k_docs)
    # citations = query_engine.get_stored_citations(top_k_docs, dnu_metadata)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ip', action='store', default='127.0.0.1',
                        help='IP address, just for vagrant')
    parser.add_argument('-p', '--port', action='store', default=5000,
                        help='Listen port')
    parser.add_argument('-e', '--env', action='store', default='dev',
                        help='Environment [dev, prod]')

    args = parser.parse_args()
    ip = str(args.ip)
    port = str(args.port)
    env = str(args.env)
    app = create_application()
    initialize_query_engine()

    if (env == 'prod'):
        sys.stdout.flush()
        kwargs_flask = {"ip": ip, "port": port}
        flask_proc = multiprocessing.Process(name='flask',
                                                target=flaskServer,
                                                kwargs=kwargs_flask)
        flask_proc.daemon = True

        sys.stdout.flush()
        httpd_proc = multiprocessing.Process(name='httpd',
                                                target=httpServer)
        httpd_proc.daemon = True

        flask_proc.start()
        httpd_proc.start()
        flask_proc.join()
        httpd_proc.join()

    else:
        app.run(port=port, debug=True, host=ip, use_reloader=False)

