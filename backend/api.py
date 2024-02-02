# application.py
# -*- encoding: utf-8 -*-

from src.rag_session import RAGSession
import src.custom_logging as logger
import logging

from flask import Flask
from flask_cors import CORS
from flask import Blueprint, request, jsonify

import sys
import argparse
import textwrap
import traceback

import multiprocessing
import http.server
import socketserver

# Set up logging level:
logging.basicConfig(level=logging.INFO)

# Path to session configuration file:
config_filepath = r"./config.json"

# Set up RAG session:
session = RAGSession(config_filepath)

home = Blueprint('home_views', __name__)

try:
    # Set up session:
    session.set_up()

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


def flaskServer(ip='127.0.0.1', port=5000):
    app = create_application()
    # app.run(host=ip, port=port, debug=True)
    logging.info("Flask serving...")
    app.run(port=port, debug=True, host=ip, use_reloader=False)


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
# Heartbeat
################################################
@home.route("/heartbeat", methods=["GET"])
def r_heartbeat():
    return jsonify({"heartbeat": "OK"})


################################################
# Question
################################################
@home.route("/question", methods=["POST"])
def r_question():
    json_result = request.get_json()
    user_query = json_result.get("question", "")

    logging.info(f"Param received")
    logging.info(f"Question : {user_query}")

    # Respond user query:
    text, citations = session.generate(user_query)

    logging.info(f"Response Answer: {text}")
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

