from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
import logging
from typing import Optional

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from src.rag_session import QueryEngine, DataLoader, Embedder, VectorStoreManager, get_stored_citations
    from openai import OpenAI
except ImportError as e:
    logging.error(f"Import error: {e}")
    # Fallback for development/testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["*"])

# Global variables for query engine
query_engine: Optional[QueryEngine] = None
embedder: Optional[Embedder] = None
dnu_metadata: Optional[dict] = None

def initialize_query_engine():
    """Initialize the query engine with necessary data."""
    global query_engine, embedder, dnu_metadata
    
    try:
        # Initialize embedder
        embedder = Embedder("dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn")
        
        # Load data
        data_loader = DataLoader()
        
        # Check if running in Vercel environment
        backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
        
        # Try to load the decree data
        dnu_path = os.path.join(backend_path, "data", "ALI", "decreto_flat.json")
        metadata_path = os.path.join(backend_path, "data", "dnu_metadata.json")
        vectorstore_path = os.path.join(backend_path, "data", "dnu_vectorstore.json")
        
        if os.path.exists(dnu_path):
            dnu = data_loader.load_json(dnu_path)
        else:
            # Fallback empty data for now
            dnu = []
            logging.warning("DNU data file not found, using empty data")
        
        if os.path.exists(metadata_path):
            dnu_metadata = data_loader.load_json(metadata_path)
        else:
            dnu_metadata = {}
            logging.warning("DNU metadata file not found, using empty metadata")
        
        # Handle vectorstore
        if os.path.exists(vectorstore_path):
            vectorstore = VectorStoreManager.read_vectorstore(vectorstore_path)
        else:
            # Create empty vectorstore if files don't exist
            vectorstore = []
            logging.warning("Vectorstore not found, using empty vectorstore")
        
        # Initialize QueryEngine
        query_engine = QueryEngine(vectorstore, embedder, legal_docs=dnu, legal_metadata=dnu_metadata, top_k=5)
        
        logging.info("Query engine initialized successfully")
        
    except Exception as e:
        logging.error(f"Error initializing query engine: {e}")
        # Set up minimal fallback
        query_engine = None
        embedder = None
        dnu_metadata = {}

@app.route("/api/heartbeat", methods=["GET"])
@app.route("/heartbeat", methods=["GET"])
def heartbeat():
    """Health check endpoint."""
    return jsonify({"heartbeat": "OK", "status": "running"})

@app.route("/api/question", methods=["POST"])
@app.route("/question", methods=["POST"])
def question():
    """Main question answering endpoint."""
    global query_engine, dnu_metadata
    
    try:
        # Get the JSON data from request
        json_result = request.get_json()
        if not json_result:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_query = json_result.get("question", "")
        if not user_query:
            return jsonify({"error": "No question provided"}), 400
        
        logging.info(f"Question received: {user_query}")
        
        # Check if query engine is initialized
        if not query_engine:
            return jsonify({
                "answer": "Lo siento, el sistema está inicializándose. Por favor, intenta nuevamente en unos momentos.",
                "sources": [],
                "error": "System initializing"
            })
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return jsonify({
                "answer": "Lo siento, el sistema no está configurado correctamente. Falta la clave de API.",
                "sources": [],
                "error": "API key not configured"
            })
        
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Query similarity search
        top_k_docs, matching_docs = query_engine.query_similarity(query=user_query)
        
        # Generate LLM response
        text = query_engine.generate_llm_response(
            query=user_query, 
            client=client,
            model_name='gpt-3.5-turbo',
            temperature=0,
            max_tokens=2000,
            streaming=False,
            top_k_docs=top_k_docs,
            matching_docs=matching_docs,
        )
        
        # Get citations
        citations = get_stored_citations(top_k_docs, dnu_metadata)
        
        sources = []
        for citation in citations.values():
            metadata = citation['metadata']
            source_text = citation['text'].strip('\n')
            if metadata.get('documento', '').lower() == 'decreto':
                source = f'{metadata.get("documento", "")}\n{metadata.get("titulo", "")}\n{metadata.get("capitulo", "") + " - " if "capitulo" in metadata else ""}{metadata.get("articulo", "")}\n\n"{source_text}"'
            else:
                source = f'{metadata.get("documento", "")}\n\n"{source_text}"'
            sources.append(source)
        
        return jsonify(answer=text, sources=sources, error="OK")
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return jsonify({
            "answer": "Lo siento, hubo un error procesando tu pregunta. Por favor, intenta nuevamente.",
            "sources": [],
            "error": str(e)
        }), 500

# Initialize on startup
initialize_query_engine()

# For Vercel, we need to export the app
def handler(request, context):
    """Vercel serverless function handler."""
    return app(request, context)

# For local development
if __name__ == "__main__":
    app.run(debug=True) 