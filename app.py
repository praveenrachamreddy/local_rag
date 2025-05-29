from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import time

# LangChain & Dependencies
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models import LLM
from langchain.llms.base import LLM

# Local Imports
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.labels import DocItemLabel
from transformers import AutoTokenizer
import requests
import tempfile
import json
import urllib3

# ---------------------------
# Enhanced Logging Setup
# ---------------------------

def setup_logging():
    """Setup comprehensive logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('rag_app.log', mode='a')
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------
# Enhanced TinyLLM Wrapper
# ---------------------------

class TinyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "tinyllm"
    
    class Config:
        arbitrary_types_allowed = True

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.2
        }

        try:
            response = requests.post(
                "https://vllm-vllm.apps.ocp4.imss.work/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                verify=False
            )

            print("Raw LLM Response:", response.text)
            result = response.json()

            if "choices" not in result:
                raise ValueError(f"'choices' not found in LLM response: {result}")

            if len(result["choices"]) == 0:
                raise ValueError("Empty 'choices' list in LLM response")

            content = result["choices"][0]["message"].get("content")
            if not content:
                raise ValueError(f"Missing 'content' in message: {result['choices'][0]['message']}")

            return content

        except json.JSONDecodeError as je:
            raise RuntimeError(f"Failed to parse LLM response as JSON: {str(je)}") from je
        except Exception as e:
            raise RuntimeError(f"Error calling TinyLLM API: {str(e)}") from e
        
# ---------------------------
# Global Variables & State Management
# ---------------------------

class AppState:
    """Container for application state"""
    def __init__(self):
        self.embeddings = None
        self.tokenizer = None
        self.vector_db = None
        self.converter = None
        self.llm = None
        self.document_metadata = {}  # Track document metadata
        
    def is_initialized(self) -> bool:
        return all([self.embeddings, self.tokenizer, self.vector_db, self.converter, self.llm])

app_state = AppState()

# ---------------------------
# Enhanced Initialization Functions
# ---------------------------

def initialize_embedding_model():
    """Initialize embedding model with enhanced error handling"""
    logger.info("Initializing embedding model...")
    try:
        embedding_model_path = os.getenv('EMBEDDING_MODEL_PATH', '/tmp/embeddings')
        Path(embedding_model_path).mkdir(parents=True, exist_ok=True)
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Loading embedding model: {model_name}")
        
        app_state.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, 
            cache_folder=embedding_model_path
        )
        
        app_state.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=embedding_model_path
        )
        
        logger.info("Embedding model initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def initialize_milvus():
    """Initialize Milvus vector database with enhanced configuration"""
    logger.info("Initializing Milvus vector database...")
    try:
        milvus_data_path = os.getenv('MILVUS_DATA_PATH', '/tmp/milvus')
        db_file = os.path.join(milvus_data_path, "milvus.db")
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        
        logger.info(f"Milvus database path: {db_file}")

        app_state.vector_db = Milvus(
            embedding_function=app_state.embeddings,
            connection_args={"uri": db_file},
            collection_name="rag_collection",
            auto_id=True,
            enable_dynamic_field=True,
            index_params={"index_type": "AUTOINDEX"},
            drop_old=False
        )
        
        # Test the connection
        # logger.info("Testing Milvus connection...")
        # collection_info = app_state.vector_db._collection.describe()
        # logger.info(f"Milvus collection info: {collection_info}")
        logger.info("Milvus initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def initialize_converter():
    """Initialize document converter"""
    logger.info("Initializing document converter...")
    try:
        app_state.converter = DocumentConverter()
        logger.info("Document converter initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize document converter: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def initialize_llm():
    """Initialize LLM with custom configuration"""
    logger.info("Initializing LLM...")
    try:
        app_state.llm = TinyLLM()  # No parameters needed since URL is hardcoded
        logger.info("LLM initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        return False

# ---------------------------
# Enhanced Helper Functions
# ---------------------------

def process_document(source: str, is_url: bool = True, max_chunks: int = 100):
    """Enhanced document processing with metadata tracking"""
    logger.info(f"Processing document: {source} (URL: {is_url})")
    texts = []
    doc_id = str(uuid.uuid4())
    
    try:
        # Process document based on source type
        if is_url:
            logger.info(f"Converting URL document: {source}")
            doc_result = app_state.converter.convert(source=source).document
            source_name = source
        else:
            logger.info(f"Converting uploaded file: {source.filename}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{source.filename}") as tmpfile:
                source.save(tmpfile.name)
                doc_result = app_state.converter.convert(source=tmpfile.name).document
                source_name = source.filename
                # Clean up temp file
                os.unlink(tmpfile.name)

        # Chunk the document
        logger.info("Chunking document...")
        chunks_iterator = HybridChunker(tokenizer=app_state.tokenizer).chunk(doc_result)
        
        # Convert iterator to list to get length and enable multiple iterations
        chunks = list(chunks_iterator)
        logger.info(f"Generated {len(chunks)} chunks")

        # Process chunks with enhanced metadata
        processed_chunks = 0
        for i, chunk in enumerate(chunks):
            if processed_chunks >= max_chunks:
                logger.warning(f"Reached maximum chunk limit ({max_chunks}), stopping processing")
                break
                
            # Filter for relevant content
            if any(item.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH] for item in chunk.meta.doc_items):
                chunk_metadata = {
                    "source": source_name,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "processed_at": datetime.now().isoformat(),
                    "chunk_length": len(chunk.text)
                }
                
                texts.append(Document(
                    page_content=chunk.text, 
                    metadata=chunk_metadata
                ))
                processed_chunks += 1

        # Store document metadata
        app_state.document_metadata[source_name] = {
            "doc_id": doc_id,
            "total_chunks": len(texts),
            "processed_at": datetime.now().isoformat(),
            "is_url": is_url,
            "original_source": source
        }

        logger.info(f"Successfully processed document '{source_name}': {len(texts)} chunks created")
        return texts

    except Exception as e:
        logger.error(f"Error processing document '{source}': {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
def get_filtered_retriever(selected_source: str = None, top_k: int = 5):
    """Create a retriever with optional source filtering"""
    logger.info(f"Creating retriever with source filter: {selected_source}, top_k: {top_k}")
    
    search_kwargs = {"k": top_k}
    
    if selected_source:
        # Create metadata filter for specific source
        search_kwargs["expr"] = f"source == '{selected_source}'"
        logger.info(f"Applied source filter: {selected_source}")
    
    return app_state.vector_db.as_retriever(search_kwargs=search_kwargs)

# ---------------------------
# Flask App Setup
# ---------------------------

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])

@app.before_request
def log_request_info():
    """Log incoming requests for debugging"""
    logger.debug(f"Request: {request.method} {request.path}")
    if request.method == 'POST':
        logger.debug(f"Form data keys: {list(request.form.keys())}")
        logger.debug(f"Files: {list(request.files.keys())}")

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal server error",
        "message": str(e) if app.debug else "An unexpected error occurred"
    }), 500

@app.route('/health')
def health_check():
    """Enhanced health check endpoint"""
    logger.debug("Health check requested")
    
    components_status = {
        'embedding_model': app_state.embeddings is not None,
        'tokenizer': app_state.tokenizer is not None,
        'vector_db': app_state.vector_db is not None,
        'converter': app_state.converter is not None,
        'llm': app_state.llm is not None
    }
    
    all_healthy = all(components_status.values())
    
    # Get database stats
    db_stats = {}
    if app_state.vector_db:
        try:
            db_stats = {
                'total_documents': len(app_state.document_metadata),
                'collection_name': app_state.vector_db.collection_name
            }
        except Exception as e:
            logger.warning(f"Could not get DB stats: {str(e)}")
            db_stats = {'error': 'Could not retrieve stats'}
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'unhealthy',
        'components': components_status,
        'database': db_stats,
        'timestamp': datetime.now().isoformat()
    }), 200 if all_healthy else 503

@app.route('/')
def home():
    """Enhanced home endpoint with API documentation"""
    return jsonify({
        'message': 'Enhanced RAG Application',
        'version': '2.0',
        'status': 'running',
        'endpoints': {
            '/health': 'GET - Health check',
            '/list-documents': 'GET - List stored documents',
            '/document-stats': 'GET - Get document statistics', 
            '/process': 'POST - Process documents and answer questions',
            '/upload': 'POST - Upload document only',
            '/ask': 'POST - Ask question only'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/list-documents', methods=['GET'])
def list_documents():
    """Enhanced document listing with metadata"""
    logger.info("Listing documents requested")
    try:
        if not app_state.vector_db:
            logger.error("Vector database not initialized")
            return jsonify({"error": "Vector database not initialized"}), 500
        
        # Get documents from metadata store (more reliable)
        documents = []
        for source, metadata in app_state.document_metadata.items():
            documents.append({
                'source': source,
                'doc_id': metadata['doc_id'],
                'chunks': metadata['total_chunks'],
                'processed_at': metadata['processed_at'],
                'is_url': metadata['is_url']
            })
        
        logger.info(f"Found {len(documents)} documents")
        return jsonify({
            "documents": [doc['source'] for doc in documents],
            "detailed_info": documents,
            "total_count": len(documents)
        })
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/document-stats', methods=['GET'])
def document_stats():
    """Get detailed document statistics"""
    logger.info("Document stats requested")
    try:
        return jsonify({
            "total_documents": len(app_state.document_metadata),
            "documents": app_state.document_metadata
        })
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_document():
    """Separate endpoint for document upload only"""
    logger.info("Document upload requested")
    try:
        if not app_state.is_initialized():
            logger.error("Application components not fully initialized")
            return jsonify({"error": "Application not ready"}), 503

        # Get document source
        document_url = request.form.get('document_url')
        file = request.files.get('file')

        if not (file or document_url):
            return jsonify({"error": "Either document_url or file is required"}), 400

        # Process document
        if file:
            logger.info(f"Processing uploaded file: {file.filename}")
            chunks = process_document(file, is_url=False)
        else:
            logger.info(f"Processing URL: {document_url}")
            chunks = process_document(document_url, is_url=True)

        # Add to vector database
        logger.info(f"Adding {len(chunks)} chunks to vector database")
        ids = app_state.vector_db.add_documents(chunks)
        logger.info(f"Successfully added {len(ids)} document chunks")

        return jsonify({
            "message": "Document uploaded successfully",
            "chunks_added": len(ids),
            "document_id": chunks[0].metadata.get('doc_id') if chunks else None
        })

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Separate endpoint for asking questions only"""
    logger.info("Question asked")
    try:
        if not app_state.is_initialized():
            return jsonify({"error": "Application not ready"}), 503

        question = request.form.get('question')
        if not question:
            return jsonify({"error": "Question is required"}), 400

        selected_source = request.form.get('selected_source')
        top_k = int(request.form.get('top_k', 5))

        logger.info(f"Processing question: '{question[:50]}...' for source: {selected_source}")

        # Create retriever with filtering
        retriever = get_filtered_retriever(selected_source, top_k)

        # Enhanced prompt template
        prompt_template_str = """<|start_of_role|>system<|end_of_role|>
You are an intelligent AI assistant that provides accurate answers based on the given context.

Instructions:
- Answer the question using ONLY the information provided in the context below
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but comprehensive in your responses
- Maintain a helpful and professional tone

Context:
{context}

<|start_of_role|>user<|end_of_role|>
Question: {input}

Please provide a clear and accurate answer based on the context provided above."""

        prompt = PromptTemplate.from_template(prompt_template_str)
        
        # Create chains
        combine_docs_chain = create_stuff_documents_chain(
            llm=app_state.llm,
            prompt=prompt
        )

        rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )

        # Get answer
        logger.info("Generating answer...")
        output = rag_chain.invoke({"input": question})
        
        # Extract source information from retrieved documents
        sources_used = []
        if 'context' in output:
            for doc in output['context']:
                source_info = {
                    'source': doc.metadata.get('source', 'unknown'),
                    'chunk_index': doc.metadata.get('chunk_index', 0)
                }
                if source_info not in sources_used:
                    sources_used.append(source_info)

        logger.info(f"Answer generated successfully, used {len(sources_used)} source chunks")

        return jsonify({
            "answer": output["answer"],
            "sources_used": sources_used,
            "selected_source": selected_source or "all_documents",
            "question": question
        })

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    """Enhanced main processing endpoint (backward compatibility)"""
    logger.info("Process endpoint called")
    try:
        if not app_state.is_initialized():
            logger.error("Application components not fully initialized")
            return jsonify({"error": "Application not ready"}), 503

        question = request.form.get('question')
        if not question:
            return jsonify({"error": "Question is required"}), 400

        document_url = request.form.get('document_url')
        file = request.files.get('file')
        selected_source = request.form.get('selected_source')

        # Process and store document if provided
        if file or document_url:
            if file:
                logger.info(f"Processing uploaded file: {file.filename}")
                chunks = process_document(file, is_url=False)
            else:
                logger.info(f"Processing URL: {document_url}")
                chunks = process_document(document_url, is_url=True)

            logger.info(f"Adding {len(chunks)} chunks to vector database")
            ids = app_state.vector_db.add_documents(chunks)
            logger.info(f"Successfully added {len(ids)} document chunks")

        # Create retriever with filtering
        top_k = int(request.form.get('top_k', 5))
        retriever = get_filtered_retriever(selected_source, top_k)

        # Enhanced prompt template
        prompt_template_str = """<|start_of_role|>system<|end_of_role|>
You are an intelligent AI assistant that provides accurate answers based on the given context.

Instructions:
- Answer the question using ONLY the information provided in the context below
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but comprehensive in your responses
- Maintain a helpful and professional tone

Context:
{context}

<|start_of_role|>user<|end_of_role|>
Question: {input}

Please provide a clear and accurate answer based on the context provided above."""

        prompt = PromptTemplate.from_template(prompt_template_str)
        
        combine_docs_chain = create_stuff_documents_chain(
            llm=app_state.llm,
            prompt=prompt
        )

        rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )

        logger.info("Generating answer...")
        output = rag_chain.invoke({"input": question})
        
        # Extract source information
        sources_used = []
        if 'context' in output:
            for doc in output['context']:
                source_info = {
                    'source': doc.metadata.get('source', 'unknown'),
                    'chunk_index': doc.metadata.get('chunk_index', 0)
                }
                if source_info not in sources_used:
                    sources_used.append(source_info)

        logger.info("Answer generated successfully")

        return jsonify({
            "answer": output["answer"],
            "sources_used": sources_used,
            "selected_source": selected_source or "all_documents"
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Application Startup
# ---------------------------

def initialize_application():
    """Initialize all application components"""
    logger.info("=" * 50)
    logger.info("Starting Enhanced RAG Application...")
    logger.info("=" * 50)
    
    initialization_steps = [
        ("Embedding Model", initialize_embedding_model),
        ("Milvus Database", initialize_milvus),
        ("Document Converter", initialize_converter),
        ("Language Model", initialize_llm)
    ]
    
    for step_name, init_func in initialization_steps:
        logger.info(f"Initializing {step_name}...")
        if not init_func():
            logger.error(f"Failed to initialize {step_name}. Exiting.")
            return False
        logger.info(f"✓ {step_name} initialized successfully")
    
    logger.info("=" * 50)
    logger.info("All components initialized successfully!")
    logger.info("Application is ready to serve requests")
    logger.info("=" * 50)
    return True

if __name__ == '__main__':
    if not initialize_application():
        logger.error("Application initialization failed. Exiting.")
        sys.exit(1)
    
    # Get configuration from environment
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
