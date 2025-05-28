import sys
import json
import requests
from typing import Any, List, Optional, Dict
from io import BytesIO
import tempfile
import os
from flask import Flask, request, jsonify
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from transformers import AutoTokenizer
import urllib3
import logging
from pathlib import Path

# Disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------
# TinyLLM Wrapper
# ---------------------------

class TinyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "tinyllm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "messages": [{"role": "user", "content": prompt}],
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

            result = response.json()
            if "choices" not in result:
                raise ValueError(f"'choices' not found in LLM response: {result}")
            content = result["choices"][0]["message"].get("content")
            if not content:
                raise ValueError("Missing 'content' in message")
            return content

        except Exception as e:
            raise RuntimeError(f"Error calling TinyLLM API: {str(e)}")

# ---------------------------
# Global Variables
# ---------------------------

embeddings = None
tokenizer = None
vector_db = None
converter = None

def initialize_embedding_model():
    """Initialize embedding model from persistent storage"""
    global embeddings, tokenizer
    
    embedding_model_path = os.getenv('EMBEDDING_MODEL_PATH', '/mnt/embeddings')
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    try:
        # Ensure embedding directory exists
        Path(embedding_model_path).mkdir(parents=True, exist_ok=True)
        
        # Check if model exists in persistent storage
        model_cache_path = Path(embedding_model_path) / "models--sentence-transformers--all-MiniLM-L6-v2"
        
        if model_cache_path.exists():
            print(f"Loading embedding model from cache: {embedding_model_path}")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=embedding_model_path
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=embedding_model_path
            )
        else:
            print(f"Downloading embedding model to: {embedding_model_path}")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=embedding_model_path
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=embedding_model_path
            )
            print("Embedding model downloaded and cached successfully")
        
        return True
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return False

def initialize_milvus():
    """Initialize Milvus vector database"""
    global vector_db
    
    try:
        milvus_data_path = os.getenv('MILVUS_DATA_PATH', '/mnt/milvus')
        db_file = os.path.join(milvus_data_path, "milvus.db")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        
        print(f"Initializing Milvus database at: {db_file}")
        
        vector_db = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": db_file},
            auto_id=True,
            enable_dynamic_field=True,
            index_params={"index_type": "AUTOINDEX"},
        )
        
        print("Milvus vector database initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing Milvus: {e}")
        return False

def initialize_converter():
    """Initialize document converter"""
    global converter
    try:
        converter = DocumentConverter()
        print("Document converter initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing document converter: {e}")
        return False

# ---------------------------
# Flask App Setup
# ---------------------------

app = Flask(__name__)

def process_document(source: str, is_url: bool = True):
    """Process document and return chunks"""
    texts = []
    doc_id = 0

    try:
        if is_url:
            print(f"Processing URL: {source}")
            doc_result = converter.convert(source=source).document
        else:
            print(f"Processing uploaded file")
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                source.save(tmpfile.name)
                doc_result = converter.convert(source=tmpfile.name).document

        chunks = HybridChunker(tokenizer=tokenizer).chunk(doc_result)

        for chunk in chunks:
            if any(item.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH] for item in chunk.meta.doc_items):
                texts.append(
                    Document(page_content=chunk.text, metadata={"doc_id": (doc_id + 1), "source": source})
                )
        doc_id += 1

        return texts

    except Exception as e:
        raise RuntimeError(f"Error processing document: {str(e)}")

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'embedding_model_loaded': embeddings is not None,
        'tokenizer_loaded': tokenizer is not None,
        'vector_db_initialized': vector_db is not None,
        'converter_initialized': converter is not None,
        'embedding_path': os.getenv('EMBEDDING_MODEL_PATH', '/mnt/embeddings'),
        'milvus_path': os.getenv('MILVUS_DATA_PATH', '/mnt/milvus')
    })

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'RAG Application is running',
        'endpoints': {
            'health': '/health',
            'process': '/process (POST)'
        }
    })

@app.route('/process', methods=['POST'])
def process():
    """Main processing endpoint for RAG"""
    try:
        # Check if all components are initialized
        if not all([embeddings, tokenizer, vector_db, converter]):
            return jsonify({"error": "Application components not fully initialized"}), 500

        data = request.form
        question = data.get('question')
        if not question:
            return jsonify({"error": "Question is required"}), 400

        document_url = data.get('document_url')
        file = request.files.get('file')

        if not document_url and not file:
            return jsonify({"error": "Either document_url or file is required"}), 400

        # Process document
        if file:
            chunks = process_document(file, is_url=False)
        else:
            chunks = process_document(document_url, is_url=True)

        # Add documents to vector database
        ids = vector_db.add_documents(chunks)
        print(f"{len(ids)} documents added to the vector database")

        # Create retriever
        retriever = vector_db.as_retriever()

        # Define prompt template
        prompt_template_str = """<|start_of_role|>system<|end_of_role|>
You are an AI assistant that answers questions based on provided context.
Answer the question based only on the following context:
{context}

<|start_of_role|>user<|end_of_role|>
{input}"""

        prompt = PromptTemplate.from_template(prompt_template_str)

        document_prompt = PromptTemplate.from_template(
            "<|start_of_role|>document{{\"id\": \"{doc_id}\"}}<|end_of_role|>\n{page_content}"
        )

        # Create chains
        combine_docs_chain = create_stuff_documents_chain(
            llm=TinyLLM(),
            prompt=prompt,
            document_prompt=document_prompt,
            document_separator="\n\n"
        )

        rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )

        # Generate answer
        output = rag_chain.invoke({"input": question})
        return jsonify({"answer": output["answer"]})

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting RAG application...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize all components
    print("Initializing embedding model...")
    if not initialize_embedding_model():
        print("Failed to initialize embedding model. Exiting...")
        sys.exit(1)
    
    print("Initializing Milvus database...")
    if not initialize_milvus():
        print("Failed to initialize Milvus. Exiting...")
        sys.exit(1)
    
    print("Initializing document converter...")
    if not initialize_converter():
        print("Failed to initialize document converter. Exiting...")
        sys.exit(1)
    
    print("All components initialized successfully!")
    print("Starting Flask application...")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)
