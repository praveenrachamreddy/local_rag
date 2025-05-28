import sys
import json
import requests
from typing import Any, List, Optional, Dict
from io import BytesIO
import tempfile

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
# Initialize Embedding Model
# ---------------------------

embeddings_model_path = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_path)
tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

# ---------------------------
# Initialize Vector DB
# ---------------------------

# Instead of using tempfile...
# db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
# Use a fixed path inside container
db_file = "/app/milvusdb/milvus.db"
print(f"The vector database will be saved to {db_file}")

vector_db = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": db_file},
    auto_id=True,
    enable_dynamic_field=True,
    index_params={"index_type": "AUTOINDEX"},
)

# ---------------------------
# Flask App Setup
# ---------------------------

app = Flask(__name__)
converter = DocumentConverter()

def process_document(source: str, is_url: bool = True):
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


@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.form
        question = data.get('question')
        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Check for document URL or file upload
        document_url = data.get('document_url')
        file = request.files.get('file')

        if not document_url and not file:
            return jsonify({"error": "Either document_url or file is required"}), 400

        # Process document
        if file:
            chunks = process_document(file, is_url=False)
        else:
            chunks = process_document(document_url, is_url=True)

        # Add to vector DB
        ids = vector_db.add_documents(chunks)
        print(f"{len(ids)} documents added to the vector database")

        # Build RAG chain
        retriever = vector_db.as_retriever()

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

        output = rag_chain.invoke({"input": question})
        return jsonify({"answer": output["answer"]})

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000)
