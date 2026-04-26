import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("api-key is not set")


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pydantic import Field
from typing import List


class EnsembleRetriever(BaseRetriever):
    retrievers: list = Field(default_factory=list)
    weights: List[float] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str):
        seen = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for doc in docs:
                key = doc.page_content
                if key not in seen:
                    seen[key] = (doc, weight)
                else:
                    seen[key] = (seen[key][0], seen[key][1] + weight)

        sorted_docs = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]


app = Flask(__name__)
CORS(app)


hybrid_retriever = None

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the context below.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.route("/upload", methods=["POST"])
def upload_pdf():
    global hybrid_retriever

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = f"temp_{file.filename}"
    file.save(filepath)


    loader = PyPDFLoader(filepath)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Embeddings + FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4


    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    return jsonify({"message": "PDF uploaded & processed successfully!"})


@app.route("/ask", methods=["POST"])
def ask():
    global hybrid_retriever

    if hybrid_retriever is None:
        return jsonify({"error": "Upload a PDF first"}), 400

    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query missing"}), 400

    try:
        chain = (
            {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(query)
        return jsonify({"answer": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)