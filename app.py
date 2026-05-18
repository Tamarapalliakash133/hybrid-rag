import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_community.retrievers import (
    BM25Retriever
)

from langchain_core.retrievers import (
    BaseRetriever
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

from sentence_transformers import (
    CrossEncoder
)

from pydantic import Field
from typing import List

load_dotenv()

OPENAI_API_KEY=os.getenv(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:
    raise Exception(
        "OPENAI_API_KEY missing"
    )



app=Flask(__name__)

CORS(app)

hybrid_retriever=None


#memory using the sessions

store={}


def get_session_history(
    session_id:str
):

    if session_id not in store:

        store[
            session_id
        ]=ChatMessageHistory()

    return store[
        session_id
    ]


#llm openai

llm=ChatOpenAI(

    model="gpt-4o-mini",

    temperature=0

)


reranker=CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# =====================================
# Ensemble Retriever
# =====================================

class EnsembleRetriever(
    BaseRetriever
):

    retrievers:list=Field(
        default_factory=list
    )

    weights:List[float]=Field(
        default_factory=list
    )


    def _get_relevant_documents(
        self,
        query:str
    ):

        seen={}

        for retriever,weight in zip(

            self.retrievers,

            self.weights

        ):

            docs=retriever.invoke(
                query
            )

            for doc in docs:

                key=doc.page_content

                if key not in seen:

                    seen[key]=(
                        doc,
                        weight
                    )

                else:

                    seen[key]=(

                        seen[key][0],

                        seen[key][1]+weight
                    )


        sorted_docs=sorted(

            seen.values(),

            key=lambda x:x[1],

            reverse=True
        )


        return[

            doc

            for doc,_ in sorted_docs

        ]




refine_prompt=(
    ChatPromptTemplate
    .from_template("""

Rewrite only if needed.

Preserve meaning.

Resolve:

it
that
this
previous topic

Question:

{question}

""")
)


query_refiner=(

    refine_prompt

    |llm

    |StrOutputParser()

)


def should_refine(query):

    words=[

        "it",
        "that",
        "this",
        "tell",
        "explain"

    ]

    query=query.lower()

    return any(
        x in query
        for x in words
    )




answer_prompt=(

ChatPromptTemplate
.from_messages([

(

"system",

"""

You are a helpful assistant.

Use PDF context first.

If answer exists in PDF:
answer using PDF.

If answer is not in PDF:
answer from AI knowledge.

Mention:

(Generated from AI knowledge)

Give detailed answers.


PDF Context:

{context}

"""

),

MessagesPlaceholder(

variable_name=
"history"

),

(

"human",

"{question}"

)

])

)




def format_docs(docs):

    return "\n\n".join([

        d.page_content

        for d in docs

    ])




def rerank(
    query,
    docs
):

    if not docs:

        return [],0.0


    pairs=[

        [query,d.page_content]

        for d in docs

    ]


    scores=reranker.predict(
        pairs
    )


    ranked=sorted(

        zip(
            docs,
            scores
        ),

        key=lambda x:x[1],

        reverse=True
    )


    top_docs=[

        doc

        for doc,_ in ranked[:6]

    ]


    top_scores=sorted(

        scores,

        reverse=True

    )[:3]


    confidence=float(

        sum(top_scores)

        /

        len(top_scores)

    )


    return(

        top_docs,

        confidence

    )


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =====================================
# Upload PDF
# =====================================

@app.route(
    "/upload",
    methods=["POST"]
)

def upload_pdf():

    global hybrid_retriever


    file=request.files.get(
        "file"
    )


    if not file:

        return jsonify({

            "error":
            "No PDF uploaded"

        }),400


    filepath=(
        f"temp_{file.filename}"
    )


    file.save(
        filepath
    )


    try:

        loader=PyPDFLoader(
            filepath
        )

        documents=loader.load()


        splitter=(
            RecursiveCharacterTextSplitter(

                chunk_size=1500,

                chunk_overlap=300

            )
        )


        docs=(
            splitter
            .split_documents(
                documents
            )
        )


        embeddings=(
            OpenAIEmbeddings(

                model=
                "text-embedding-3-small"

            )
        )


        vectorstore=(
            FAISS.from_documents(

                docs,

                embeddings

            )
        )


        vector_retriever=(

            vectorstore
            .as_retriever(

                search_kwargs={

                    "k":15

                }

            )

        )


        bm25=(
            BM25Retriever
            .from_documents(
                docs
            )
        )


        bm25.k=15


        hybrid_retriever=(
            EnsembleRetriever(

                retrievers=[

                    vector_retriever,

                    bm25

                ],

                weights=[

                    0.7,

                    0.3

                ]

            )
        )


        return jsonify({

            "message":
            "PDF uploaded successfully"

        })


    finally:

        if os.path.exists(
            filepath
        ):

            os.remove(
                filepath
            )


@app.route(
    "/ask",
    methods=["POST"]
)

def ask():

    global hybrid_retriever

    try:

        data=request.get_json()

        query=data.get(
            "query"
        )

        session_id=data.get(
            "session_id",
            "default"
        )


        if not query:

            return jsonify({

                "error":
                "Question missing"

            }),400


        refined_query=query


        if should_refine(query):

            refined_query=(
                query_refiner.invoke({

                    "question":
                    query

                })
            )


        context=""

        source="AI Knowledge"

        confidence=0


        if hybrid_retriever:

            docs=(
                hybrid_retriever
                .invoke(
                    refined_query
                )
            )


            docs,confidence=(
                rerank(
                    refined_query,
                    docs
                )
            )


            if docs:

                context=(
                    format_docs(
                        docs
                    )
                )

                source="PDF"



        base_chain=(

            answer_prompt

            |llm

            |StrOutputParser()

        )


        chain=RunnableWithMessageHistory(

            base_chain,

            get_session_history,

            input_messages_key=
            "question",

            history_messages_key=
            "history"

        )


        answer=chain.invoke(

            {

                "context":
                context,

                "question":
                refined_query

            },

            config={

                "configurable":{

                    "session_id":
                    session_id

                }

            }

        )


        return jsonify({

            "answer":
            answer,

            "source":
            source,

            "confidence":
            float(
                round(
                    confidence,
                    2
                )
            ),

            "refined_query":
            refined_query

        })


    except Exception as e:

        return jsonify({

            "error":
            str(e)

        }),500



if __name__=="__main__":

    app.run(
        debug=True
    )