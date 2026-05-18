import os
from dotenv import load_dotenv
from flask import Flask,request,jsonify,render_template
from flask_cors import CORS

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
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

from sentence_transformers import (
    CrossEncoder
)

from lettucedetect import (
    TransformerDetector
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

store={}


################################################
# MEMORY
################################################

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


################################################
# LLM
################################################

llm=ChatOpenAI(

    model="gpt-4o-mini",

    temperature=0
)


################################################
# RERANKER
################################################

reranker=CrossEncoder(

"cross-encoder/ms-marco-MiniLM-L-6-v2"

)


################################################
# HALLUCINATION DETECTOR
################################################

hallu_detector=TransformerDetector(

model_path=
"KRLabsOrg/lettucedetect-roberta-base"

)


################################################
# HYBRID RETRIEVER
################################################

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
        query
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


        ranked=sorted(

            seen.values(),

            key=lambda x:x[1],

            reverse=True

        )


        return[

            doc

            for doc,_ in ranked

        ]


################################################
# QUERY REFINER
################################################

refiner_prompt=(

ChatPromptTemplate
.from_template(

"""

Rewrite only if required.

Resolve references:

it
this
that

Question:

{question}

"""

)

)


query_refiner=(

refiner_prompt
|llm
|StrOutputParser()

)


def should_refine(query):

    query=query.lower()

    words=[

        "it",
        "that",
        "this",
        "explain",
        "tell"

    ]

    return any(

        x in query

        for x in words

    )


################################################
# ANSWER PROMPT
################################################

answer_prompt=(

ChatPromptTemplate
.from_messages([

(

"system",

"""

You are an intelligent PDF assistant.

Rules:

Use PDF context first.

Never invent facts.

If context insufficient:

Use AI knowledge.

Mention:

(Generated from AI knowledge)

If confidence low:

Say:

Information unavailable.


Context:

{context}

Confidence:

{confidence}

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


################################################
# DOC FORMATTER
################################################

def format_docs(docs):

    return "\n\n".join([

        d.page_content

        for d in docs

    ])


################################################
# RERANK
################################################

def rerank(
query,
docs
):


    if not docs:

        return [],0


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


    top=[

        doc

        for doc,_ in ranked[:6]

    ]


    confidence=float(

        sum(
            sorted(
                scores,
                reverse=True
            )[:3]
        )/3

    )


    return(
        top,
        confidence
    )


################################################
# HALLUCINATION CHECK
################################################

def detect_hallucination(

question,
context,
answer

):


    try:

        result=hallu_detector.detect(

            question=question,

            context=context,

            answer=answer

        )

        return result[
            "score"
        ]

    except:

        return 0


################################################
# PDF UPLOAD
################################################

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
            "No pdf"

        })


    filepath=f"temp_{file.filename}"

    file.save(filepath)


    try:


        loader=PyPDFLoader(
            filepath
        )

        documents=loader.load()


        splitter=(

        RecursiveCharacterTextSplitter(

            chunk_size=700,

            chunk_overlap=150

        )

        )


        docs=splitter.split_documents(
            documents
        )


        embeddings=OpenAIEmbeddings(

            model=
            "text-embedding-3-small"

        )


        vectorstore=FAISS.from_documents(

            docs,

            embeddings

        )


        dense=vectorstore.as_retriever(

            search_kwargs={

                "k":30,

                "fetch_k":60

            }

        )


        sparse=BM25Retriever.from_documents(
            docs
        )

        sparse.k=30


        hybrid_retriever=EnsembleRetriever(

            retrievers=[

                dense,
                sparse

            ],

            weights=[

                0.7,
                0.3

            ]

        )


        return jsonify({

            "message":
            "uploaded"

        })


    finally:

        os.remove(
            filepath
        )


################################################
# ASK
################################################

@app.route(
"/ask",
methods=["POST"]
)

def ask():

    global hybrid_retriever

    try:

        data=request.json

        query=data.get(
            "query"
        )

        session=data.get(
            "session_id",
            "default"
        )


        if not query:

            return jsonify({

            "error":
            "missing question"

            })


        refined=query


        if should_refine(query):

            refined=query_refiner.invoke({

                "question":
                query

            })


        context=""

        confidence=0


        if hybrid_retriever:


            docs=hybrid_retriever.invoke(
                refined
            )


            docs,confidence=rerank(

                refined,
                docs

            )


            context=format_docs(
                docs
            )


        chain=(

        answer_prompt
        |llm
        |StrOutputParser()

        )


        chain=RunnableWithMessageHistory(

            chain,

            get_session_history,

            input_messages_key=
            "question",

            history_messages_key=
            "history"

        )


        answer=chain.invoke(

        {

            "question":
            refined,

            "context":
            context,

            "confidence":
            confidence

        },

        config={

        "configurable":{

        "session_id":
        session

        }

        }

        )


        hallu_score=0


        if context:

            hallu_score=detect_hallucination(

                refined,

                context,

                answer

            )


        if hallu_score>.7:

            answer="""
Potential hallucination detected.

I couldn't verify this answer
from PDF context.

Upload more information.
"""


        return jsonify({

        "answer":
        answer,

        "confidence":
        round(
            confidence,
            2
        ),

        "hallucination":
        round(
            hallu_score,
            2
        ),

        "refined":
        refined

        })


    except Exception as e:

        return jsonify({

        "error":
        str(e)

        })


if __name__=="__main__":

    app.run(
        debug=True
    )