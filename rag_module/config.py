import os 
from operator import itemgetter
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from langchain.schema import BaseMessage, format_document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from .connection import es_connection_details
from .prompts import CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT, LLM_CONTEXT_PROMPT
from elasticsearch import Elasticsearch
from icecream import ic
load_dotenv()


ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME", "elastic")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "news_article_embedding")
ES_CA_CERT =os.getenv("ES_CA_CERT")
ic(ES_URL)
ic(ES_INDEX_NAME)
ic(ELASTIC_USERNAME)
ic(ELASTIC_PASSWORD)
ic(ES_CA_CERT)


embeddings=HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask", model_kwargs={"device": "cpu"}
    )

# Setup connecting to Elasticsearch
es = Elasticsearch([ES_URL], basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD), ca_certs=ES_CA_CERT)   
ic(es.info())
vectorstore = ElasticsearchStore(
    index_name=ES_INDEX_NAME,
    embedding=embeddings,
    es_connection=es
)
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings, 
    similarity_threshold=0.5
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
compression_retriever = ContextualCompressionRetriever(
	# embeddings_filter 설정
    base_compressor=embeddings_filter, 
    # retriever 를 호출하여 검색쿼리와 유사한 텍스트를 찾음
    base_retriever=vectorstore.as_retriever()
)
# Set up LLM to user
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
# llm = LlamaCpp(
# 	# model_path: 로컬머신에 다운로드 받은 모델의 위치
#     # model_path="TheBloke/Llama-2-7b-Chat-GGUF",
#     model_path="/data/HF_HOME/hub/models--TheBloke--Llama-2-7b-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q4_K_M.gguf",
#     temperature=0.75,
#     top_p=0.95,
#     max_tokens=8192,
#     verbose=True,
#     # n_ctx: 모델이 한 번에 처리할 수 있는 최대 컨텍스트 길이
#     n_ctx=8192,
#     # n_gpu_layers: 실리콘 맥에서는 1이면 충분하다고 한다
#     n_gpu_layers=1,
#     n_batch=512,
#     f16_kv=True,
#     n_threads=16,
# )


def _combine_documents(
    docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"
):
    ic(docs)
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    ic(doc_strings)
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


class ChainInput(BaseModel):
    chat_history: Optional[List[BaseMessage]] = Field(
        description="Previous chat messages."
    )
    question: str = Field(..., description="The question to answer.")


_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | compression_retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# _context = {
#     "context": itemgetter("standalone_question") | retriever | _combine_documents,
#     "question": lambda x: x["standalone_question"],
# }

chain = _inputs | _context | LLM_CONTEXT_PROMPT | llm | StrOutputParser()

chain = chain.with_types(input_type=ChainInput)
