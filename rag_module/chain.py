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
from .prompts import CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT, LLM_CONTEXT_PROMPT
from elasticsearch import Elasticsearch
from icecream import ic
load_dotenv()
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import logging
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


# for metadata in es 
# "title": "롯데홈쇼핑, 취미·자기관리 상품 늘린다...\"MZ세대 공략\"",
# "created_date": "2024-02-05T09:28:29",
# "portal": "daum",
# "media": "파이낸셜뉴스",
# "url": "https://v.daum.net/v/20240205092829939",
# "image_url": "https://img1.daumcdn.net/thumb/S1200x630/?fname=https://t1.daumcdn.net/news/202402/05/fnnewsi/20240205092829561edjr.jpg"
            
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The title of the article or document",
        type="string",
    ),
    AttributeInfo(
        name="created_date",
        description="The creation date of the article or document. That field format is 'YYYY-MM-DD' ",
        type="date",
    ),
    
    AttributeInfo(
        name="portal",
        description="portal of the article or document. It can be daum, naver, etc",
        type="string",
    ),
    AttributeInfo(
        name="media", description="The newspaper that created the article", type="String"
    ),
    AttributeInfo(
        name="url", description="The URL address where the data was posted", type="String"
    ),
    AttributeInfo(
        name="image_url", description="The URL of the image included in the article", type="String"
    ),
]
document_content_description = "An article published by an internet media or newspaper"
llm = ChatOpenAI(model="gpt-4", temperature=0)

embeddings=HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask", model_kwargs={"device": "cpu"}
    )

# Setup connecting to Elasticsearch
ic("create es instance")
            
vectorstore = ElasticsearchStore(
    index_name=ES_INDEX_NAME,
    embedding=embeddings,
    es_url=ES_URL,
    es_user=ELASTIC_USERNAME,
    es_password=ELASTIC_PASSWORD,
    es_params={
        "ca_certs":ES_CA_CERT,
        "sniff_on_connection_fail":True,
        # "sniff_on_start":True, 
        "min_delay_between_sniffing":600,
        "request_timeout":600, 
        "sniff_timeout":300,
        "max_retries":5, 
        "retry_on_timeout":True
        }
)
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings, 
    similarity_threshold=0.5
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50})

self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
)

compression_retriever = ContextualCompressionRetriever(
	# embeddings_filter 설정
    base_compressor=embeddings_filter, 
    # retriever 를 호출하여 검색쿼리와 유사한 텍스트를 찾음
    base_retriever=vectorstore.as_retriever()
)

# Set up LLM to user
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

def _combine_documents(
    docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"
):
    # ic(docs)
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    # ic(doc_strings)
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
    "context": itemgetter("standalone_question") | self_query_retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

# _context = {
#     "context": itemgetter("standalone_question") | compression_retriever | _combine_documents,
#     "question": lambda x: x["standalone_question"],
# }


# _context = {
#     "context": itemgetter("standalone_question") | retriever | _combine_documents,
#     "question": lambda x: x["standalone_question"],
# }

chain = _inputs | _context | LLM_CONTEXT_PROMPT | llm | StrOutputParser()

chain = chain.with_types(input_type=ChainInput)
