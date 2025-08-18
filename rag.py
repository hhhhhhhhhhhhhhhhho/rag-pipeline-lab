import os
from pprint import pprint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType


class Rag_pipeline():
       def __init__(): 
            '''
            rag pipeline 에 필요한 라이브러리들의 세팅값들을 설정합니다.
            '''
            
            load_dotenv()
            os.environ["TOKENZIERS_PARALLELISM"] = "false"

            FILE_PATH = ["2501.17887v1.pdf"]
            RERANKER_MODEL_ID = "Alibaba-NLP/gte-multilingual-reranker-base"
            EXPORT_TYPE = ExportType.DOC_CHUNKS

            RAG_PROMPT_TEMPLATE ="""
            context information is below.
            ---------------
            {context}
            ---------------
            Given the context information and not prior knowlege, asnwer the query.
            Query: {query}
            Answer: {in Korean}
            """

            RAG_PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

       