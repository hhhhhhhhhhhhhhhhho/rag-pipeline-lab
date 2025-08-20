from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import os
from document_parser import pdfPlumber

class VectorDB(ABC):
    def __init__(self):
        self.vector_store = None
        self.embedding_model = None
        self.reranker_model = None
        self.db_address = None
        self.is_localDB = False
        self.db_name = None
        
    @abstractmethod
    def insert_vector_store(self,document):
        pass
    

class MilvusDB(VectorDB):
    def __init__(self,EMBED_MODEL_ID,RERANKER_MODEL,DB_ADDRESS):
        self.reranker_model = RERANKER_MODEL
        self.db_address = DB_ADDRESS
        self.is_localDB = False
        #self.langfuse_handler = CallbackHandler()
        #self.langfuse = Langfuse()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID, 
            model_kwargs={"trust_remote_code":True}
         )
        self.vector_store = Milvus(
            embedding_function = EMBED_MODEL_ID,
            connection_args={"uri": DB_ADDRESS}
        )

    def return_address(self):
        return self.db_address
    
    def insert_vector_store(self,document):
        try:
            '''
            document parser 를 사용하여 얻은 document 객체를 document 로 넣어줘야 함.
            '''
            # Milvus 설정
            self.vector_store.from_documents(
                documents=document,
                ### 새로 입력 될 파일은 어떻게 벡터 DB에 추가하는지 확인해보자.
                ### Vector DB에 등록 된             
                embedding=self.embedding_model,
                collection_name="docling_transformer",
                connection_args={"uri": self.db_address,
                                "db_name":"default"
                                },
                index_params={
                    "index_type":"FLAT",
                    "metric_type":"COSINE",
                },
                drop_old=False
            )
            print(f"{document} 가 {self.db_address}에 정상적으로 입력되었습니다.")
        except Exception:
            raise(Exception)

    

if __name__ =="__main__":
    load_dotenv()
    milDB = MilvusDB(os.getenv("EMBED_MODEL"),os.getenv("RERANKER_MODEL"),os.getenv("DB_ADDRESS"))
    parser = pdfPlumber().document_parsing("박태정_Cv.pdf")
    milDB.insert_vector_store(parser)
    #milDB.searching_vector_store_aka_retriever()
#    milDB.searching_vector_store_aka_retriever()
        