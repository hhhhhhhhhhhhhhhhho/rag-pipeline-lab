from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv
import os
class MilvusDB():
    def __init__(self,EMBED_MODEL_ID,DB_ADDRESS):
        self.db_address = DB_ADDRESS
        self.langfuse_handler = CallbackHandler()
        self.langfuse = Langfuse()
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID, model_kwargs={"trust_remote_code":True})

    def return_address(self):
        return self.db_address
    
    def insert_vector_store(self,document):
        '''
        document parser 를 사용하여 얻은 document 객체를 document 로 넣어줘야 함.
        '''
        # Milvus 설정
        vectorstore = Milvus.from_documents(
            documents=document,
            ### 새로 입력 될 파일은 어떻게 벡터 DB에 추가하는지 확인해보자.
            ### Vector DB에 등록 된             
            embedding=self.embedding_model,
            collection_name="docling_transformer",
            connection_args={"uri": self.db_address,
                            "db_name":"edu"
                            },
            index_params={
                "index_type":"FLAT",
                "metric_type":"COSINE",
            },
            drop_old=True
        )
        print(f"{document} 가 {self.db_address}에 정상적으로 입력되었습니다.")

if __name__ =="__main__":
    load_dotenv()
    milDB = MilvusDB(os.getenv("EMBED_MODEL"),os.getenv("DB_ADDRES"))
    milDB.insert_vector_store("2501.17887v1.pdf")