from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv
import os
class MilvusDB():
    def __init__(self,EMBED_MODEL_ID,RERANKER_MODEL,DB_ADDRESS):
        self.reranker_model = RERANKER_MODEL
        self.db_address = DB_ADDRESS
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

    def searching_vector_store_aka_retriever(self):

        from langchain_huggingface import HuggingFaceEndpoint
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        base_retriever = self.vector_store.as_retriever(search_kwargs={"k":5})
        print(f"모델을 호출합니다.. {self.reranker_model}")
        print("="*40)
        re_ranker = CrossEncoderReranker(
            model = HuggingFaceCrossEncoder(model_name=self.reranker_model, model_kwargs={"trust_remote_code":True}),
                top_n = 3,
        )
        print("="*40)
        print(f"모델을 정상적으로 불러왔습니다. {self.reranker_model}")
        print("="*40)
        cross_encoder_rerank_retiever = ContextualCompressionRetriever(           
            base_compressor=re_ranker,
            base_retriever = base_retriever,
        )
#        retrieved_docs = cross_encoder_rerank_retiever.invoke("박태정에 대해 설명하세요.")
    

        try:
            retrieved_docs = cross_encoder_rerank_retiever.invoke("박태정에 대해 설명하세요.")
        except Exception as e:
            print(f"An error occurred during retrieval: {e}")
            import traceback
            traceback.print_exc() # This will print the full stack trace

        for doc in retrieved_docs:
            print(doc.page_content)
            print(doc.metadata)
            print("="*50)
        return 
    
    

if __name__ =="__main__":
    load_dotenv()
    milDB = MilvusDB(os.getenv("EMBED_MODEL"),os.getenv("RERANKER_MODEL"),os.getenv("DB_ADDRESS"))
    #milDB.insert_vector_store("2501.17887v1.pdf")
    #milDB.searching_vector_store_aka_retriever()
    milDB.searching_vector_store_aka_retriever()
        