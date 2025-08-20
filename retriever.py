from langchain_milvus import Milvus
from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEndpoint
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.parent_document import ParentDocumentRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class RetrieverBase(ABC):
    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    @abstractmethod
    def get_retriever(self):
        """Return a retriever object"""
        pass


# --- 단순 Vector Retriever ---
class VectorRetriever(RetrieverBase):
    def get_retriever(self, k=5):
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# --- CrossEncoder 기반 ContextualCompressionRetriever ---
class CrossEncoderRetriever(RetrieverBase):
    def __init__(self, vector_store, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(vector_store)
        self.reranker_model = reranker_model

    def get_retriever(self, k=5, top_n=3):
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        re_ranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(
                model_name=self.reranker_model, model_kwargs={"trust_remote_code": True}
            ),
            top_n=top_n,
        )
        return ContextualCompressionRetriever(
            base_compressor=re_ranker,
            base_retriever=base_retriever
        )


# --- MultiQueryRetriever ---
class MultiQueryRetrieverWrapper(RetrieverBase):
    def __init__(self, vector_store, llm):
        super().__init__(vector_store)
        self.llm = llm

    def get_retriever(self):
        return MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),
            llm=self.llm
        )


# --- ParentDocumentRetriever ---
class ParentDocRetrieverWrapper(RetrieverBase):
    def __init__(self, vector_store, docstore, child_splitter):
        super().__init__(vector_store)
        self.docstore = docstore
        self.child_splitter = child_splitter

    def get_retriever(self):
        return ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=self.child_splitter
        )
class Retriever(ABC):
    def __init__(self,vector_store):
        self.vector_store = vector_store

    @abstractmethod
    def return_retriever(self,search_number):
        pass
        return self.vector_store.as_retriever(search_kwargs={"k":search_number})
    
    @abstractmethod
    def retriever_with_score(self,query):
        return self.vector_store.similarity_search_with_score(query)




if __name__ == "__main__":
    from vectordb import MilvusDB
    from dotenv import load_dotenv
    import os
    load_dotenv()
    milDB = MilvusDB(os.getenv("EMBED_MODEL"),os.getenv("RERANKER_MODEL"),os.getenv("DB_ADDRESS"))
    retriever = Retriever(milDB.vector_store)
    print(retriever.return_retriever(5))