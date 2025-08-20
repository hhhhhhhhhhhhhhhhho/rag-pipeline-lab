from langchain_milvus import Milvus
from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEndpoint
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.parent_document import ParentDocumentRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from enum import Enum
from typing import Optional, Dict, Any


class RetrieverType(Enum):
    """Retriever 타입을 정의하는 Enum"""
    VECTOR = "vector"
    CROSS_ENCODER = "cross_encoder"
    MULTI_QUERY = "multi_query"
    PARENT_DOC = "parent_doc"


class RetrieverBase(ABC):
    """Retriever의 기본 추상 클래스"""
    
    def __init__(self, vector_store=None, **kwargs):
        self.vector_store = vector_store
        self.config = kwargs

    @abstractmethod
    def get_retriever(self, **kwargs):
        """Retriever 객체를 반환"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Retriever 이름을 반환"""
        pass


class VectorRetriever(RetrieverBase):
    """단순 Vector Retriever"""
    
    def get_retriever(self, k=5, **kwargs):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def get_name(self) -> str:
        return "VectorRetriever"


class CrossEncoderRetriever(RetrieverBase):
    """CrossEncoder 기반 ContextualCompressionRetriever"""
    
    def __init__(self, vector_store, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs):
        super().__init__(vector_store, **kwargs)
        self.reranker_model = reranker_model

    def get_retriever(self, k=5, top_n=3, **kwargs):
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        re_ranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(
                model_name=self.reranker_model, 
                model_kwargs={"trust_remote_code": True}
            ),
            top_n=top_n,
        )
        return ContextualCompressionRetriever(
            base_compressor=re_ranker,
            base_retriever=base_retriever
        )
    
    def get_name(self) -> str:
        return "CrossEncoderRetriever"


class MultiQueryRetrieverWrapper(RetrieverBase):
    """MultiQueryRetriever 래퍼"""
    
    def __init__(self, vector_store, llm, **kwargs):
        super().__init__(vector_store, **kwargs)
        self.llm = llm

    def get_retriever(self, **kwargs):
        return MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),
            llm=self.llm
        )
    
    def get_name(self) -> str:
        return "MultiQueryRetriever"


class ParentDocRetrieverWrapper(RetrieverBase):
    """ParentDocumentRetriever 래퍼"""
    
    def __init__(self, vector_store, docstore, child_splitter, **kwargs):
        super().__init__(vector_store, **kwargs)
        self.docstore = docstore
        self.child_splitter = child_splitter

    def get_retriever(self, **kwargs):
        return ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=self.child_splitter
        )
    
    def get_name(self) -> str:
        return "ParentDocRetriever"


class RetrieverFactory:
    """실험용 Retriever Factory 클래스"""
    
    def __init__(self):
        self._retrievers = {}
        self._current_retriever = None
        self._current_config = {}
    
    def register_retriever(self, retriever_type: RetrieverType, retriever_class: type):
        """Retriever 타입을 등록"""
        self._retrievers[retriever_type] = retriever_class
    
    def create_retriever(self, 
                        retriever_type: RetrieverType, 
                        vector_store, 
                        **kwargs) -> RetrieverBase:
        """Retriever 인스턴스 생성"""
        if retriever_type not in self._retrievers:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        retriever_class = self._retrievers[retriever_type]
        retriever = retriever_class(vector_store, **kwargs)
        
        # 현재 설정 저장
        self._current_retriever = retriever
        self._current_config = {
            'type': retriever_type,
            'kwargs': kwargs
        }
        
        return retriever
    
    def switch_retriever(self, 
                        retriever_type: RetrieverType, 
                        vector_store, 
                        **kwargs) -> RetrieverBase:
        """실험 중 Retriever 타입 변경"""
        print(f"🔄 Switching retriever from {self._current_retriever.get_name() if self._current_retriever else 'None'} to {retriever_type.value}")
        return self.create_retriever(retriever_type, vector_store, **kwargs)
    
    def get_current_retriever(self) -> Optional[RetrieverBase]:
        """현재 Retriever 반환"""
        return self._current_retriever
    
    def get_current_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        return self._current_config.copy()
    
    def list_available_retrievers(self) -> list:
        """사용 가능한 Retriever 목록 반환"""
        return list(self._retrievers.keys())


# Factory 인스턴스 생성 및 기본 Retriever 등록
retriever_factory = RetrieverFactory()
retriever_factory.register_retriever(RetrieverType.VECTOR, VectorRetriever)
retriever_factory.register_retriever(RetrieverType.CROSS_ENCODER, CrossEncoderRetriever)
retriever_factory.register_retriever(RetrieverType.MULTI_QUERY, MultiQueryRetrieverWrapper)
retriever_factory.register_retriever(RetrieverType.PARENT_DOC, ParentDocRetrieverWrapper)


# 실험용 편의 함수들
def create_experiment_retriever(retriever_type: str, vector_store, **kwargs) -> RetrieverBase:
    """실험용 Retriever 생성 편의 함수"""
    try:
        retriever_enum = RetrieverType(retriever_type.lower())
        return retriever_factory.create_retriever(retriever_enum, vector_store, **kwargs)
    except ValueError:
        raise ValueError(f"Invalid retriever type: {retriever_type}. Available types: {[t.value for t in RetrieverType]}")


def switch_experiment_retriever(retriever_type: str, vector_store, **kwargs) -> RetrieverBase:
    """실험 중 Retriever 변경 편의 함수"""
    try:
        retriever_enum = RetrieverType(retriever_type.lower())
        return retriever_factory.switch_retriever(retriever_enum, vector_store, **kwargs)
    except ValueError:
        raise ValueError(f"Invalid retriever type: {retriever_type}. Available types: {[t.value for t in RetrieverType]}")


# 기존 호환성을 위한 클래스 (deprecated)
class Retriever(ABC):
    """기존 호환성을 위한 클래스 (deprecated)"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store

    @abstractmethod
    def return_retriever(self, search_number):
        pass
    
    @abstractmethod
    def retriever_with_score(self, query):
        return self.vector_store.similarity_search_with_score(query)


if __name__ == "__main__":
    from vectordb import MilvusDB
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    # Vector DB 초기화
    milDB = MilvusDB(os.getenv("EMBED_MODEL"), os.getenv("RERANKER_MODEL"), os.getenv("DB_ADDRESS"))
    
    # 실험 1: Vector Retriever
    print("🧪 Experiment 1: Vector Retriever")
    vector_retriever = create_experiment_retriever("vector", milDB.vector_store, k=5)
    print(f"Created: {vector_retriever.get_name()}")
    
    # 실험 2: CrossEncoder Retriever로 변경
    print("\n🧪 Experiment 2: CrossEncoder Retriever")
    cross_encoder_retriever = switch_experiment_retriever(
        "cross_encoder", 
        milDB.vector_store, 
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        k=5,
        top_n=3
    )
    print(f"Switched to: {cross_encoder_retriever.get_name()}")
    
    # 실험 3: MultiQuery Retriever로 변경 (LLM이 필요한 경우)
    print("\n🧪 Experiment 3: MultiQuery Retriever")
    # LLM 인스턴스가 필요하므로 주석 처리
    # multi_query_retriever = switch_experiment_retriever(
    #     "multi_query", 
    #     milDB.vector_store, 
    #     llm=some_llm_instance
    # )
    
    # 현재 설정 확인
    print(f"\n📊 Current Configuration: {retriever_factory.get_current_config()}")
    print(f"📋 Available Retrievers: {[t.value for t in retriever_factory.list_available_retrievers()]}")