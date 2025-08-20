#!/usr/bin/env python3
"""
RAG Pipeline Lab - 실험용 Factory 패턴 예제

이 파일은 Factory 패턴을 사용하여 다양한 Vector DB와 Retriever 조합을
쉽게 실험할 수 있도록 구성된 예제입니다.
"""

import os
from dotenv import load_dotenv
from document_parser import pdfPlumber, Docling
from vectordb import (
    create_experiment_vector_db, 
    switch_experiment_vector_db,
    vector_db_factory
)
from retriever import (
    create_experiment_retriever,
    switch_experiment_retriever,
    retriever_factory
)
from llm import Ollama_Models


class RAGExperiment:
    """RAG 실험을 위한 클래스"""
    
    def __init__(self):
        load_dotenv()
        self.current_vector_db = None
        self.current_retriever = None
        self.documents = None
        
    def load_documents(self, file_path: str, parser_type: str = "pdfplumber"):
        """문서 로드"""
        print(f"📄 Loading documents from: {file_path}")
        
        if parser_type.lower() == "pdfplumber":
            parser = pdfPlumber()
        elif parser_type.lower() == "docling":
            parser = Docling()
        else:
            raise ValueError(f"Unknown parser type: {parser_type}")
        
        self.documents = parser.document_parsing(file_path, 1000, 10)
        print(f"✅ Loaded {len(self.documents)} documents")
        return self.documents
    
    def experiment_vector_db_comparison(self):
        """Vector DB 비교 실험"""
        print("\n" + "="*60)
        print("🧪 VECTOR DB COMPARISON EXPERIMENT")
        print("="*60)
        
        # 실험 1: Milvus
        print("\n🔍 Experiment 1: Milvus DB")
        milvus_db = create_experiment_vector_db(
            "milvus",
            EMBED_MODEL_ID=os.getenv("EMBED_MODEL"),
            RERANKER_MODEL=os.getenv("RERANKER_MODEL"),
            DB_ADDRESS=os.getenv("DB_ADDRESS")
        )
        self.current_vector_db = milvus_db
        print(f"✅ Created: {milvus_db.get_name()}")
        print(f"📊 Connection Info: {milvus_db.get_connection_info()}")
        
        # 문서 삽입
        if self.documents:
            milvus_db.insert_vector_store(self.documents)
        
        # 실험 2: Chroma로 변경
        print("\n🔍 Experiment 2: Chroma DB")
        chroma_db = switch_experiment_vector_db(
            "chroma",
            EMBED_MODEL_ID=os.getenv("EMBED_MODEL"),
            persist_directory="./chroma_experiment"
        )
        self.current_vector_db = chroma_db
        print(f"✅ Switched to: {chroma_db.get_name()}")
        
        if self.documents:
            chroma_db.insert_vector_store(self.documents)
        
        # 실험 3: FAISS로 변경
        print("\n🔍 Experiment 3: FAISS DB")
        faiss_db = switch_experiment_vector_db(
            "faiss",
            EMBED_MODEL_ID=os.getenv("EMBED_MODEL"),
            index_path="./faiss_experiment"
        )
        self.current_vector_db = faiss_db
        print(f"✅ Switched to: {faiss_db.get_name()}")
        
        if self.documents:
            faiss_db.insert_vector_store(self.documents)
    
    def experiment_retriever_comparison(self):
        """Retriever 비교 실험"""
        print("\n" + "="*60)
        print("🧪 RETRIEVER COMPARISON EXPERIMENT")
        print("="*60)
        
        if not self.current_vector_db:
            print("❌ No vector DB available. Please run vector DB experiment first.")
            return
        
        # 실험 1: Vector Retriever
        print("\n🔍 Experiment 1: Vector Retriever")
        vector_retriever = create_experiment_retriever(
            "vector", 
            self.current_vector_db.vector_store, 
            k=5
        )
        self.current_retriever = vector_retriever
        print(f"✅ Created: {vector_retriever.get_name()}")
        
        # 실험 2: CrossEncoder Retriever로 변경
        print("\n🔍 Experiment 2: CrossEncoder Retriever")
        cross_encoder_retriever = switch_experiment_retriever(
            "cross_encoder",
            self.current_vector_db.vector_store,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            k=5,
            top_n=3
        )
        self.current_retriever = cross_encoder_retriever
        print(f"✅ Switched to: {cross_encoder_retriever.get_name()}")
        
        # 실험 3: MultiQuery Retriever (LLM이 필요한 경우)
        print("\n🔍 Experiment 3: MultiQuery Retriever")
        try:
            # LLM 인스턴스 생성
            llm = Ollama_Models.load_models(os.getenv("LLM_MODEL", "llama2"))
            
            multi_query_retriever = switch_experiment_retriever(
                "multi_query",
                self.current_vector_db.vector_store,
                llm=llm
            )
            self.current_retriever = multi_query_retriever
            print(f"✅ Switched to: {multi_query_retriever.get_name()}")
        except Exception as e:
            print(f"⚠️ MultiQuery Retriever requires LLM: {e}")
    
    def test_retrieval(self, query: str = "박태정에 대해 설명하세요"):
        """검색 테스트"""
        print("\n" + "="*60)
        print("🧪 RETRIEVAL TEST")
        print("="*60)
        
        if not self.current_retriever:
            print("❌ No retriever available. Please run retriever experiment first.")
            return
        
        print(f"🔍 Query: {query}")
        print(f"📋 Using: {self.current_retriever.get_name()}")
        
        try:
            # Retriever 객체 가져오기
            retriever = self.current_retriever.get_retriever()
            
            # 검색 실행
            retrieved_docs = retriever.invoke(query)
            
            print(f"\n📄 Retrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"\n--- Document {i} ---")
                print(f"Content: {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata'):
                    print(f"Metadata: {doc.metadata}")
                
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
    
    def run_comprehensive_experiment(self, file_path: str):
        """종합 실험 실행"""
        print("🚀 Starting Comprehensive RAG Experiment")
        print("="*60)
        
        # 1. 문서 로드
        self.load_documents(file_path)
        
        # 2. Vector DB 실험
        self.experiment_vector_db_comparison()
        
        # 3. Retriever 실험
        self.experiment_retriever_comparison()
        
        # 4. 검색 테스트
        self.test_retrieval()
        
        # 5. 현재 설정 출력
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Current Vector DB: {self.current_vector_db.get_name() if self.current_vector_db else 'None'}")
        print(f"Current Retriever: {self.current_retriever.get_name() if self.current_retriever else 'None'}")
        print(f"Documents Loaded: {len(self.documents) if self.documents else 0}")
        
        # Factory 설정 정보
        print(f"\nVector DB Factory Config: {vector_db_factory.get_current_config()}")
        print(f"Retriever Factory Config: {retriever_factory.get_current_config()}")
        
        print("\n✅ Experiment completed!")


def main():
    """메인 실행 함수"""
    print("🎯 RAG Pipeline Lab - Factory Pattern Experiment")
    print("="*60)
    
    # 실험 인스턴스 생성
    experiment = RAGExperiment()
    
    # 사용 가능한 파일들
    available_files = [
        "박태정_Cv.pdf",
        "2501.17887v1.pdf"
    ]
    
    print("📁 Available files:")
    for i, file in enumerate(available_files, 1):
        print(f"  {i}. {file}")
    
    # 파일 선택 (실제로는 첫 번째 파일 사용)
    selected_file = available_files[0]
    print(f"\n📄 Using file: {selected_file}")
    
    # 종합 실험 실행
    experiment.run_comprehensive_experiment(selected_file)


if __name__ == "__main__":
    main()
