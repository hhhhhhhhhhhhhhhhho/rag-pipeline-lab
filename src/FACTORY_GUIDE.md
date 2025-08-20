# Factory 패턴 실험 가이드

## 🎯 개요

RAG Pipeline Lab에서 Factory 패턴을 사용하여 다양한 Vector DB와 Retriever를 쉽게 실험할 수 있습니다.

## 🏗️ Factory 패턴 구조

### 1. Vector DB Factory
```python
from vectordb import create_experiment_vector_db, switch_experiment_vector_db

# Vector DB 생성
milvus_db = create_experiment_vector_db("milvus", **config)

# 실험 중 Vector DB 변경
chroma_db = switch_experiment_vector_db("chroma", **config)
```

### 2. Retriever Factory
```python
from retriever import create_experiment_retriever, switch_experiment_retriever

# Retriever 생성
vector_retriever = create_experiment_retriever("vector", vector_store, k=5)

# 실험 중 Retriever 변경
cross_encoder_retriever = switch_experiment_retriever("cross_encoder", vector_store, **config)
```

## 🧪 실험 시나리오

### 시나리오 1: Vector DB 성능 비교
```python
# 1. Milvus로 시작
milvus_db = create_experiment_vector_db("milvus", **milvus_config)
milvus_db.insert_vector_store(documents)

# 2. Chroma로 변경하여 비교
chroma_db = switch_experiment_vector_db("chroma", **chroma_config)
chroma_db.insert_vector_store(documents)

# 3. FAISS로 변경하여 비교
faiss_db = switch_experiment_vector_db("faiss", **faiss_config)
faiss_db.insert_vector_store(documents)
```

### 시나리오 2: Retriever 성능 비교
```python
# 1. Vector Retriever로 시작
vector_retriever = create_experiment_retriever("vector", vector_store, k=5)
results1 = vector_retriever.get_retriever().invoke(query)

# 2. CrossEncoder Retriever로 변경
cross_encoder_retriever = switch_experiment_retriever(
    "cross_encoder", 
    vector_store, 
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    k=5,
    top_n=3
)
results2 = cross_encoder_retriever.get_retriever().invoke(query)

# 3. MultiQuery Retriever로 변경
multi_query_retriever = switch_experiment_retriever(
    "multi_query", 
    vector_store, 
    llm=llm_instance
)
results3 = multi_query_retriever.get_retriever().invoke(query)
```

### 시나리오 3: 종합 실험
```python
from experiment_example import RAGExperiment

# 종합 실험 실행
experiment = RAGExperiment()
experiment.run_comprehensive_experiment("your_document.pdf")
```

## 📊 사용 가능한 옵션

### Vector DB 타입
- **milvus**: 분산 벡터 데이터베이스
- **chroma**: 로컬 임베딩 데이터베이스
- **faiss**: Facebook AI Similarity Search
- **pinecone**: 클라우드 벡터 데이터베이스 (준비 중)

### Retriever 타입
- **vector**: 기본 벡터 유사도 검색
- **cross_encoder**: 재순위화를 통한 정확도 향상
- **multi_query**: 다중 쿼리 생성을 통한 검색 범위 확장
- **parent_doc**: 부모-자식 문서 관계 기반 검색

## 🔧 설정 예제

### 환경 변수 설정 (.env)
```env
# 임베딩 모델
EMBED_MODEL=BAAI/bge-m3

# 재순위화 모델
RERANKER_MODEL=Alibaba-NLP/gte-multilingual-reranker-base

# Milvus 설정
DB_ADDRESS=localhost:19530

# LLM 모델
LLM_MODEL=llama2
```

### Vector DB 설정 예제
```python
# Milvus 설정
milvus_config = {
    "EMBED_MODEL_ID": "BAAI/bge-m3",
    "RERANKER_MODEL": "Alibaba-NLP/gte-multilingual-reranker-base",
    "DB_ADDRESS": "localhost:19530"
}

# Chroma 설정
chroma_config = {
    "EMBED_MODEL_ID": "BAAI/bge-m3",
    "persist_directory": "./chroma_db"
}

# FAISS 설정
faiss_config = {
    "EMBED_MODEL_ID": "BAAI/bge-m3",
    "index_path": "./faiss_index"
}
```

### Retriever 설정 예제
```python
# Vector Retriever 설정
vector_config = {
    "k": 5
}

# CrossEncoder Retriever 설정
cross_encoder_config = {
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "k": 5,
    "top_n": 3
}

# MultiQuery Retriever 설정
multi_query_config = {
    "llm": llm_instance
}
```

## 📈 실험 결과 추적

### 현재 설정 확인
```python
# Vector DB Factory 설정
print(vector_db_factory.get_current_config())

# Retriever Factory 설정
print(retriever_factory.get_current_config())

# 사용 가능한 옵션 확인
print(vector_db_factory.list_available_vector_dbs())
print(retriever_factory.list_available_retrievers())
```

### 실험 로그 예제
```
🧪 VECTOR DB COMPARISON EXPERIMENT
============================================================

🔍 Experiment 1: Milvus DB
✅ Created: MilvusDB
📊 Connection Info: {'type': 'milvus', 'address': 'localhost:19530', ...}
✅ 15 documents inserted into Milvus at localhost:19530

🔄 Switching vector DB from MilvusDB to chroma
🔍 Experiment 2: Chroma DB
✅ Switched to: ChromaDB
✅ 15 documents inserted into Chroma at ./chroma_experiment

🔄 Switching vector DB from ChromaDB to faiss
🔍 Experiment 3: FAISS DB
✅ Switched to: FAISSDB
✅ 15 documents inserted into FAISS at ./faiss_experiment
```

## 🚀 고급 사용법

### 커스텀 Retriever 추가
```python
from retriever import RetrieverBase, RetrieverType, retriever_factory

class CustomRetriever(RetrieverBase):
    def get_retriever(self, **kwargs):
        # 커스텀 로직 구현
        return custom_retriever_instance
    
    def get_name(self) -> str:
        return "CustomRetriever"

# Factory에 등록
retriever_factory.register_retriever(RetrieverType.CUSTOM, CustomRetriever)
```

### 커스텀 Vector DB 추가
```python
from vectordb import VectorDB, VectorDBType, vector_db_factory

class CustomVectorDB(VectorDB):
    def insert_vector_store(self, document):
        # 커스텀 삽입 로직
        pass
    
    def get_name(self) -> str:
        return "CustomVectorDB"
    
    def get_connection_info(self) -> Dict[str, Any]:
        return {"type": "custom", "config": self.config}

# Factory에 등록
vector_db_factory.register_vector_db(VectorDBType.CUSTOM, CustomVectorDB)
```

## ⚠️ 주의사항

1. **의존성 관리**: 각 Retriever나 Vector DB가 필요로 하는 의존성을 올바르게 설정
2. **메모리 관리**: 실험 중 메모리 사용량 모니터링
3. **에러 처리**: 각 단계에서 발생할 수 있는 예외 상황 처리
4. **설정 백업**: 실험 전 현재 설정 백업

## 📝 실험 체크리스트

- [ ] 환경 변수 설정 완료
- [ ] 필요한 모델 다운로드 완료
- [ ] Vector DB 서비스 실행 (Milvus 등)
- [ ] 테스트 문서 준비
- [ ] 실험 시나리오 계획
- [ ] 결과 저장 경로 설정
- [ ] 성능 메트릭 정의

## 🔍 문제 해결

### 일반적인 문제들
1. **Milvus 연결 실패**: Milvus 서비스가 실행 중인지 확인
2. **모델 로딩 실패**: 인터넷 연결 및 모델 경로 확인
3. **메모리 부족**: 배치 크기 조정 또는 모델 양자화 고려
4. **의존성 충돌**: 가상환경 사용 권장

### 디버깅 팁
```python
# 상세 로그 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 현재 상태 확인
print(f"Current Vector DB: {vector_db_factory.get_current_vector_db()}")
print(f"Current Retriever: {retriever_factory.get_current_retriever()}")
```
