# RAG Pipeline Lab

RAG 시스템 실험을 위한 기능별 모듈화한 레포지토리입니다. 

## 🏗️ 시스템 구조도

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline Lab                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Document    │    │ Vector      │    │ LLM         │         │
│  │ Parser      │    │ Database    │    │ Interface   │         │
│  │             │    │             │    │             │         │
│  │ • Docling   │───▶│ • MilvusDB  │───▶│ • Ollama    │         │
│  │ • pdfPlumber│    │ • Embedding │    │ • Models    │         │
│  │ • Chunking  │    │ • Reranking │    │ • Inference │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Text        │    │ RAG         │    │ Chain       │         │
│  │ Embedding   │    │ Pipeline    │    │ Orchestrator│         │
│  │             │    │             │    │             │         │
│  │ • Models    │    │ • Retrieval │    │ • LabChain  │         │
│  │ • Encoding  │    │ • Generation│    │ • Workflow  │         │
│  │ • Similarity│    │ • Prompting │    │ • Metrics   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 프로젝트 구조

```
rag-pipeline-lab/
├── main.py                 # 메인 실행 파일
├── document_parser.py      # 문서 파싱 모듈
├── vectordb.py            # 벡터 데이터베이스 관리
├── rag.py                 # RAG 파이프라인 구현
├── llm.py                 # LLM 인터페이스
├── text_embedding.py      # 텍스트 임베딩 모듈
├── chain.py               # 체인 오케스트레이터
├── models/                # 로컬 모델 저장소
│   └── ggml-model-Q5_K_M/
├── metric/                # 평가 메트릭
├── venv/                  # 가상환경
└── README.md              # 프로젝트 문서
```

## 🔧 모듈별 기능

### 1. Document Parser (`document_parser.py`)
- **Docling**: 하이브리드 청킹을 사용한 문서 파싱
- **pdfPlumber**: PDF 텍스트 추출 및 청킹
- 추상 클래스 기반으로 확장 가능한 구조

### 2. Vector Database (`vectordb.py`)
- **MilvusDB**: Milvus 벡터 데이터베이스 관리
- 임베딩 모델 통합 (HuggingFace)
- 재순위화(Re-ranking) 기능
- 컨텍스트 압축 검색기

### 3. RAG Pipeline (`rag.py`)
- RAG 파이프라인 설정 및 관리
- 프롬프트 템플릿 관리
- 컨텍스트 처리 로직

### 4. LLM Interface (`llm.py`)
- **Ollama**: 로컬 LLM 모델 관리
- 모델 로딩 및 추론 인터페이스
- 다양한 모델 지원

### 5. Text Embedding (`text_embedding.py`)
- 텍스트 임베딩 모델 관리
- 벡터화 및 유사도 계산

### 6. Chain Orchestrator (`chain.py`)
- 전체 RAG 워크플로우 오케스트레이션
- 모듈 간 연결 및 데이터 흐름 관리

## 🚀 사용법

### 1. 환경 설정
```bash
# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 필요한 설정 추가
```

### 2. 기본 실행
```bash
python main.py
```

### 3. 모듈별 테스트
```bash
# 문서 파싱 테스트
python document_parser.py

# 벡터 DB 테스트
python vectordb.py

# LLM 테스트
python llm.py
```

## ⚙️ 환경 변수

`.env` 파일에 다음 설정이 필요합니다:

```env
EMBED_MODEL=BAAI/bge-m3
RERANKER_MODEL=Alibaba-NLP/gte-multilingual-reranker-base
DB_ADDRESS=localhost:19530
LLM_MODEL=llama2
```

## 🔬 실험 기능

### 1. 문서 파싱 실험
- 다양한 청킹 크기 및 오버랩 설정
- 여러 파서 비교 (Docling vs pdfPlumber)
- 청킹 품질 평가

### 2. 임베딩 실험
- 다양한 임베딩 모델 비교
- 임베딩 품질 및 성능 측정
- 벡터 유사도 분석

### 3. 검색 실험
- 재순위화 모델 성능 비교
- 검색 정확도 및 재현율 측정
- 컨텍스트 압축 효과 분석

### 4. 생성 실험
- 다양한 LLM 모델 비교
- 프롬프트 엔지니어링 실험
- 응답 품질 평가

## 📊 평가 메트릭

- **검색 정확도**: Precision@K, Recall@K
- **생성 품질**: BLEU, ROUGE, BERTScore
- **응답 관련성**: Semantic Similarity
- **실행 시간**: Latency 측정

## 🛠️ 개발 가이드

### 새로운 파서 추가
```python
class NewParser(Document):
    def __init__(self):
        super().__init__()
        self.model_name = "NewParser"
    
    def document_parsing(self, file_path, chunk_size, overlap):
        # 구현 로직
        pass
```

### 새로운 벡터 DB 추가
```python
class NewVectorDB:
    def __init__(self, embed_model, reranker_model, db_address):
        # 초기화 로직
        pass
    
    def insert_vector_store(self, document):
        # 삽입 로직
        pass
```
