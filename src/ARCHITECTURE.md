# RAG Pipeline Lab - 시스템 아키텍처

## 🏗️ 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RAG Pipeline Lab                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   Document      │    │   Vector        │    │   LLM           │             │
│  │   Processing    │    │   Database      │    │   Interface     │             │
│  │                 │    │                 │    │                 │             │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │             │
│  │ │  Document   │ │    │ │   MilvusDB  │ │    │ │   Ollama    │ │             │
│  │ │   Parser    │ │    │ │             │ │    │ │   Models    │ │             │
│  │ │             │ │    │ │ • Embedding │ │    │ │             │ │             │
│  │ │ • Docling   │ │───▶│ │ • Reranking │ │───▶│ │ • llama2    │ │             │
│  │ │ • pdfPlumber│ │    │ │ • Retrieval │ │    │ │ • mistral   │ │             │
│  │ │ • Chunking  │ │    │ │ • Storage   │ │    │ │ • codellama │ │             │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│           │                       │                       │                     │
│           └───────────────────────┼───────────────────────┘                     │
│                                   │                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   Text          │    │   RAG           │    │   Chain         │             │
│  │   Embedding     │    │   Pipeline      │    │   Orchestrator  │             │
│  │                 │    │                 │    │                 │             │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │             │
│  │ │   Models    │ │    │ │  Retrieval  │ │    │ │  LabChain   │ │             │
│  │ │             │ │    │ │             │ │    │ │             │ │             │
│  │ │ • BGE-M3    │ │    │ │ • Context   │ │    │ │ • Workflow  │ │             │
│  │ │ • Sentence- │ │    │ │   Retrieval │ │    │ │ • Pipeline  │ │             │
│  │ │   Transformers│ │    │ │ • Re-ranking│ │    │ │ • Metrics   │ │             │
│  │ │ • E5        │ │    │ │ • Generation│ │    │ │ • Evaluation│ │             │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 데이터 플로우

### 1. 문서 처리 단계
```
PDF/Document → Document Parser → Chunks → Metadata
     ↓              ↓              ↓         ↓
  File Input → Text Extraction → Chunking → Storage
```

### 2. 벡터화 단계
```
Chunks → Text Embedding → Vector Database → Index
  ↓           ↓              ↓              ↓
Documents → Encoding → Milvus Storage → Search Index
```

### 3. 검색 및 재순위화 단계
```
Query → Embedding → Vector Search → Re-ranking → Top-K Results
  ↓        ↓           ↓            ↓           ↓
User Input → Encoding → Retrieval → Reranking → Context
```

### 4. 생성 단계
```
Context + Query → LLM → Response → Evaluation
      ↓              ↓       ↓         ↓
Retrieved Docs → Generation → Answer → Metrics
```

## 📊 모듈별 상세 구조

### Document Parser Module
```
Document (Abstract Base Class)
├── Docling
│   ├── HybridChunker
│   ├── DocumentConverter
│   └── BGE-M3 Tokenizer
└── pdfPlumber
    ├── PDF Text Extraction
    ├── Page-based Chunking
    └── Metadata Generation
```

### Vector Database Module
```
MilvusDB
├── Embedding Layer
│   ├── HuggingFace Embeddings
│   ├── Model Configuration
│   └── Vector Encoding
├── Storage Layer
│   ├── Milvus Connection
│   ├── Collection Management
│   └── Index Configuration
└── Retrieval Layer
    ├── Base Retriever
    ├── Cross-Encoder Reranker
    └── Contextual Compression
```

### LLM Interface Module
```
Ollama_Models
├── Model Management
│   ├── Model Listing
│   ├── Model Loading
│   └── Model Configuration
├── Inference Engine
│   ├── Text Generation
│   ├── Response Processing
│   └── Error Handling
└── Integration Layer
    ├── LangChain Integration
    ├── Prompt Management
    └── Output Formatting
```

### RAG Pipeline Module
```
Rag_pipeline
├── Configuration
│   ├── Model Settings
│   ├── Prompt Templates
│   └── Export Types
├── Processing Pipeline
│   ├── Document Processing
│   ├── Retrieval Logic
│   └── Generation Logic
└── Output Management
    ├── Response Formatting
    ├── Error Handling
    └── Logging
```

## 🔧 기술 스택

### Core Technologies
- **Python 3.8+**: 메인 프로그래밍 언어
- **LangChain**: LLM 프레임워크
- **Milvus**: 벡터 데이터베이스
- **Ollama**: 로컬 LLM 실행

### ML/AI Models
- **Embedding Models**: BAAI/bge-m3, Sentence-Transformers
- **Reranking Models**: Alibaba-NLP/gte-multilingual-reranker-base
- **LLM Models**: llama2, mistral, codellama (via Ollama)

### Libraries & Frameworks
- **Document Processing**: docling, pdfplumber
- **Vector Operations**: langchain-huggingface, langchain-milvus
- **Evaluation**: langfuse (for monitoring)
- **Environment**: python-dotenv

## 🎯 실험 설계

### A/B Testing Framework
```
Experiment Configuration
├── Document Parser Comparison
│   ├── Docling vs pdfPlumber
│   ├── Chunk Size Variations
│   └── Overlap Size Testing
├── Embedding Model Comparison
│   ├── BGE-M3 vs Sentence-Transformers
│   ├── Model Performance Metrics
│   └── Vector Quality Assessment
├── Retrieval Strategy Testing
│   ├── Base Retrieval vs Re-ranking
│   ├── Top-K Variations
│   └── Context Compression Effects
└── LLM Model Comparison
    ├── Response Quality
    ├── Generation Speed
    └── Resource Usage
```

### Evaluation Metrics
```
Performance Metrics
├── Retrieval Metrics
│   ├── Precision@K
│   ├── Recall@K
│   ├── NDCG@K
│   └── MRR (Mean Reciprocal Rank)
├── Generation Metrics
│   ├── BLEU Score
│   ├── ROUGE Score
│   ├── BERTScore
│   └── Semantic Similarity
├── System Metrics
│   ├── Latency
│   ├── Throughput
│   ├── Memory Usage
│   └── CPU/GPU Utilization
└── User Experience Metrics
    ├── Response Relevance
    ├── Answer Completeness
    ├── User Satisfaction
    └── Task Completion Rate
```

## 🔄 확장성 고려사항

### Horizontal Scaling
- **Document Processing**: 병렬 처리 지원
- **Vector Database**: Milvus 클러스터 구성
- **LLM Inference**: 다중 인스턴스 로드 밸런싱

### Vertical Scaling
- **Model Optimization**: 양자화 및 압축
- **Memory Management**: 효율적인 메모리 사용
- **Caching Strategy**: 중간 결과 캐싱

### Modularity
- **Plugin Architecture**: 새로운 모듈 쉽게 추가
- **Configuration Management**: 동적 설정 변경
- **API Interface**: RESTful API 지원

## 🛡️ 보안 및 안정성

### Data Security
- **Input Validation**: 문서 및 쿼리 검증
- **Access Control**: 사용자 권한 관리
- **Data Encryption**: 민감 정보 암호화

### System Reliability
- **Error Handling**: 포괄적인 예외 처리
- **Logging**: 상세한 로그 기록
- **Monitoring**: 시스템 상태 모니터링
- **Backup**: 데이터 백업 및 복구

## 📈 성능 최적화

### Caching Strategy
```
Multi-Level Caching
├── Document Cache
│   ├── Parsed Documents
│   ├── Chunked Content
│   └── Metadata
├── Embedding Cache
│   ├── Vector Embeddings
│   ├── Similarity Scores
│   └── Index Structures
├── LLM Cache
│   ├── Generated Responses
│   ├── Prompt Templates
│   └── Model States
└── Result Cache
    ├── Final Answers
    ├── Intermediate Results
    └── Evaluation Metrics
```

### Optimization Techniques
- **Batch Processing**: 대량 데이터 처리
- **Async Operations**: 비동기 처리
- **Memory Pooling**: 메모리 재사용
- **Model Quantization**: 모델 크기 최적화
