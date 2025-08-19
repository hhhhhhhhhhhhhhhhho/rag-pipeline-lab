from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from abc import ABC, abstractmethod

class Document(ABC):
    def __init__(self):
        self.model_name = ""
        self.chunking_size = 10
        self.overlap=10
        self.tokenizer = ""

    @abstractmethod
    def document_parsing():
        '''
        추상메소드로 구현해서 자식 클래스들이 구현해줘야 한다 이따가 ㄱㄱ
        '''
        pass

class Docling(Document):
    def __init__(self):
        super().__init__()
        self.model_name="Docling"
        self.tokenizer="BAAI/bge-m3"

    
    def document_parsing(self,file_path,chunk_size,overlap):
        ""
        "문서를 파싱하고, 그것을 chunks 로 반환 받는다."
        ""
        chunked_res = DocumentConverter().convert(file_path)
        docs = chunked_res.document

        chunker = HybridChunker(
        tokenizer ="BAAI/bge-m3", # 토크나이저 설정
        chunk_size = 1000 , #chunk size setting
        overlap = 100, # chunking 간 overlap size 설정 ( 토큰 단위인지 확인 필요)
        )
        chunks = chunker.chunk(docs)
        #chunks = list(chunk_iter)
        #print(f"type of chunk list if {type(chunks)}")
        #print(f"Original Chunk is {len(chunks)}, \n\n Convert list is {list(chunks)}\n")

        return chunks

class pdfPlumber(Document):
    def __init__(self):
        super().__init__()
        self.model_name="pdfplumber"
        self.tokenizer="BAAI/bge-m3"

    
    def document_parsing(self,file_path,chunk_size,overlap):
        import pdfplumber
        from langchain.schema import Document

        documents = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:  # 빈 페이지는 무시
                    metadata = {"source": file_path, "page": i + 1}
                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)

        return documents


if __name__ == "__main__":
    doc = Docling()
    doc.document_parsing("2501.17887v1.pdf",1000,100)