from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter


def document_parsing(file_path,chunk_size,overlap):
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

if __name__ == "__main__":
    document_parsing("2501.17887v1.pdf",1000,100)