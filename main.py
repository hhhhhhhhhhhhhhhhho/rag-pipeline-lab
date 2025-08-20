from src/vectordb import MilvusDB
from src/rag import Rag_pipeline
from dotenv import load_dotenv
import os
from document_parser import Docling,pdfPlumber
load_dotenv()

if __name__ == "__main__":
    parser = pdfPlumber()
    ###### PDF parsing Block ##############
    parsed_document= parser.document_parsing(
        "박태정_Cv.pdf",
        1000,
        10)
    ######################################

    ##### Embedding and Vector Store Block ######
    vectorDB =MilvusDB(
        os.getenv("EMBED_MODEL"),
        os.getenv("RERANKER_MODEL"),
        os.getenv("DB_ADDRESS"))
    vectorDB.insert_vector_store(parsed_document)
    #############################################


    ###### Reranking and Retriever Block ######
    rag = Rag_pipeline()
    ###########################################


    ##### Context + Prompt Answering Block #####
    ############################################

    
