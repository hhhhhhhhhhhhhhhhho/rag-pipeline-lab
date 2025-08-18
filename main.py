from document_parser import document_parsing
from vectordb import MilvusDB
from rag import Rag_pipeline
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    
    ###### PDF parsing Block ##############
    parsed_document= document_parsing(
        "박태정_Cv.pdf",
        1000,
        10)
    ######################################

    ##### Embedding and Vector Store Block ######
    vectorDB =MilvusDB(
        os.getenv("EMBED_MODEL"),
        os.getenv("DB_ADDRESS"))
    vectorDB.insert_vector_store(parsed_document)
    #############################################


    ###### Reranking and Retriever Block ######
    #내일 학습 해야하는 것 
    rag = Rag_pipeline()
    ###########################################


    ##### Context + Prompt Answering Block #####
    ############################################

    
