from langchain_community.llms import ollama
from dotenv import load_dotenv
import os

class Ollama_Models():
    def __init__(self):
        pass
    
    def list_ollama():
        print("ollama model list 를 반환합니다. | 구현해야함.")
        pass 

    def load_models(model_name):
        return ollama.Ollama(model=model_name)


if __name__=="__main__":
    load_dotenv()
    llm = ollama.Ollama(model=os.environ.get("LLM_MODEL"))
    print("모델을 성공적으로 불러왔습니다.")
    #llm.invoke(
    #    "안녕? langchain 에 대해서 설명해줘"
    #)