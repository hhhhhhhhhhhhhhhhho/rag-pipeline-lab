from base import BaseChain
class LabChain(BaseChain):
    def __init__(
            self,
            document_parser,
            text_embedding_model,
            vectorDB,
            llm_model,
            system_prompt : Optional[str] = None,
            **kwargs,
    ):
        super().__init__(document_parser,
                         text_embedding_model,
                         vectorDB,
                         llm_model
                         )
        
        self.system_prompt = None
        
    def setup():
        '''
        rag pipeline 에 필요한 라이브러리들의 세팅값들을 설정합니다.
        '''
        pass