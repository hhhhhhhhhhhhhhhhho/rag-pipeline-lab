
from base import BaseChain


class LabChain(BaseChain):
    def __init__(
            self,
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
        
    def setup(self):
