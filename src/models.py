from openai import AzureOpenAI
from src.settings import Settings, set_logging


class AzureOpenAIClient:
    settings = Settings()
        
    @classmethod
    def initialize_client(cls):
        client = AzureOpenAI(
            api_version=cls.settings.openai.api_version,
            azure_endpoint=cls.settings.openai.endpoint,
            api_key=cls.settings.openai.credential,
        )
        return client
    
if __name__=="__main__":
    __all__=["AzureOpenAIClient"]