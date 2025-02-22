
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI

Class AIModels:

    def __init__(self, model_name:str, embedding_model_name:str, temperature: float) -> None:
        """
        Initialize the instance with the file directory and load the app config.
        
        Args:
            file_directory (str): The directory path of the file to be processed.
        """
        self.model_name = model_name
        self.embedding_model_name= embedding_model_name
        self.temperature = file_directory

    def load_openai_models(self):
        azure_openai_api_key = os.environ["OPENAI_API_KEY"]
        azure_openai_endpoint = os.environ["OPENAI_API_BASE"]
        # This will be used for the GPT and embedding models
        self.azure_openai_client = AzureOpenAI(
            api_key=azure_openai_api_key,
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=azure_openai_endpoint
        )
        self.langchain_llm = AzureChatOpenAI(
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_deployment=self.model_name,
            model_name=self.model_name,
            temperature=self.temperature)
