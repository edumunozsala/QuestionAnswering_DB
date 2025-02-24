
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI

class AIModels:

    def __init__(self, model_name:str, embedding_model_name:str, temperature: float, max_tokens: int) -> None:
        """
        Initialize the instance with the file directory and load the app config.
        
        Args:
            file_directory (str): The directory path of the file to be processed.
        """
        self.model_name = model_name
        self.embedding_model_name= embedding_model_name
        self.temperature = temperature
        self.max_tokens= max_tokens

    def load_azure_openai_models(self):
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

    def load_openai_models(self):
        # This will be used for the GPT and embedding models
        self.openai_client = OpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )
        self.langchain_llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )
        self.embeddings_model = OpenAIEmbeddings(
                                    model=self.embedding_model_name
                                )
 