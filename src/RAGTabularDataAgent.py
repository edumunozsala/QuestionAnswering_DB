import os

class RAGTabularDataAgent:
    """
    A RAG Agent to query Tabular Data 
    """
    def __init__(self,agent_system_role: str, collection_name:str, embedding_model, langchain_llm, vectordb) -> None:
        """
        Initialize an instance of PrepareSQLFromTabularData.

        Args:
            files_dir (str): The directory containing the CSV or XLSX files to be converted to SQL tables.
        """
        self.langchain_llm= langchain_llm
        self.embeddings_model= embedding_model
        self.vectordb= vectordb
        self.agent_system_role= agent_system_role
        self.collection_name= collection_name
        self.chatbot= []

    @staticmethod
    def respond(message: str, topk) -> Tuple:
        """
        Respond to a message based on the given chat and application functionality types.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Describes the type of the chat (interaction with SQL DB or RAG).
            app_functionality (str): Identifies the functionality for which the chatbot is being used (e.g., 'Chat').

        Returns:
            Tuple[str, List, Optional[Any]]: A tuple containing an empty string, the updated chatbot conversation list,
                                             and an optional 'None' value. The empty string and 'None' are placeholder
                                             values to match the required return type and may be updated for further functionality.
                                             Currently, the function primarily updates the chatbot conversation list.
        """
        search_params = {"metric_type": "L2",  "params": {"level": 2}}
        query_embeddings  = self.embeddings_model.embed_query(
                    message,
                )
        results = self.vectordb.milvus_client.search(self.collection_name, query_embeddings, 
                        limit=topk, search_params=search_params, anns_field="row_embedding")

        prompt = f"User's question: {message} \n\n Search results:\n {results}"

        messages = [
                    {"role": "system", "content": str(
                        APPCFG.rag_llm_system_role
                    )},
                    {"role": "user", "content": prompt}
        ]
        llm_response = APPCFG.azure_openai_client.chat.completions.create(
                    model=APPCFG.model_name,
                    messages=messages
        )
        response = llm_response.choices[0].message.content

        chatbot.append(
                (message, response))
        return response, chatbot
