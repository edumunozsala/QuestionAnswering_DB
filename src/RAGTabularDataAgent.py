
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

    def respond(self, message: str, search_params: dict, topk) -> tuple:
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

        query_embeddings  = self.embeddings_model.embed_query(
                    message,
                )

        results = self.vectordb.milvus_client.search(collection_name= self.collection_name, data=[query_embeddings], 
                        limit=topk, search_params=search_params, anns_field="row_embedding")

        docs_context= self.vectordb.get_docs_results(self.collection_name, results)

        prompt = f"User's question: {message} \n\n Search results:\n {" ".join(docs_context)}"
        
        print(prompt)

        messages = [
                    {"role": "system", "content": str(
                        self.agent_system_role
                    )},
                    {"role": "user", "content": prompt}
        ]
        llm_response= self.langchain_llm.invoke(messages)

        response = llm_response.content

        self.chatbot.append(
                (message, response))

        return response, self.chatbot

