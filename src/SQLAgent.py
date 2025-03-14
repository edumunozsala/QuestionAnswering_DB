from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

class SQLAgent:
    """
    Construct a SQL agent from an LLM and toolkit or database.
    """
    def __init__(self,sql_db: SQLDatabase, agent_system_role: str, langchain_llm) -> None:
        """
        Initialize an instance of PrepareSQLFromTabularData.

        Args:
            files_dir (str): The directory containing the CSV or XLSX files to be converted to SQL tables.
        """

        # Set the DB
        self.db= sql_db                
        # Set the LLM
        self.langchain_llm= langchain_llm
        # Set the prompt for the system role
        self.agent_system_role= agent_system_role
        # Start the chat history
        self.chatbot= []

    def respond(self, message: str) -> tuple:
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
        # Create the SQL Agent
        agent_executor = create_sql_agent(
                    self.langchain_llm, db=self.db, agent_type="openai-tools", verbose=True)
        # Invoke the agent to get the response
        response = agent_executor.invoke({"input": message})
        # Extract the response
        response = response["output"]        

        # Append the response to the chat history
        self.chatbot.append(
                (message, response))

        return response, self.chatbot
