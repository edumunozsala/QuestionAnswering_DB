import os
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

class TextToSQLAgent:
    """
    A TextToSQL Agent 
    """
    def __init__(self,sqldb_dir: str, agent_system_role: str, langchain_llm) -> None:
        """
        Initialize an instance of PrepareSQLFromTabularData.

        Args:
            files_dir (str): The directory containing the CSV or XLSX files to be converted to SQL tables.
        """

        # Set the DB
        if os.path.exists(sqldb_dir):
            self.db = SQLDatabase.from_uri(
                        f"sqlite:///{sqldb_dir}")
        self.langchain_llm= langchain_llm
        self.agent_system_role= agent_system_role
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
        execute_query = QuerySQLDataBaseTool(db=self.db)

        write_query = create_sql_query_chain(
                        self.langchain_llm, self.db)

        answer_prompt = PromptTemplate.from_template(
                        self.agent_system_role)
        answer = answer_prompt | self.langchain_llm | StrOutputParser()
        chain = (
                    RunnablePassthrough.assign(query=write_query).assign(
                            result=itemgetter("query") | execute_query
                    )
                    | answer
                )
        response = chain.invoke({"question": message})

        # Get the `response` variable from any of the selected scenarios and pass it to the user.
        self.chatbot.append(
                (message, response))
        return response, self.chatbot
