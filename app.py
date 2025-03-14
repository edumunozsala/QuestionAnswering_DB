from src.VectorsFromTabularData import PrepareVectorDBFromTabularData
from src.AIModels import AIModels
from src.VectorDB import VectorDB
from src.RAGTabularDataAgent import RAGTabularDataAgent
from src.SQLDBFromTabularData import PrepareSQLFromTabularData
from src.TextToSQLAgent import TextToSQLAgent
from src.SQLDB import SQLDB
from src.SQLAgent import SQLAgent
from src.LoadConfig import LoadConfig

import os
from dotenv import load_dotenv

def process_docs_to_vectordb():
    vectordb= VectorDB(os.getenv("MILVUS_URI"), os.getenv("MILVUS_TOKEN"))
    vectordb.load_milvus_client()
    vectordb.prepare_vectordb(os.getenv("COLLECTION_NAME"), int(os.getenv("EMBEDDINGS_DIM")))

    prepdata= PrepareVectorDBFromTabularData(os.getenv("DATAFILE_TO_LOAD"), os.getenv("COLLECTION_NAME"), 
                                            os.getenv("CSV_CODEC"), os.getenv("CSV_SEP"), models.embeddings_model, vectordb)

    prepdata.load_data(100)

def question_answer():
    vectordb= VectorDB(os.getenv("MILVUS_URI"), os.getenv("MILVUS_TOKEN"))
    vectordb.load_milvus_client()
    vectordb.load_milvus_collection(os.getenv("COLLECTION_NAME"))

    search_params = {"metric_type": "L2"}
    rag_agent= RAGTabularDataAgent(os.getenv("RAG_LLM_SYSTEM_ROLE"), os.getenv("COLLECTION_NAME"), 
            models.embeddings_model, models.langchain_llm, vectordb)
    response,chat= rag_agent.respond("¿Cuantos turistas visitaron la ciudad de A coruña en el año 2019?", search_params, 3)

    print(response)
    return response

def load_csv_to_sqldb():
    prepdata= PrepareSQLFromTabularData(os.getenv("DATA_DIR"), os.getenv("DB_DIR"), 
                                            os.getenv("CSV_CODEC"), os.getenv("CSV_SEP"))

    prepdata.run_pipeline(100)

def load_models():
    models= AIModels(os.getenv("OPENAI_MODEL"),os.getenv("EMBEDDINGS_MODEL"),os.getenv("TEMPERATURE"), os.getenv("MAX_TOKENS"))
    models.load_openai_models()

    return models

def run_text_to_sql_agent():
    agent_sql= TextToSQLAgent(os.getenv("DB_DIR"), os.getenv("AGENT_LLM_SYSTEM_ROLE"), models.langchain_llm)
    response,chat= agent_sql.respond("¿Cuantos turistas visitaron la ciudad de A coruña en el año 2019?")

def run_sql_agent():
    # Create a SQL Database engine
    sql_database= SQLDB(os.getenv("DB_DIR"))
    # Create the SQL Agent
    sql_agent= SQLAgent(sql_database.db, os.getenv("AGENT_LLM_SYSTEM_ROLE"), models.langchain_llm)
    # Query the database
    response, chat= sql_agent.respond("¿Cuantos turistas visitaron la ciudad de A coruña en el año 2019?")
    print(response)

if __name__ == "__main__":
    print("Environent variables are loaded:", load_dotenv())
    #models= AIModels(os.getenv("OPENAI_MODEL"),os.getenv("EMBEDDINGS_MODEL"),os.getenv("TEMPERATURE"), os.getenv("MAX_TOKENS"))
    #models.load_openai_models()

    # Load the configuration
    config= LoadConfig(os.getenv("DATAFILES_CONFIG"))
    file_descriptions= config.get_file_descriptions()
    print(file_descriptions)



