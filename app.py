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

def create_vectordb(rerank: bool= False):
    # Create the vector database
    vectordb= VectorDB(os.getenv("MILVUS_URI"), os.getenv("MILVUS_TOKEN"), rerank= rerank)
    vectordb.load_milvus_client()
    # Create the collection
    vectordb.prepare_vectordb(os.getenv("COLLECTION_NAME"), int(os.getenv("EMBEDDINGS_DIM")))

    return vectordb

def load_collection_vectordb(rerank: bool= False):
    # Create the vector database
    vectordb= VectorDB(os.getenv("MILVUS_URI"), os.getenv("MILVUS_TOKEN"), rerank= rerank)
    # Load the collection
    vectordb.load_milvus_client()
    vectordb.load_milvus_collection(os.getenv("COLLECTION_NAME"))

    return vectordb

def process_docs_to_vectordb(models: AIModels, datafiles: list, vectordb: VectorDB, limit: int= 100, batch_size: int= 25):
    prepdata= PrepareVectorDBFromTabularData(os.getenv("DATA_DIR"), os.getenv("COLLECTION_NAME"), 
                                             os.getenv("CSV_CODEC"), os.getenv("CSV_SEP"), models.embeddings_model, vectordb)

    prepdata.load_data(datafiles, limit, batch_size)
    #prepdata.test_load()


    return vectordb

def prepare_rag_db(models: AIModels, file_descriptions: list, limit: int= 100, batch_size: int= 25, 
                   create_collection: bool=False, rerank: bool= False):
    # If we want to create a new collection
    if create_collection:
        # Create the vector database
        vectordb= create_vectordb(rerank= rerank)
    else:
        # Load the collection
        vectordb= load_collection_vectordb(rerank= rerank)

    # Process the documents to the vector database
    process_docs_to_vectordb(models, file_descriptions, vectordb, limit, batch_size)

    return vectordb
    
def question_answer(question: str, vectordb: VectorDB, models: AIModels):

    search_params = {"metric_type": "COSINE"}
    rag_agent= RAGTabularDataAgent(os.getenv("RAG_LLM_SYSTEM_ROLE"), os.getenv("COLLECTION_NAME"), 
            models.embeddings_model, models.langchain_llm, vectordb)
    response,chat= rag_agent.respond(question, search_params, topk= 5)

    print(response)
    return response, chat

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
    
    # Load the configuration
    config= LoadConfig(os.getenv("DATAFILES_CONFIG"))
    file_descriptions= config.get_file_descriptions()
    print(file_descriptions)
    # Load the models   
    models= load_models()

    # Create the vector database
    vectordb= prepare_rag_db(models, file_descriptions, 2000, 25, True, False)
    
    # Set the question
    question= "¿Cuantos viviendas turísticas existían en la provincia de Albacete en Agosto del año 2020?"
    # Get the response
    response, chat= question_answer(question, vectordb, models)
    print(response)

    # Set the question
    #question= "¿Cual fue el gasto medio de los turistas de tipo Internacional en A coruña en el año 2019?"
    question= "¿Cual fue el gasto de los turistas de tipo Nacional en La Rioja en Febrero del año 2019?"
    # Get the response
    response, chat= question_answer(question, vectordb, models)
    print(response)

    # Set the question
    question= "¿Cuantos turistas visitaron la ciudad de A coruña en Septiembre del año 2019?"
    # Get the response
    response, chat= question_answer(question, vectordb, models)
    print(response)
