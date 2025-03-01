#from src.VectorDB import VectorDB
from src.VectorsFromTabularData import PrepareVectorDBFromTabularData
from src.AIModels import AIModels
from src.VectorDB import VectorDB

import os
from dotenv import load_dotenv

if __name__ == "__main__":
    print("Environment variables are loaded:", load_dotenv())
    #vectordb= VectorDB(os.getenv("PERSIST_DIRECTORY"))   
    #vectordb= load_chroma_client() 
    models= AIModels(os.getenv("OPENAI_MODEL"),os.getenv("EMBEDDINGS_MODEL"),os.getenv("TEMPERATURE"), os.getenv("MAX_TOKENS"))
    models.load_openai_models()

    vectordb= VectorDB(os.getenv("MILVUS_URI"), os.getenv("MILVUS_TOKEN"))
    vectordb.load_milvus_client()
    vectordb.prepare_vectordb(os.getenv("COLLECTION_NAME"), int(os.getenv("EMBEDDINGS_DIM")))

    prepdata= PrepareVectorDBFromTabularData(os.getenv("DATAFILE_TO_LOAD"), os.getenv("COLLECTION_NAME"), 
                                            os.getenv("CSV_CODEC"), os.getenv("CSV_SEP"), models.embeddings_model, vectordb)

    df= prepdata.load_data(100)

    print("DF")
    #print(df.head())
    #print(df.info())
    #print(len(df))
