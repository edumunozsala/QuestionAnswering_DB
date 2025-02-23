from src.VectorDB import VectorDB
from dotenv import load_dotenv

if __name__ == "__main__":
    print("Environment variables are loaded:", load_dotenv())
    vectordb= VectorDB(os.getenv("PERSIST_DIRECTORY"))   
    vectordb= load_chroma_client() 
    PrepareVectorDBFromTabularData(os.getenv("DATA_DIR"), "turismo_provincia", os.getenv("EMBEDDINGS_MODEL"), vectordb)

    df= load_data(limit=100)
    print("DF")
    print(df.head())
    print(len(df))
