from pymilvus import MilvusClient
from pymilvus import DataType
import cohere
import os

class VectorDB:

    def __init__(self, uri:str, token: str, rerank: bool = False) -> None:
        """
        Initialize the instance with the file directory and load the app config.
        
        Args:
            file_directory (str): The directory path of the file to be processed.
        """
        self.uri = uri
        self.token = token
        if rerank:
            self.reranker = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

    def load_milvus_client(self):
        self.milvus_client = MilvusClient(uri=self.uri, token=self.token)
        print(f"Connected to DB: {self.uri}")

    def prepare_vectordb(self, collection_name: str, dim: int):
        self.dim= dim
        self.collection_name= collection_name

        # Check if the collection exists
        check_collection = self.milvus_client.has_collection(collection_name)

        if check_collection:
            self.milvus_client.drop_collection(collection_name)
            print("Success to drop the existing collection %s" % collection_name)

        print("Preparing schema")
        self.schema = self.milvus_client.create_schema(auto_id= True)
        self.schema.add_field("row_id", DataType.INT64, is_primary=True, description="Row id")
        self.schema.add_field("batch", DataType.INT64, is_primary=False, description="Batch")
        self.schema.add_field("source", DataType.VARCHAR, max_length=100, description="Source datafile name")
        self.schema.add_field("row", DataType.VARCHAR, max_length= 16384, description="Row content")
        self.schema.add_field("description", DataType.VARCHAR, max_length= 256, description="Data content description")
        self.schema.add_field("row_embedding", DataType.FLOAT_VECTOR, dim=dim, description="Row embedding")
        print("Preparing index parameters")
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index("row_embedding", index_type="AUTOINDEX", metric_type="COSINE")

        print(f"Creating collection: {collection_name}")
        # create collection with the above schema and index parameters, and then load automatically
        self.milvus_client.create_collection(collection_name, dimension=dim, schema=self.schema, index_params=index_params)
        collection_property = self.milvus_client.describe_collection(collection_name)
        print("Show collection details: %s" % collection_property)

    def load_milvus_collection(self, collection_name: str):
        # 7. Load the collection
        self.milvus_client.load_collection(
            collection_name
        )

        res = self.milvus_client.get_load_state(
            collection_name
        )

        print(res)
        print(f"Loaded Vector DB: {self.uri}")

    def print_milvus_results(self, results: list):
        for hits in results:
            print("TopK results:")
            for hit in hits:
                print(hit) 

    def get_docs_results(self, collection_name: str, results: list):

        res = self.milvus_client.get(
            collection_name=collection_name,
            ids=[hit["row_id"] for res in results for hit in res],
            output_fields=["row"]
        )

        return [r["row"] for r in res]
    
    def rerank_docs(self, query: str, documents: list, top_n: int = 2) -> list:
        """
        Rerank the documents using Cohere's reranking model.
        Args:
            query (str): The user query.
            documents (list): List of documents to rerank.
        Returns:
            list: Reranked documents.
        """
        # Rerank the documents
        results = self.reranker.rerank(
            model=os.getenv("RERANK_MODEL"), query=query, documents=documents, top_n=top_n)
        # Extract the reranked documents
        reranked_docs = [result.document for result in results.results]

        #for result in results.results:
        #    print(result)
        return reranked_docs
