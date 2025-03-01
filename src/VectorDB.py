from pymilvus import MilvusClient
from pymilvus import DataType

class VectorDB:

    def __init__(self, uri:str, token: str) -> None:
        """
        Initialize the instance with the file directory and load the app config.
        
        Args:
            file_directory (str): The directory path of the file to be processed.
        """
        self.uri = uri
        self.token = token

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
        self.schema = self.milvus_client.create_schema()
        self.schema.add_field("row_id", DataType.INT64, is_primary=True, description="Row id")
        self.schema.add_field("batch", DataType.INT64, is_primary=False, description="Batch")
        self.schema.add_field("source", DataType.VARCHAR, max_length=100, description="Source datafile name")
        self.schema.add_field("row", DataType.VARCHAR, max_length= 256, description="Row content")
        self.schema.add_field("row_embedding", DataType.FLOAT_VECTOR, dim=dim, description="Row embedding")
        print("Preparing index parameters")
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index("row_embedding", metric_type="L2")

        print(f"Creating collection: {collection_name}")
        # create collection with the above schema and index parameters, and then load automatically
        self.milvus_client.create_collection(collection_name, dimension=dim, schema=self.schema, index_params=index_params)
        collection_property = self.milvus_client.describe_collection(collection_name)
        print("Show collection details: %s" % collection_property)



