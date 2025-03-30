import os
import pandas as pd
import time
import json

class PrepareVectorDBFromTabularData:
    def __init__(self, file_directory:str, collection_name: str, csv_codec: str, 
                 csv_sep: str, embedding_model, vectordb) -> None:
        """
        Initialize the instance with the file directory and load the app config.
        
        Args:
            file_directory (str): The directory path of the file to be processed.
        """
        self.file_directory= file_directory
        self.embeddings_model= embedding_model
        self.vectordb= vectordb
        self.collection_name= collection_name
        self.csv_codec= csv_codec
        self.csv_sep= csv_sep
        self.docs = None
        self.metadatas = None
        self.ids = None
        self.embeddings = None

    def _load_dataframe(self, file_directory: str, limit: int, sort_by: list = None):
        """
        Load a DataFrame from the specified CSV or Excel file.
        
        Args:
            file_directory (str): The directory path of the file to be loaded.
            
        Returns:
            DataFrame, str: The loaded DataFrame and the file's base name without the extension.
            
        Raises:
            ValueError: If the file extension is neither CSV nor Excel.
        """
        file_names_with_extensions = os.path.basename(file_directory)
        print(file_names_with_extensions)
        file_name, file_extension = os.path.splitext(
                file_names_with_extensions)
        print(file_name)
        print(file_extension)
        # CSV datafile        
        if file_extension == ".csv":
            df = pd.read_csv(file_directory, sep= self.csv_sep, nrows=limit, encoding= self.csv_codec)
        # Excel datafile                    
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_directory, sep= self.csv_sep, nrows=limit, encoding= self.csv_codec)
        else:
            raise ValueError("The selected file type is not supported")

        if sort_by:
            df.sort_values(by=sort_by, inplace=True)

        return df, file_name

    
    def dataframe_to_json_batches(self, df, batch_size=50, limit=0):
        """
        Converts a pandas DataFrame to JSON format in batches of a given size.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.
        batch_size (int): The number of rows per batch.
        
        Returns:
        List of JSON strings, each containing a batch of rows.
        """
        json_batches = []
        if limit>0:
            num_rows=limit
        else:
            num_rows= len(df)

        for i in range(0, num_rows, batch_size):
            batch = df.iloc[i:i+batch_size]
            json_batches+=[eval(batch.to_json(orient='records'))]
        
        return json_batches

    def _generate_embeddings_from_json(self, json_data:list, file_name:str, file_description: str):
        """
        Generate embeddings, one per json row, and prepare documents for data ingestion. 
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.
            file_name (str): The base name of the file for use in metadata.
            
        Returns:
            list, list, list, list: Lists containing documents, metadatas, ids, and embeddings respectively.
        """
        self.docs = []
        self.metadatas = []
        self.ids = []
        self.embeddings = []

        for i, batch in enumerate(json_data):
            #output_str = ""
            # Treat each row as a separate chunk
            for j,row in enumerate(batch):
                #output_str += f"{col}: {row[col]},\n"
                # Add the file description as a new json value in the row
                row["description"]= file_description

                if j==0:
                    print(row)

                response = self.embeddings_model.embed_query(
                    str(row),
                )

                self.embeddings.append(response)
                self.docs.append(str(row))
                self.metadatas.append({"source": file_name, "description": file_description, "batch": i})
                self.ids.append(i*100+j)
            
        return self.docs, self.metadatas, self.ids, self.embeddings

    def _generate_batches_embeddings_from_json(self, json_data:list, file_name:str, file_description: str):
        """
        Generate embeddings, one per batch of json rows, and prepare documents in batches for data ingestion.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.
            file_name (str): The base name of the file for use in metadata.
            
        Returns:
            list, list, list, list: Lists containing documents, metadatas, ids, and embeddings respectively.
        """
        self.docs = []
        self.metadatas = []
        self.ids = []
        self.embeddings = []

        for i, batch in enumerate(json_data):
            #output_str = ""
            # Create a single json string from a list of json objects
            output_str = '"Content Description": "'+file_description+'",'+ ','.join([str(json.dumps(row, ensure_ascii=False).encode('latin-1')) for row in batch])
            #output_str = ','.join([str(json.dumps(row, ensure_ascii=False).encode('latin-1')) for row in batch])
            #print(type(output_str))

            response = self.embeddings_model.embed_query(
                    str(output_str),
            )
            #response= "Test"
            self.embeddings.append(response)
            self.docs.append(str(output_str))
            self.metadatas.append({"source": file_name, "description": file_description, "batch": i})
            self.ids.append(i)
            
        return self.docs, self.metadatas, self.ids, self.embeddings

    def _generate_embeddings(self, df:pd.DataFrame, file_name:str):
        """
        Generate embeddings from a pandas dataframe and prepare documents for data injection.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.
            file_name (str): The base name of the file for use in metadata.
            
        Returns:
            list, list, list, list: Lists containing documents, metadatas, ids, and embeddings respectively.
        """
        self.docs = []
        self.metadatas = []
        self.ids = []
        self.embeddings = []
        for index, row in df.iterrows():
            output_str = ""
            # Treat each row as a separate chunk
            for col in df.columns:
                output_str += f"{col}: {row[col]},\n"
            response = self.embeddings_model.embed_query(
                output_str,
            )

            self.embeddings.append(response)
            self.docs.append(output_str)
            self.metadatas.append({"source": file_name})
            self.ids.append(f"id{index}")
            
        return self.docs, self.metadatas, self.ids, self.embeddings

    def _load_data_into_vectordb(self, batch_size:int, collection_name:str):
        """
        Inject the prepared data into the Vector DB.
        
        Raises an error if the collection_name already exists in vector DB
        The method prints a confirmation message upon successful data injection.
        """
        # insert data with customized ids
        batches = len(self.docs)//batch_size
        print("Batches: ", batches)
        start = 0           # first primary key id
        total_rt = 0        # total response time for inert

        print(f"inserting {batch_size*batches} entities into example collection: {collection_name}")
        for _ in range(batches):
            rows = [{"batch": self.metadatas[i]["batch"], "source": self.metadatas[i]["source"], 
                     "description": self.metadatas[i]["description"], 
                     "row": self.docs[i], "row_embedding": self.embeddings[i]} #"row_id": self.ids[i]} 
                    for i in range(start, start+batch_size)]
            t0 = time.time()
            self.vectordb.milvus_client.insert(collection_name, rows)
            ins_rt = time.time() - t0
            start += batch_size
            total_rt += ins_rt
        print(f"Succeed in {round(total_rt,4)} seconds!")
        print("Data stored in Vector DB.")

    def _validate_db(self):
        """
        Validate the contents of the database to ensure that the data injection has been successful.
        Prints the number of vectors in the Vector DB collection for confirmation.
        """
        vectordb =  self.vectordb.get_collection(name=self.collection_name)
        print("==============================")
        print("Number of vectors in vectordb:", vectordb.count())
        print("==============================")

    def load_datafile(self, datafile_name: str, datafile_description:str, limit: int=100, batch_size: int=25,
                      chunk_mode: str='batch'):

        df, file_name= self._load_dataframe(os.path.join(self.file_directory,datafile_name), limit)
        print("File readed:", file_name)
        print("File description:", datafile_description) 
        json_data= self.dataframe_to_json_batches(df, batch_size, limit)

        # Generate embeddings from the json data
        if chunk_mode=='row':
            docs, metadata, ids, embeddings= self._generate_embeddings_from_json(json_data, file_name, datafile_description)
        else:
            docs, metadata, ids, embeddings= self._generate_batches_embeddings_from_json(json_data, file_name, datafile_description)

        print("Docs readed: ",len(docs))
        print("Metadata readed: ", len(metadata))
        print("Embeddings created: ", len(embeddings))
        print("Docs: ", docs[0])
        #print("Metadata: ", metadata[0])
        #print("Embeddings: ", embeddings[0])
        #print(eval(json_data[0]))
        #print(type(eval(json_data[0])))
        #print(type(json_data[0]))
        #print(len(json_data))
        self._load_data_into_vectordb(5, self.collection_name)

    def load_data(self, datafiles: list, limit: int, batch_size: int, chunk_mode: str='batch'):
        for datafile in datafiles:
            print(datafile)
            self.load_datafile(datafile["filename"], datafile["description"], limit, batch_size, chunk_mode)
            
        print("Data loaded into Vector DB.")
    
    def test_load(self,):
        df, file_name= self._load_dataframe(os.path.join('data','destino_prov_mes.csv'), 100)
        json_data= self.dataframe_to_json_batches(df, 25, 100)

        docs, metadata, ids, embeddings= self._generate_batches_embeddings_from_json(json_data, file_name, 'gasto medio por visitante y tipo de origen')
        print("Docs readed: ",len(docs))
        print("Metadata readed: ", len(metadata))
        print("Docs: ", docs[0])
        print("Metadata: ", metadata[0])
        print("Ids: ", ids)
        