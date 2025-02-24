import os
import pandas as pd

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

    def _load_dataframe(self, file_directory: str, limit: int):
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
            return df, file_name
        # Excel datafile                    
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_directory, sep= self.csv_sep, nrows=limit, encoding= self.csv_codec)
            return df, file_name
        else:
            raise ValueError("The selected file type is not supported")

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
            #json_batches.append(batch.to_json(orient='records'))
            json_batches+=[eval(batch.to_json(orient='records'))]
        
        return json_batches

    def _generate_embeddings_from_json(self, json_data:list, file_name:str):
        """
        Generate embeddings and prepare documents for data injection.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.
            file_name (str): The base name of the file for use in metadata.
            
        Returns:
            list, list, list, list: Lists containing documents, metadatas, ids, and embeddings respectively.
        """
        docs = []
        metadatas = []
        ids = []
        embeddings = []
        for i, batch in enumerate(json_data):
            #output_str = ""
            # Treat each row as a separate chunk
            for row in batch:
                #output_str += f"{col}: {row[col]},\n"

                response = self.embeddings_model.embed_query(
                    str(row),
                )

                embeddings.append(response)
                docs.append(row)
                metadatas.append({"source": file_name, "batch": i})
                #ids.append(f"id{index}")
            
        return docs, metadatas, ids, embeddings

    def _generate_embeddings(self, df:pd.DataFrame, file_name:str):
        """
        Generate embeddings and prepare documents for data injection.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.
            file_name (str): The base name of the file for use in metadata.
            
        Returns:
            list, list, list, list: Lists containing documents, metadatas, ids, and embeddings respectively.
        """
        docs = []
        metadatas = []
        ids = []
        embeddings = []
        for index, row in df.iterrows():
            output_str = ""
            # Treat each row as a separate chunk
            for col in df.columns:
                output_str += f"{col}: {row[col]},\n"
            response = self.embeddings_model.embed_query(
                output_str,
            )

            embeddings.append(response)
            docs.append(output_str)
            metadatas.append({"source": file_name})
            ids.append(f"id{index}")
            
        return docs, metadatas, ids, embeddings

    def _load_data_into_vectordb(self):
        """
        Inject the prepared data into the Vector DB.
        
        Raises an error if the collection_name already exists in vector DB
        The method prints a confirmation message upon successful data injection.
        """
        collection = self.vectordb.create_collection(name=self.collection_name)
        collection.add(
            documents=self.docs,
            metadatas=self.metadatas,
            embeddings=self.embeddings,
            ids=self.ids
        )
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

    def load_data(self, limit: int):

        df, file_name= self._load_dataframe(self.file_directory, limit)
        print("File readed:", file_name) 
        json_data= self.dataframe_to_json_batches(df, batch_size=25)
        #print("\n JSON:", json_data)
        print(json_data[0])
        for i in json_data[0]:
            print(i)

        docs, metadata, ids, embeddings= self._generate_embeddings_from_json(json_data, file_name)
        print(len(docs))
        print(len(metadata))
        print(len(embeddings))
        #print(eval(json_data[0]))
        #print(type(eval(json_data[0])))
        #print(type(json_data[0]))
        #print(len(json_data))
        return df
        