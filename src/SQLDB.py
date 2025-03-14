import os
from langchain_community.utilities import SQLDatabase

class SQLDB:

    def __init__(self,sqldb_dir: str) -> None:
        """
        Initialize an instance of PrepareSQLFromTabularData.

        Args:
            files_dir (str): The directory containing the CSV or XLSX files to be converted to SQL tables.
        """

        # Set the DB
        if os.path.exists(sqldb_dir):
            self.db = SQLDatabase.from_uri(
                        f"sqlite:///{sqldb_dir}")

        self.sqldb_dir= sqldb_dir
