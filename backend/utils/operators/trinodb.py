import sys
sys.path.append('./')

from utils.operators.support_query import QueryTemplate
from core.config import Settings
from contextlib import closing
import trino
import time


class SQLOperators:
    """
    A class for handling SQL operations with Trino database.

    This class provides methods for executing queries, managing data operations,
    and handling database connections with Trino. It supports various operations
    including query execution, data insertion, updates, and batch processing.

    Attributes:
        settings (Settings): Configuration settings for database connection
        __dbconn: Trino database connection object

    Methods:
        execute_query: Execute a SQL query and return results as dictionaries
        get_latest_fetching_time: Get the latest successful data fetching time
        data_generator: Generate batches of data from a table
        upsert_dataframe_table: Upsert data into a table with conflict handling
        insert_dataframe_table_nonconflict: Insert data without conflict handling
        insert_dataframe_table: Insert data into a table
        close: Close the database connection

    Note:
        All methods use context managers for cursor handling and include error handling.
        Batch processing is supported for large datasets with configurable chunk sizes.
    """
    def __init__(self, conn_id: str, settings: Settings):
        try:
            self.settings = settings
            self.__dbconn = trino.dbapi.connect(
                host=self.settings.TRINO_HOST,
                port=self.settings.TRINO_PORT,
                user=self.settings.TRINO_USER,
                catalog=self.settings.TRINO_CATALOG,
                schema=conn_id
            )
        except Exception as ex:
            raise Exception(f"====> Can't connect to '{conn_id}' database with host: {self.settings.TRINO_HOST} - {str(ex)}")
    
    def execute_query(self, query):
        """Execute a SQL query and return results as dictionaries."""
        try:
            with closing(self.__dbconn.cursor()) as cursor:
                cursor.execute(query)
                data = cursor.fetchall()
                return [dict(zip([col[0] for col in cursor.description], row)) for row in data]
        except Exception as ex:
            raise Exception(f"====> Can't execute query: {query} - {str(ex)}")
    
    def get_latest_fetching_time(self, layer, table_name):
        """Get the latest successful data fetching time from audit table."""
        query = f"""
            SELECT MAX(end_time) 
            FROM audit
            WHERE status='SUCCESS' AND layer='{layer}' AND table_name='{table_name}'
        """
        try:
            with closing(self.__dbconn.cursor()) as cursor:
                cursor.execute(query)
                latest_time = cursor.fetchone()
                return latest_time[0]
        except Exception as ex:
            raise Exception(f"====> Can't execute {query} - {str(ex)}")
    
    def data_generator(self, table_name, columns, latest_time="1970-01-01T00:00:00.000+00:00", batch_size=10000):
        """Generate batches of data from a table with specified columns."""
        query = QueryTemplate(table_name).create_query_select(columns, latest_time)
        try:
            batch = []
            with closing(self.__dbconn.cursor()) as cursor:
                cursor.execute(query)
                data = cursor.fetchall()
                dataset = [dict(zip([col[0] for col in cursor.description], row)) for row in data]
                for doc in dataset:
                    batch.append(doc)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch
                
        except Exception as ex:
            raise Exception(f"====> Can't execute {query} - {str(ex)}")
    
    def upsert_dataframe_table(self, table_name: str, schema: str, data: list, columns: list, conflict_column: tuple = None, arrjson: list = [], chunk_size=10000):
        """Upsert data into a table with conflict handling in chunks."""
        query = QueryTemplate(table_name, schema).create_query_upsert(columns, conflict_column, arrjson)
        try:
            with closing(self.__dbconn.cursor()) as cursor:
                for i in range(0, len(data), chunk_size):
                    partitioned_data = data[i:i+chunk_size]
                    cursor.execute(query, partitioned_data)
                    print(f"Merged or updated {len(partitioned_data)} records")
        except Exception as ex:
            raise Exception(f"====> Can't execute {query} - {str(ex)}")
        
    def insert_dataframe_table_nonconflict(self, table_name: str, schema: str, data: list, columns: list, conflict_column: tuple = None, arrjson: list = [], chunk_size: int = 10000):
        """Insert data into a table without conflict handling in chunks."""
        query = QueryTemplate(table_name, schema).create_query_insert_nonconflict(columns, conflict_column, arrjson)
        try:
            with closing(self.__dbconn.cursor()) as cursor:
                for i in range(0, len(data), chunk_size):
                    partitioned_data = data[i:i+chunk_size]
                    cursor.execute(query, partitioned_data)
                    print(f"Inserted {len(partitioned_data)} records")
                    time.sleep(1)
        except Exception as ex:
            raise Exception(f"====> Can't execute {query} - {str(ex)}")
        
    def insert_dataframe_table(self, table_name: str, schema: str, data: list, columns: list, arrjson: list = [], chunk_size: int = 10000):
        """Insert data into a table in chunks."""
        query = QueryTemplate(table_name, schema).create_query_insert(columns, arrjson)
        try:
            with closing(self.__dbconn.cursor()) as cursor:
                for i in range(0, len(data), chunk_size):
                    partitioned_data = data[i:i+chunk_size]
                    cursor.execute(query, partitioned_data)
                    print(f"Inserted {len(partitioned_data)} records")
        except Exception as ex:
            raise Exception(f"====> Can't execute {query} - {str(ex)}")
    
    def close(self):
        """Close the database connection."""
        self.__dbconn.close()
