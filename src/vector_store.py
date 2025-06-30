
import time
from typing import Any, List, Optional, Tuple, Union
from textwrap import dedent
from datetime import datetime
import pandas as pd
from contextlib import ExitStack

from src.settings import get_settings, Settings, VectorStoreSettings
from src.models import AzureOpenAIClient

from openai import OpenAI, AzureOpenAI
from src.timescale_vector import client
from src.settings import set_logging
from src.utils import Utils


import psycopg2


logger = set_logging()

utils= Utils()

class VectorStore():
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        super().__init__()
        self.settings:Settings = Settings()
        self.openai_client:AzureOpenAIClient = AzureOpenAIClient.initialize_client()
        self.embedding_model:str = self.settings.openai.embedding_model
        self.vector_settings:VectorStoreSettings = self.settings.vector_store
        self.vec_client = client.Sync(
            service_url=self.settings.database.service_url,
            table_name=self.vector_settings.table_name,
            num_dimensions=self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval
        )
    @logger.catch
    def create_keyword_search_index(self):
        """Create a GIN index for keyword search if it doesn't exist."""
        index_name = f"idx_{self.vector_settings.table_name}_contents_gin"
        create_index_sql=f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {self.vector_settings.table_name} USING gin(to_tsvector('english',contents));
        """
        try:
            with ExitStack() as conext_sync_stack:
                conn =  conext_sync_stack.enter_context(psycopg2.connect(self.settings.database.service_url))
                cur = conext_sync_stack.enter_context(conn.cursor())
                cur.execute(create_index_sql)
                conn.commit()
                logger.info(f"GIN index '{index_name}' created or already exists.")
        except Exception as e:
            logger.error(f"Error while creating GIN index: {str(e)}")

    def get_embedding(self, text:str)-> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embeddings = self.openai_client.embeddings.create(
            input = [text],
            model =self.embedding_model,
            dimensions=self.vector_settings.embedding_dimensions
        ).data[0].embedding
        elapsed_time = time.time() - start_time
        logger.info(f"{len(embeddings)} os embeddings  generated in {elapsed_time:.3f} seconds")    
        return embeddings
    
    def create_tables(self) -> None:
        """Create the necessary tablesin the database"""
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to spseed up similarity search"""
        self.vec_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logger.info(
            f"Inserted {len(df)} records into {self.vector_settings.table_name}"
        )
    
    def search(self, query_text:str,
               limit:int=3, 
               metadata_filter:Union[dict, List[dict]]=None, 
               predicates:Optional[client.Predicates]=None, 
               time_range:Optional[Tuple[datetime, datetime]]=None,
               return_df:bool=True)-> Union[List[Tuple], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        More info:
            https://github.com/timescale/docs/blob/latest/ai/python-interface-for-pgvector-and-timescale-vector.md

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
                - Predicates objects are defined by the name of the metadata key, an operator, and a value.
                - Operators: ==, !=, >, >=, <, <=
                - & is used to combine multiple predicates with AND operator.
                - | is used to combine multiple predicates with OR operator.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.

        Basic Examples:
            Basic search:
                vector_store.search("What are your shipping options?")
            Search with metadata filter:
                vector_store.search("Shipping options", metadata_filter={"category": "Shipping"})
        
        Predicates Examples:
            Search with predicates:
                >>> vector_store.search("Pricing", predicates=client.Predicates("price", ">", 100))
            Search with complex combined predicates:

                >>> complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
                >>> vector_store.search("High-quality products", predicates=complex_pred)
        Time-based filtering:
            Search with time range:
                >>> vector_store.search("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        """
                
        query_embeddings = self.get_embedding(query_text)
        start_time = time.time()

        search_args = {
            "limit":limit,
        }
        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_end = time_range
            search_args['uuid_time_filter'] = client.UUIDTimeRange(start_date=start_date, end_date=end_end)
        
        results = self.vec_client.search(query_embedding=query_embeddings, **search_args)

        elapsed_time = time.time() - start_time

        logger.info(f"Vector search completed in {elapsed_time:.3f} seconds")

        if return_df:
            return self._create_dataframe_from_results(results)
        else:
            return results
        
    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
        )

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df
    
    @utils.log(log_info='Keyword Search')
    def keyword_search(self,query:str, limit:int=5, return_dataframe:bool=True)->Union[List[Tuple[str, str, float]],pd.DataFrame]:
        """
        Perform a keyword search on the contents of the vector store.

        Args:
            query: The search query string.
            limit: The maximum number of results to return. Defaults to 5.
            return_dataframe: Whether to return results as a DataFrame. Defaults to True.

        Returns:
            Either a list of tuples (id, contents, rank) or a pandas DataFrame containing the search results.

        Example:
        >>> results = vector_store.keyword_search("shipping options")
        """
        search_sql = f"""
        SELECT id, contents, ts_rank_cd(to_tsvector('english', contents), query) as rank
        FROM {self.vector_settings.table_name}, websearch_to_tsquery('english', %s) query
        WHERE to_tsvector('english',contents) @@query
        ORDER BY rank DESC
        LIMIT %s
        """

        with ExitStack() as context_sync_stack:
            conn = context_sync_stack.enter_context(psycopg2.connect(self.settings.database.service_url))
            cur = conn.cursor()
            cur.execute(search_sql, (query, limit))
            results = cur.fetchall()

        if return_dataframe:
            df = pd.DataFrame(results,columns=["id","content","rank"])
            df['id'] = df['id'].astype(str)
            return df
        else:
            return results
    
    def hybrid_search(self, query:str, keyword_k:int=5, semantic_k:int=5, rerank:bool=False, top_n:int=5)->pd.DataFrame:
        """
        Perform a hybrid search combining keyword and semantic search results,
        with optional reranking using Cohere.

        Args:
            query: The search query string.
            keyword_k: The number of results to return from keyword search. Defaults to 5.
            semantic_k: The number of results to return from semantic search. Defaults to 5.
            rerank: Whether to apply Cohere reranking. Defaults to True.
            top_n: The number of top results to return after reranking. Defaults to 5.

        Returns:
            A pandas DataFrame containing the combined search results with a 'search_type' column.

        Example:
            >>> results = vector_store.hybrid_search("shipping options", keyword_k=3, semantic_k=3, rerank=True, top_n=5)
        """

        keyword_results = self.keyword_search(query=query, limit=keyword_k, return_dataframe=True)
        keyword_results["search_type"] = "keyword"

        keyword_results = keyword_results[['id','content','search_type']]

        semantic_results = self.search(query_text=query, limit=5, return_df=True)
        semantic_results['search_type'] ='semantic_search'
        semantic_results = semantic_results[['id','content','search_type']]

        hybrid_results = pd.concat([keyword_results, semantic_results])
        hybrid_results.drop_duplicates(subset=['id'], keep='first', inplace=True)

        return hybrid_results


    def delete(self, ids:List[str]=None, metadata_filter:dict=None, delete_all:bool = False)->None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """

        if (sum(x) for x in (ids, metadata_filter, delete_all)) !=1:
            logger.error("Provide exactly one of: ids, metadata_filter, or delete_all")
            raise ValueError("Provide exactly one of: ids, metadata_filter, or delete_all")
        
        if delete_all:
            self.vec_client.delete_all()
            logger.info(f"Deleted all records from {self.vector_settings.table_name}")

        elif ids:
            self.vec_client.delete_by_ids(ids)
            logger.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )

        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logger.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )

if __name__ == "__main__":
    __all__=["VectorStore"]