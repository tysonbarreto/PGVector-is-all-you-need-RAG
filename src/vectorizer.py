from datetime import datetime
import pandas as pd
from typing import Iterator, List
import json

from src.settings import Settings, set_logging
from src.vector_store import VectorStore

from timescale_vector.client import uuid_from_time

logger=set_logging()


@logger.catch()
def preprocess_csv(row,columns:List[str], vec:VectorStore,metdatada:dict={})->pd.Series:
        content = json.dumps({col:row[col] for col in columns})
        embedding = vec.get_embedding(content)
        metadata_={"_created_at_": datetime.now().isoformat()}
        metadata_.update(**metdatada)
        return pd.Series(
            {
                "id": str(uuid_from_time(datetime.now())),
                "metadata": metadata_,
                "contents": content,
                "embedding": embedding,
            }
        )
if __name__=="__main__":
        __all__=["preprocess_csv"]
