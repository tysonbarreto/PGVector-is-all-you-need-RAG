
import pandas as pd

from src.vector_store import VectorStore
from src.vectorizer import preprocess_csv
from src.models import AzureOpenAIClient


vec = VectorStore()
########################### LOAD DATA INTO POSTGRES VECTOR DB#########################
df = pd.read_csv('data/faq_dataset.csv', sep=";")

df.head(10)

records = df.apply(lambda x: preprocess_csv(row=x, columns=df.columns, vec=vec,metdatada={"category":x['category']}), axis=1)

records.head()

vec.create_tables()
vec.create_index()
vec.create_keyword_search_index()
#vec.delete(delete_all=True)
vec.upsert(records)

######################## KEYWORD SEARCH ####################################
query = "Is there any news related to London?"
keyword_results = vec.keyword_search(query=query, limit=5)

print(keyword_results)

query = "What are the shipping options?"
keyword_results = vec.keyword_search(query=query, limit=5)

keyword_results

######################## HYBRID SEARCH ####################################

query = "Is there any news related to London?"
keyword_results = vec.hybrid_search(query=query)

context = keyword_results['content'].to_json(orient='records', indent=4)

from src.synthesizer import Synthesizer

response = Synthesizer.generate_response(question=query, context=keyword_results['content'])

from pprint import pprint
pprint(f"{context}\n{'='*100}\n{response}")