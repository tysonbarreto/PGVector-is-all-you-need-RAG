
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
vec.delete(delete_all=True)
vec.upsert(records)
########################### SIMILARITY SEARCH WITHOUT METADATA##########################

from src.synthesizer import Synthesizer
from src.vector_store import VectorStore
from pprint import pprint
vec = VectorStore()
relevant_question = "What are your shipping options?"
results = vec.search(relevant_question, limit=3)

print(results['content'].to_json(orient='records', index=4))
results

response = Synthesizer.generate_response(question=relevant_question, context=results['content'])

print(response)

########################### SIMILARITY SEARCH WITH METADATA##########################

from src.synthesizer import Synthesizer
from src.vector_store import VectorStore
from pprint import pprint
vec = VectorStore()
metdata_filter = {'category':'Shipping'}

relevant_question = "What are your shipping options?"
results = vec.search(query_text=relevant_question,limit=3, metadata_filter=metdata_filter)

response = Synthesizer.generate_response(question=relevant_question, context=results['content'])

import json
print(response)

########################### SIMILARITY SEARCH WITH PREDICATES##########################
from timescale_vector import client


predicates = client.Predicates("category", "==", "Shipping")

# predicates = client.Predicates("category", "==", "Shipping") | client.Predicates(
#     "category", "==", "Services"
# )

# predicates = client.Predicates("category", "==", "Shipping") & client.Predicates(
#     "created_at", ">", "2024-09-01"
# )


results = vec.search(relevant_question, limit=3, predicates=predicates)

response = Synthesizer.generate_response(question=relevant_question, context=results['content'])

print(response)

########################### TIME BASED FILTERING ##########################
from datetime import datetime
time_range = (datetime(2024, 9, 1), datetime(2024, 9, 30))
results = vec.search(relevant_question, limit=3, time_range=time_range)
response = Synthesizer.generate_response(question=relevant_question, context=results['content'])

print(response)
