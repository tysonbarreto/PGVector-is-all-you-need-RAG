from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from textwrap import dedent
import json
from src.models import AzureOpenAIClient



class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class Synthesizer:
    SYSTEM_PROMPT:str=dedent("""
    # Role and Purpose
    You are an AI assistant for an e-commerce FAQ system. Your task is to synthesize a coherent and helpful answer 
    based on the given question and relevant context retrieved from a knowledge database.

    # Guidelines:
    1. Provide a clear and concise answer to the question.
    2. Use only the information from the relevant context to support your answer.
    3. The context is retrieved based on cosine similarity, so some information might be missing or irrelevant.
    4. Be transparent when there is insufficient information to fully answer the question.
    5. Do not make up or infer information not present in the provided context.
    6. If you cannot answer the question based on the given context, clearly state that.
    7. Maintain a helpful and professional tone appropriate for customer service.
    8. Adhere strictly to company guidelines and policies by using only the provided knowledge base.
    
    Review the question from the user:
    {question}   
                             
    Context:
    {context}                                                                
    """)

    @staticmethod
    def generate_response(question:str, context:pd.DataFrame)->SynthesizedResponse:
        context_str = context.to_json(orient='records', indent=4)
        messages = [
            {'role':'system','content':Synthesizer.SYSTEM_PROMPT.format(question=question, context=context)}
        ]
        client = AzureOpenAIClient.initialize_client()
        response = client.beta.chat.completions.parse(model='gpt-4.1', messages=messages,response_format=SynthesizedResponse)
        return json.loads(response.choices[0].message.content)
    
if __name__=="__main__":
    __all__ = ["Synthesizer"]