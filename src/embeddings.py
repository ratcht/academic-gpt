import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import argparse
import dotenv
import langchain
import logging
from transformers import AutoModel


# For Testing Purposes...
langchain.debug = True
logging.basicConfig(level=logging.INFO)


class Embedding():
  """
  A class that integrates with the Cohere API for conversational AI purposes.

  Attributes:
    __conversation_id (str): Unique ID for tracking the conversation.
    __embedding Union(IndexEmbedding, ChromaEmbedding): an embedding system for the RAG to use
    __max_tokens int: max number of tokns a response to a query can be
  """

  def __init__(self, embedding_model: str = "openai", hf_model: str|None = None, *args, **kwargs):
    """
    Initializes the Chat class, setting up the embedding model used for queries.

    Args:
      num_matches (int): Number of matching documents to return upon a query.
    """
    self.__embedding_model = embedding_model
    dotenv.load_dotenv()
    self.__key = os.getenv(f'{self.__embedding_model.upper()}_API_KEY')
    self.__org = os.getenv(f'{self.__embedding_model.upper()}_ORG')

    if self.__embedding_model == "openai":
      self.__client = OpenAIEmbeddings(
          api_key=self.__key, model="text-embedding-3-large", organization=self.__org)
    elif self.__embedding_model == "hf":
      # load in model seperately so it can trust remote code
      self.__client = HuggingFaceEmbeddings(model_name=hf_model,model_kwargs={"trust_remote_code":True})



  def embed_documents(self, documents: list[str]) -> str:
    """
    Embed a list of documents.

    Args:
      query (str): The query string.

    Returns:
      str: The response from the language model.
    """
    embeddings = self.__client.embed_documents(documents)
    logging.info(f"Embeddings Result: {len(embeddings), len(embeddings[0])}")

    return embeddings
  
  def embed_query(self, query: str) -> str:
    """
    Embed a list of documents.

    Args:
      query (str): The query string.

    Returns:
      str: The response from the language model.
    """
    embeddings = self.__client.embed_query(query)
    logging.info(f"Embeddings Result: {len(embeddings)}")

    return embeddings





if __name__ == "__main__":
  embedding = Embedding(embedding_model="hf", hf_model="Alibaba-NLP/gte-large-en-v1.5")
  
  print(embedding.embed_query("hello"))
