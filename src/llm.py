import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.globals import set_verbose
import argparse
import dotenv
import langchain
import logging
import numpy as np
from chroma import ChromaDB

# For Testing Purposes...
langchain.debug = True
logging.basicConfig(level=logging.ERROR)


class LLM():
  """
  A class that integrates with the OpenAI API for conversational AI purposes.
  """

  def __init__(self, llm: str = "openai", verbose=False):
    """
    Initializes the Chat class, setting up the embedding model used for queries.

    Args:
      num_matches (int): Number of matching documents to return upon a query.
    """

    dotenv.load_dotenv()
    self.__key = os.getenv(f'{llm.upper()}_API_KEY')
    self.__org = os.getenv(f'{llm.upper()}_ORG')

    if llm == "openai":
      self.__client = ChatOpenAI(
          temperature=0, openai_api_key=self.__key, verbose=False, model="gpt-4-0125-preview", organization=self.__org)
    else:
      raise Exception("No other model currently supported!")

    self.__history = []
    self.__token_usage = np.zeros(shape=(3))  # ['completion_tokens', 'prompt_tokens', 'total_tokens'], Does not track streaming tokens


  def rag_query(self, query, chroma_client: ChromaDB, system_message="You're a helpful assistant") -> str:
    """
    Processes a query using LLM.

    Args:
      query (str): The query string.

    Returns:
      str: The response from the language model.
    """
    

    query_message = HumanMessage(content=query)

    messages = [
      SystemMessage(content=system_message),
      *self.__history,
      query_message
    ]

    response = self.__client.invoke(input=messages)

    token_usage: dict = response.response_metadata["token_usage"]
    logging.debug(token_usage)
    token_stats = np.array(list(token_usage.values()))
    self.__token_usage += token_stats

    content_response = response.content
    self.__history.append(query_message)
    self.__history.append(AIMessage(content=content_response))

    return content_response

  def query(self, query, system_message="You're a helpful assistant") -> str:
    """
    Processes a query using LLM.

    Args:
      query (str): The query string.

    Returns:
      str: The response from the language model.
    """
    query_message = HumanMessage(content=query)

    messages = [
      SystemMessage(content=system_message),
      *self.__history,
      query_message
    ]

    response = self.__client.invoke(input=messages)

    token_usage: dict = response.response_metadata["token_usage"]
    logging.debug(token_usage)
    token_stats = np.array(list(token_usage.values()))
    self.__token_usage += token_stats

    content_response = response.content
    self.__history.append(query_message)
    self.__history.append(AIMessage(content=content_response))

    return content_response

  def stream_query(self, query, system_message="You're a helpful assistant"):
    """
    Stream a query using LLM.

    Args:
      query (str): The query string.

    Returns:
      str: The response from the language model.
    """
    query_message = HumanMessage(content=query)

    messages = [
      SystemMessage(content=system_message),
      *self.__history,
      query_message
    ]

    response = self.__client.stream(input=messages)
    out = ""
    for chunk in response:
      current_content = chunk
      out += current_content.content
      yield current_content.content
    out = AIMessage(content=out)
    self.__history.append(query_message)
    self.__history.append(out)

  def end_chat(self) -> None:
    """
    Cleans up resources
    """
    pass


if __name__ == "__main__":
  pass
