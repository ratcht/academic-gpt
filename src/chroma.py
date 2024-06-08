import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from embeddings import Embedding
from documents import ChunkType, Document
import os

class ChromaEmbeddingFunction(EmbeddingFunction):
  def __init__(self, embedding_model, *args, **kwargs):
    self.__embedding_client = Embedding(embedding_model=embedding_model, *args, **kwargs)

  def __call__(self, input: Documents) -> Embeddings:
    embeddings = self.__embedding_client.embed_documents(input)
    return embeddings
    

class ChromaDB:
  def __init__(self, default_collection="default", embedding_model="openai", distance_function="cosine", use_http_client=False, http_host="localhost", http_port=8000, *args, **kwargs):
    if use_http_client:
      self.__client = chromadb.HttpClient(host=http_host, port=http_port)
    else:
      self.__client = chromadb.PersistentClient(
        path = os.path.join(os.getcwd(), "db") 
      )
    
    self.__ef = ChromaEmbeddingFunction(embedding_model, *args, **kwargs)

    self.__collection = self.__client.get_or_create_collection(
        name=default_collection,
        embedding_function=self.__ef,
        metadata={"hnsw:space": distance_function}
      )

  def delete_collection(self, collection):
    self.__client.delete_collection(name=collection)

  def load_collection(self, collection, distance_function="cosine"):
    self.__collection = self.__client.get_or_create_collection(
          name=collection,
          embedding_function=self.__ef,
          metadata={"hnsw:space": distance_function}
        )
    
  def add(self, chunked_documents: list[ChunkType]):
    documents = []
    metadatas = []
    ids = []
    for chunk in chunked_documents:
      documents.append(chunk.chunk)
      metadatas.append(chunk.metadata)
      ids.append(chunk.id)
    self.__collection.add(
      documents=documents,
      metadatas=metadatas,
      ids=ids
    )


  def query(self, query_texts: list[str], n_results=10, where=None, where_document=None):
    results = self.__collection.query(
      query_texts=query_texts,
      n_results=n_results,
      where=where,
      where_document=where_document
    )


    # remove duplicate documents (due to documents being crosslisted in diff cats.)
    titles = set()
    indices = set()

    for i, id in enumerate(results["ids"][0]):
      if results["metadatas"][0][i]["title"] in titles:
        indices.add(i)
      titles.add(results["metadatas"][0][i]["title"])

    metadatas = [meta for i, meta in enumerate(results["metadatas"][0]) if i not in indices]
    docs = [doc for i, doc in enumerate(results["documents"][0]) if i not in indices]
    distances = [dist for i, dist in enumerate(results["distances"][0]) if i not in indices]
    ids = [id for i, id in enumerate(results["ids"][0]) if i not in indices]

    return ids, distances, docs, metadatas

  def peek(self):
    return self.__collection.peek()

  def count(self):
    return self.__collection.count()

  def rename_collection(self, new_name: str):
    return self.__collection.modify(name=new_name)
  
  def reset_collection(self):
    return self.__collection.delete(ids=self.__collection.get()['ids'])
    
  def heartbeat(self):
    return self.__client.heartbeat()
    

if __name__ == "__main__":
  pass
