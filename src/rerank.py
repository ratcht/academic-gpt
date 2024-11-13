import cohere
from documents import Document

class Reranker:
  def __init__(self):
    self._client = cohere.ClientV2()

  def rerank(self, query: str, docs: list[Document], top_n=4, model="rerank-english-v3.0"):

    docs_summaries = [doc.summary for doc in docs]

    response = self._client.rerank(
      model=model,
      query=query,
      documents=docs_summaries,
      top_n=top_n,
    )

    results = [(res.index, res.relevance_score) for res in response.results]

    return results