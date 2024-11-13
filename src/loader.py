import arxiv
from typing import Generator
from documents import Document

class ArxivLoader:
  def __init__(self):
    self._client = arxiv.Client()

  def search(self, query: str, sort_criterion: arxiv.SortCriterion = arxiv.SortCriterion.Relevance):
    # Search for the 10 most recent articles matching the keyword "quantum."
    search = arxiv.Search(
      query = query,
      max_results = 10,
      sort_by = sort_criterion
    )

    results = self._client.results(search)

    # `results` is a generator; you can iterate over its elements one by one...
    for r in results:
      doc = Document(r.entry_id, r.primary_category, r.title, r.summary, r.published, r.pdf_url, False)
      yield doc
