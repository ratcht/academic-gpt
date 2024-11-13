import arxiv
from typing import Generator
from documents import Document

class ArxivLoader:
  _client = arxiv.Client()

  @classmethod
  def search(cls, query: str, sort_criterion: arxiv.SortCriterion = arxiv.SortCriterion.Relevance):
    # Search for the 10 most recent articles matching the keyword "quantum."
    search = arxiv.Search(
      query = query,
      max_results = 10,
      sort_by = sort_criterion
    )

    results = cls._client.results(search)

    # `results` is a generator; you can iterate over its elements one by one...
    for r in results:
      doc = Document(id=r.entry_id, category=r.primary_category, title=r.title, summary=r.summary, published_on=r.published, pdf_url=r.pdf_url)
      yield doc
