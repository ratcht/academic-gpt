import urllib, urllib.request


class ChunkType:
  def __init__(self, chunk: str, id: str, metadata: dict|None = None):
    self.chunk = chunk
    self.id = id
    self.metadata = metadata

class Document:
  def __init__(self, id, category, title, summary, published_on, pdf_url, is_downloaded=False, doc_path="docs"):
    self.id = id
    self.category = category
    self.title = title
    self.summary = summary
    self.published_on = published_on
    self.pdf_url = pdf_url
    self.doc_path = doc_path
    self.is_downloaded = is_downloaded

  def download_pdf(self):
    file_Path = f'{self.doc_path}/{self.id}.pdf'
    urllib.request.urlretrieve(self.pdf_url, file_Path)

  def to_chunk_type(self) -> ChunkType:
    return ChunkType(
      chunk=(self.title + "\n" + self.summary),
      metadata={"title": self.title, "pdf_url": self.pdf_url, "published_on": self.published_on, "category": self.category},
      id=str(self.id)
    )
  
  def __str__(self):
    return f"Category: {self.category}. Document Title: {self.title}. Published On: {self.published_on}. PDF URL: {self.pdf_url}"

