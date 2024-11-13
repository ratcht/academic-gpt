import urllib, urllib.request
from pydantic import BaseModel, Field
from pyzerox import zerox

class ChunkType:
  def __init__(self, chunk: str, id: str, metadata: dict|None = None):
    self.chunk = chunk
    self.id = id
    self.metadata = metadata

class DocumentFormat(BaseModel):
  """
  This class defines the structure for the document (split up into intro, references, etc.).

  Use zero ocr from omniAI
  """
  def __init__(self):
    pass

  @classmethod
  async def load_document_pdf(cls, pdf_url: str, model: str="gpt-4o", drop_references=True):
    ## process only some pages or all
    select_pages = None ## None for all, but could be int or list(int) page numbers (1 indexed)

    output_dir = "./output_test" ## directory to save the consolidated markdown file
    result = await zerox(file_path=pdf_url, model=model, output_dir=output_dir, select_pages=select_pages)

    document = ""
    
    for page in result.pages:
      document += page.content

    document = document.split("References")[0] if drop_references else document

    return document


class Document(BaseModel):
  id: int = Field(..., description="Unique identifier for the document")
  title: str = Field(..., description="Title of the document")
  category: str = Field(..., description="Category or classification of the document")
  summary: str = Field(..., description="Brief summary or abstract of the document")
  published_on: str = Field(..., description="Publication date of the document")
  pdf_url: str = Field(..., description="URL to access the PDF document")


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

