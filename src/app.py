from concurrent import futures
import logging
from dotenv import load_dotenv
import os
from fastapi import FastAPI, File, UploadFile
import uvicorn

from llm import LLM
from loader import ArxivLoader, arxiv
from rerank import Reranker
from documents import DocumentFormat, Document
import asyncio

load_dotenv()

logging.basicConfig(level=logging.DEBUG, filename='logs/std.log', filemode='w', format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(name="server")

app = FastAPI()

llm = LLM()

@app.get("/")
async def root():
  return {"message": "Hello World"}

@app.post("/search")
async def search(query: str):
  
  # Step 1:
  search_query = llm.template_query("user_to_search", user_query=query)\
  
  # Step 2:
  docs = list(ArxivLoader.search(search_query))

  # Step 3:
  reranked_indices = Reranker.rerank(query, docs)
  reranked_urls = [docs[i].pdf_url for i in reranked_indices]

  # Step 4:
  docs_loaded = await asyncio.gather(*[DocumentFormat.load_document_pdf(pdf_url=url) for url in reranked_urls])

  # Step 5:
  result = llm.rag_query(query, docs_loaded)

  return {"message": result}

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)