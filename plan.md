So, 

How this will work: 

User queries academic-gpt:
- I get GPT to turn the query into a search query to use on ARXIV
- Get top results from ARXIV (search by relevance)
- Rerank document titles with the query (top 20 ish docs)
- Top 3 Documents, insert them into the prompt. (Process them concurrently, cut out references, include appendix)
- Get the prompt to mark whenever it uses a quote from a document (like a reference [1], and somehow link it to a specific section of the paper)

Build a cool looking frontend. Get the thing deployed. Write up documentation on how I built it (blog post style)

instead of just arxiv, also get a collection of top ML/AI papers. Make it query other top stuff like conferences, etc.
