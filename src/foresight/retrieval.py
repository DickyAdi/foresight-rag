"""Build a per-question ephemeral Chroma vector store from a HotPotQA context pool."""
# TODO: implement build_store(context_pool) -> Chroma
# - Use chromadb.EphemeralClient() (no persistence dir)
# - Use HuggingFaceEmbeddings("all-MiniLM-L6-v2")
# - Assign a unique collection name per question (e.g. uuid4 hex)
# - Ingest all sentences from all articles (distractors included on purpose)
# - Return the LangChain Chroma object for use by tools.py
# - Caller is responsible for deleting the collection after the case is done
