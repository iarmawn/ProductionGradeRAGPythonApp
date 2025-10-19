import logging
import os
import uuid
import datetime
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import pypdf
import numpy as np
from typing import List, Dict, Any

load_dotenv()

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Qdrant client
def get_qdrant_client():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
    else:
        return QdrantClient(url=qdrant_url, timeout=30)

# Initialize Qdrant collection
def init_qdrant_collection():
    client = get_qdrant_client()
    collection_name = "docs"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    return client

# Simple PDF text extraction
def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Simple text chunking
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks

# Get embeddings from OpenAI
def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [embedding.embedding for embedding in response.data]

# Inngest client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load_and_chunk(pdf_path: str, source_id: str):
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        return {"chunks": chunks, "source_id": source_id}

    def _embed_and_upsert(data):
        chunks = data["chunks"]
        source_id = data["source_id"]
        
        # Get embeddings
        embeddings = get_embeddings(chunks)
        
        # Initialize Qdrant
        client = init_qdrant_collection()
        
        # Create points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={"source": source_id, "text": chunk}
            ))
        
        # Upsert to Qdrant
        client.upsert("docs", points=points)
        return {"ingested": len(chunks)}

    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", pdf_path)
    
    data = await ctx.step.run("load-and-chunk", lambda: _load_and_chunk(pdf_path, source_id))
    result = await ctx.step.run("embed-and-upsert", lambda: _embed_and_upsert(data))
    return result

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5):
        # Get query embedding
        query_embedding = get_embeddings([question])[0]
        
        # Search in Qdrant
        client = get_qdrant_client()
        results = client.search(
            collection_name="docs",
            query_vector=query_embedding,
            with_payload=True,
            limit=top_k
        )
        
        contexts = []
        sources = set()
        for result in results:
            payload = result.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k))

    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found["sources"], "num_contexts": len(found["contexts"])}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
