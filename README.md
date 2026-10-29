# Research Paper Recommender

A RAG-based application for discovering, analyzing, and chatting with recent AI/ML research papers from Hugging Face Daily Papers.

## Architecture

```
HF Daily Papers API → Ingest → Parse PDFs → Chunk → Embed → FAISS
                                                               ↓
Streamlit UI → Hybrid Search → Paper Detail → LangChain Paper Chat
```

## Quickstart

```bash
# 1. Clone and set up environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env and set GROQ_API_KEY, GROQ_MODEL_ID, GEMINI_API_KEY,
# GEMINI_MODEL_ID, and HF_API_TOKEN

# 3. Ingest the last 7 days of computer vision papers, downloading the top 10 by upvotes
python scripts/ingest.py --days 7 --top 10 --keyword "computer vision"

# 4. Build or incrementally update the FAISS index
python scripts/build_index.py

# Optional: run a hybrid search query against the index
python scripts/build_index.py --query "models for visual perception"

# 5. Optional: analyze an indexed paper with Groq and Gemini fallback
python scripts/analyze_paper.py --paper-id 2609.36380

# Ask a focused question about that paper
python scripts/analyze_paper.py --paper-id 2609.36380 --question "What is its main contribution and evidence?"

# 6. Optional: chat with a paper; follow-up turns reuse this session's history
python scripts/chat_paper.py --paper-id 2609.36380

# For specific paper_id search
python -c "from src.vectorstore import FaissVectorStore; s=FaissVectorStore().load(); r=s.search('What approach does this paper use for 3D scene reconstruction?', top_k=5, search_mode='hybrid', paper_id='2609.36380'); [print(x.chunk.paper_id, x.chunk.section, x.chunk.text[:100]) for x in r]"

# Use --rebuild after changing/removing paper data or chunking settings
python scripts/build_index.py --rebuild

# 7. Launch the app (Stage 7)
streamlit run app/main.py
```

The ingestion command keeps metadata for all distinct papers returned in the date
range, but downloads and parses PDFs only for the top `--top` papers by Hugging
Face upvotes (`10` by default). Papers with positive upvotes are selected first,
ordered by vote count and then ID. If fewer than `--top` unprocessed, upvoted
papers are available, the remaining slots are filled by the newest matching papers.
When `--keyword` is provided, the feed is filtered case-insensitively by substring
matches in each paper's title or abstract before ranking and cataloging. Without
it, all papers are eligible. Use `--no-pdf` for metadata-only ingestion.

Indexing is incremental by default: papers already represented in the existing
FAISS metadata are skipped before chunking and embedding. Use
`python scripts/build_index.py --rebuild` to recreate the index when paper text,
metadata, or chunking settings have changed.

Search combines FAISS semantic rankings with BM25 keyword rankings using
reciprocal rank fusion (RRF). This supports both conceptual queries and exact
technical terms. Call `FaissVectorStore.search(..., search_mode="semantic")` or
`search_mode="keyword"` to use one ranking method on its own; the default is
`search_mode="hybrid"`.

- `Semantic` ranks by embedding similarity.
- `Hybrid` combines semantic and keyword (BM25) rankings.
- `Keyword` uses BM25 alone.

Stage 5 analysis uses Groq as the primary provider and uses Gemini when Groq's
key is missing or Groq encounters rate limits, connection/timeouts, and
transient service errors. At least one provider key must be set. Configure:
`GROQ_API_KEY`, `GROQ_MODEL_ID`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), and
`GEMINI_MODEL_ID` (or `GOOGLE_MODEL_ID`) in `.env`. It retrieves evidence only
from the requested paper and returns the analysis with paper, section, and page
references plus the provider/model used. Authentication, permission, and
unrelated request errors are surfaced; a Groq `model_not_found` error triggers
Gemini fallback. The paper must already exist in the catalog and vector index.

Stage 6 provides LangChain follow-up chat through `scripts/chat_paper.py`. Each
turn retrieves fresh hybrid evidence from the selected paper and returns source
references. The bounded in-memory cache keeps up to six turns per session and
128 least-recently-used sessions; sessions are cleared when the process exits.
Enter `/exit` to end the chat.

Stage 7 opens on the **Ingest papers** workflow. Enter topic keywords and choose
an inclusive date range (the end date defaults to today and is fixed; the start
date defaults to 30 days earlier and can be changed) to fetch the top five
matching Hugging Face Daily Papers. Results can be ranked by votes (with newest
papers filling any remaining slots, as in Stage 1) or by latest date. Select one
or all results and submit to download/parse PDFs and incrementally add chunks and
embeddings to FAISS. The UI also provides paper details, grounded analysis, and
paper-scoped follow-up chat with source references. Use the sidebar to switch
between workflows. `Reload paper index` refreshes an index rebuilt while the UI
process is running.

## Project Structure

```
src/
├── config.py          # Central settings (env vars, paths, model IDs)
├── models.py          # Pydantic data models
├── ingestion/         # HF feed fetch, PDF download & parse, dedup
├── processing/        # Chunking and text normalization
├── embedding.py       # Sentence Transformers wrapper
├── vectorstore.py     # FAISS build / persist / load and semantic/hybrid search
├── generation/        # Groq/Gemini provider fallback and grounded analysis 
├── chat/              # LangChain paper-scoped follow-up chat 
└── evaluation/        # Metrics and test cases 

app/                   # Streamlit UI 
scripts/               # CLI ops: ingest, build_index, evaluate
data/
├── raw/               # Downloaded PDFs 
├── parsed/            # Extracted JSON 
└── index/             # FAISS index files 
```

## Stage Roadmap

| Stage | Focus | Status |
|---|---|---|
| 1 | Ingest HF Daily Papers metadata | ✅ Implemented |
| 2 | Download and parse PDFs | ✅ Implemented |
| 3 | Build FAISS retrieval index | ✅ Implemented |
| 4 | Semantic/hybrid search + date filters | ✅ Implemented |
| 5 | Groq analysis with Gemini fallback | ✅ Implemented |
| 6 | LangChain paper-scoped follow-up chat with bounded memory | ✅ Implemented |
| 7 | Streamlit UI | ✅ Implemented |
| 8 | LangGraph state graph | 🔜 Next |
| 9 | Evaluation suite | 🔜 Next |
| 10 | Deploy + document | 🔜 Next |
