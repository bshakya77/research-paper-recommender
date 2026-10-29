"""Streamlit UI for paper discovery, analysis, and paper-scoped chat."""

from datetime import date, timedelta
import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.deduplicator import load_catalog
from src.models import Paper, PaperAnalysis, PaperChatResponse


st.set_page_config(
    page_title="Research Paper Recommender",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stLinkButton"] a[href*="huggingface.co/papers/"] {
        background-color: #f4c542 !important;
        border-color: #d9ae25 !important;
        color: #242424 !important;
    }
    div[data-testid="stLinkButton"] a[href*="arxiv.org/pdf/"] {
        background-color: #2e8b57 !important;
        border-color: #267449 !important;
        color: white !important;
    }
    div[data-testid="stLinkButton"] a[href*="huggingface.co/papers/"]:hover {
        background-color: #e5b72f !important;
        border-color: #c59d1e !important;
        color: #242424 !important;
    }
    div[data-testid="stLinkButton"] a[href*="arxiv.org/pdf/"]:hover {
        background-color: #267449 !important;
        border-color: #205f3d !important;
        color: white !important;
    }
    div[class*="st-key-discovery-open-"] button {
        background-color: #f4c542 !important;
        border-color: #d9ae25 !important;
        color: #242424 !important;
    }
    div[class*="st-key-discovery-open-"] button:hover {
        background-color: #e5b72f !important;
        border-color: #c59d1e !important;
        color: #242424 !important;
    }
    div[class*="st-key-paper-delete-cancel"] button {
        background-color: #2e8b57 !important;
        border-color: #267449 !important;
        color: white !important;
        width: 150px !important;
        min-width: 150px !important;
        max-width: 150px !important;
        white-space: nowrap !important;
    }
    div[class*="st-key-paper-delete-cancel"] button:hover {
        background-color: #267449 !important;
        border-color: #205f3d !important;
        color: white !important;
    }
    div[class*="st-key-paper-delete-confirm"] button {
        width: 220px !important;
        min-width: 220px !important;
        max-width: 220px !important;
        white-space: nowrap !important;
    }
    div[class*="st-key-paper-workspace-delete"] button {
        background-color: #dc3545 !important;
        border-color: #c82333 !important;
        color: white !important;
        width: 200px !important;
        min-width: 200px !important;
        max-width: 200px !important;
        white-space: nowrap !important;
    }
    div[class*="st-key-paper-workspace-delete"] button:hover {
        background-color: #c82333 !important;
        border-color: #bd2130 !important;
        color: white !important;
    }
    .timed-success-notice {
        background: #eaf7ec;
        border-radius: 0.5rem;
        color: #267449;
        margin: 0 0 1rem;
        overflow: hidden;
        padding: 0.85rem 1rem;
        animation: timed-success-dismiss 10s linear forwards;
    }
    @keyframes timed-success-dismiss {
        0%, 90% {
            max-height: 100px;
            opacity: 1;
            padding-bottom: 0.85rem;
            padding-top: 0.85rem;
            margin-bottom: 1rem;
        }
        100% {
            max-height: 0;
            opacity: 0;
            padding-bottom: 0;
            padding-top: 0;
            margin-bottom: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def _load_catalog():
    return load_catalog()


def _indexed_papers(catalog: dict[str, Paper]) -> dict[str, Paper]:
    from src.vectorstore import get_vectorstore

    indexed_ids = {chunk.paper_id for chunk in get_vectorstore().chunks}
    return {paper_id: paper for paper_id, paper in catalog.items() if paper_id in indexed_ids}


def _open_paper(paper_id: str) -> None:
    st.session_state["workspace_paper_id"] = paper_id
    st.session_state["active_page"] = "Paper workspace"


def _show_timed_success(message: str) -> None:
    st.markdown(
        f'<div class="timed-success-notice">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def _clear_active_chat() -> None:
    session_id = st.session_state.get("chat_session_id")
    if session_id:
        from src.chat import clear_chat_session

        clear_chat_session(session_id)
    st.session_state["chat_session_id"] = None
    st.session_state["chat_paper_id"] = st.session_state.get("workspace_paper_id")
    st.session_state["chat_turns"] = []


def _handle_paper_change() -> None:
    current_paper_id = st.session_state.get("workspace_paper_id")
    previous_paper_id = st.session_state.get("chat_paper_id")
    if previous_paper_id and current_paper_id != previous_paper_id:
        _clear_active_chat()
    st.session_state["chat_paper_id"] = current_paper_id


@st.dialog("If you want to delete the paper?")
def _confirm_delete_paper(paper_id: str, title: str) -> None:
    st.write(
        f"Delete **{title}** ({paper_id})? This removes its embeddings, chunks, "
        "catalog entry, parsed text, and downloaded PDF if available."
    )
    confirm_col, cancel_col = st.columns([1, 1], gap="large")
    with confirm_col:
        with st.container(key="paper-delete-confirm"):
            if st.button("Yes, delete paper", type="primary", key=f"confirm-delete-{paper_id}"):
                try:
                    from src.paper_management import delete_paper_data

                    delete_paper_data(paper_id)
                except Exception as exc:
                    st.error(f"Could not delete the paper: {exc}")
                else:
                    st.session_state.setdefault("paper_analyses", {}).pop(paper_id, None)
                    _clear_active_chat()
                    st.session_state.pop("chat_notice", None)
                    _load_catalog.clear()
                    st.session_state["paper_deletion_notice"] = title
                    st.rerun()
    with cancel_col:
        with st.container(key="paper-delete-cancel"):
            if st.button("Cancel", key=f"cancel-delete-{paper_id}"):
                st.rerun()


def _show_evidence(refs) -> None:
    if not refs:
        return
    with st.expander("Sources"):
        for number, ref in enumerate(refs, start=1):
            st.markdown(
                f"**[{number}] {ref.paper_id} · {ref.section}, page {ref.page}**\n\n"
                f"> {ref.snippet}"
            )


def _render_discovery(papers: dict[str, Paper]) -> None:
    st.title("Discover research papers")
    st.write("Search the indexed papers by topic, method, or exact technical terms.")

    with st.form("paper_search"):
        query = st.text_input("Search query", placeholder="e.g. visual perception for autonomous driving")
        col_mode, col_count = st.columns([2, 1])
        with col_mode:
            search_mode = st.selectbox(
                "Search mode",
                options=["hybrid", "semantic", "keyword"],
                help="Hybrid combines semantic similarity and BM25 keyword ranking.",
            )
        with col_count:
            result_count = st.slider("Papers to show", min_value=1, max_value=20, value=5)

        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Start date", value=date.today() - timedelta(days=30)
            )
        with col_end:
            end_date = st.date_input(
                "End date", value=date.today(), disabled=True
            )

        submitted = st.form_submit_button("Search papers", type="primary")

    if submitted:
        if not query.strip():
            st.session_state["search_results"] = []
            st.warning("Enter a search query.")
        elif start_date > end_date:
            st.session_state["search_results"] = []
            st.error("Start date must be on or before the end date.")
        else:
            with st.spinner("Searching the paper index..."):
                try:
                    from src.vectorstore import get_vectorstore

                    store = get_vectorstore()
                    hits = store.search(
                        query.strip(),
                        top_k=min(len(store.chunks), result_count * 4),
                        start_date=start_date,
                        end_date=end_date,
                        chunk_type="abstract",
                        search_mode=search_mode,
                    )
                    best_by_paper = {}
                    for hit in hits:
                        previous = best_by_paper.get(hit.chunk.paper_id)
                        if previous is None or hit.score > previous["score"]:
                            best_by_paper[hit.chunk.paper_id] = {
                                "score": hit.score,
                                "chunk": hit.chunk,
                            }
                    st.session_state["search_results"] = [
                        (paper_id, data["score"])
                        for paper_id, data in sorted(
                            best_by_paper.items(), key=lambda item: item[1]["score"], reverse=True
                        )[:result_count]
                    ]
                    st.session_state["last_search_query"] = query.strip()
                    st.session_state["last_search_mode"] = search_mode
                except Exception as exc:
                    st.session_state["search_results"] = []
                    st.error(f"Search failed: {exc}")

    results = st.session_state.get("search_results", [])
    if not results:
        if st.session_state.get("last_search_query"):
            st.info("No matching papers found for this query and filter combination.")
        return

    st.subheader(f"Results for: {st.session_state.get('last_search_query', '')}")
    st.caption(f"Mode: {st.session_state.get('last_search_mode', 'hybrid')}")
    for paper_id, score in results:
        paper = papers.get(paper_id)
        if paper is None:
            continue
        with st.container(border=True):
            st.markdown(f"### {paper.title}")
            st.caption(
                f"{paper.paper_id} · {paper.published_date} · relevance {score:.3f}"
            )
            st.write(paper.abstract or "No abstract available.")
            col_open, col_links = st.columns([1, 5])
            with col_open:
                with st.container(key=f"discovery-open-{paper.paper_id.replace('.', '-')}"):
                    st.button(
                        "Open paper",
                        key=f"open-{paper.paper_id}",
                        on_click=_open_paper,
                        args=(paper.paper_id,),
                    )
            with col_links:
                if paper.pdf_url:
                    st.link_button(
                        "Download PDF", str(paper.pdf_url), key=f"pdf-{paper.paper_id}"
                    )
                elif paper.source_url:
                    st.link_button("Paper page", str(paper.source_url), key=f"url-{paper.paper_id}")


def _render_paper_workspace(papers: dict[str, Paper]) -> None:
    st.title("Paper workspace")
    deleted_title = st.session_state.pop("paper_deletion_notice", None)
    if deleted_title:
        _show_timed_success(f"Deleted {deleted_title}")

    paper_ids = list(papers)
    if not paper_ids:
        st.info("No indexed papers are available. Ingest papers and build the index first.")
        return

    if st.session_state.get("workspace_paper_id") not in papers:
        st.session_state["workspace_paper_id"] = paper_ids[0]
    paper_id = st.selectbox(
        "Select an indexed paper",
        options=paper_ids,
        format_func=lambda pid: f"{papers[pid].title} ({pid})",
        key="workspace_paper_id",
        on_change=_handle_paper_change,
    )
    if st.session_state.get("chat_paper_id") not in (None, paper_id):
        _clear_active_chat()
    st.session_state["chat_paper_id"] = paper_id
    paper = papers[paper_id]

    st.header(paper.title)
    st.caption(f"{paper.paper_id} · featured {paper.published_date}")
    if paper.authors:
        st.write("**Authors:** " + ", ".join(paper.authors))
    st.write(paper.abstract or "No abstract available.")
    actions = []
    if paper.source_url:
        actions.append(("page", str(paper.source_url)))
    if paper.pdf_url:
        actions.append(("pdf", str(paper.pdf_url)))
    actions.append(("delete", None))

    action_cols = st.columns(len(actions), gap="large")
    for index, (action, url) in enumerate(actions):
        with action_cols[index]:
            if action == "page":
                st.link_button("Paper page", url, type="secondary")
            elif action == "pdf":
                st.link_button("Download PDF", url, type="primary")
            else:
                with st.container(key="paper-workspace-delete"):
                    if st.button("Delete selected paper", key=f"delete-paper-{paper_id}"):
                        _confirm_delete_paper(paper_id, paper.title)

    st.divider()
    st.subheader("AI paper analysis")
    if st.button("Generate analysis", type="primary", key=f"analyze-{paper_id}"):
        with st.spinner("Analyzing retrieved evidence..."):
            try:
                from src.generation.analysis import analyze_paper

                st.session_state.setdefault("paper_analyses", {})[paper_id] = analyze_paper(paper_id)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    analysis: PaperAnalysis | None = st.session_state.get("paper_analyses", {}).get(paper_id)
    if analysis:
        st.caption(f"Generated by {analysis.provider} · {analysis.model}")
        st.markdown(analysis.analysis)
        _show_evidence(analysis.evidence_refs)

    st.divider()
    st.subheader("Ask follow-up questions")
    if st.button("Start a new chat", key=f"new-chat-{paper_id}"):
        from src.chat import start_chat_session

        _clear_active_chat()
        st.session_state["chat_session_id"] = start_chat_session(paper_id)
        st.session_state["chat_paper_id"] = paper_id
        st.session_state["chat_notice"] = "New chat started. Ask a question about this paper below."

    if st.session_state.get("chat_notice"):
        st.info(st.session_state["chat_notice"])

    for turn in st.session_state.get("chat_turns", []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                st.caption(f"Generated by {turn['provider']} · {turn['model']}")
                _show_evidence(turn["evidence_refs"])

    question = st.chat_input("Ask about the selected paper")
    if question:
        from src.chat import chat_with_paper

        st.session_state.pop("chat_notice", None)
        with st.spinner("Retrieving evidence and preparing an answer..."):
            try:
                response: PaperChatResponse = chat_with_paper(
                    paper_id,
                    question,
                    session_id=st.session_state.get("chat_session_id"),
                )
            except Exception as exc:
                st.error(f"Chat failed: {exc}")
            else:
                st.session_state["chat_session_id"] = response.session_id
                turns = st.session_state.setdefault("chat_turns", [])
                turns.extend([
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "provider": response.provider,
                        "model": response.model,
                        "evidence_refs": response.evidence_refs,
                    },
                ])
                st.rerun()


def _render_hf_ingestion() -> None:
    st.title("Find and index papers from Hugging Face")
    st.write(
        "Search Daily Papers by topic and date range, select up to five results, "
        "then download, parse, chunk, and index the selected papers."
    )
    st.caption(
        "Topic matching requires all entered terms to appear in the paper title or abstract. "
        "The paper ranking follows: most upvoted papers first, then newest papers to fill the list."
    )

    with st.form("hf_paper_search"):
        keyword = st.text_input("Topic keywords", placeholder="e.g. visual perception")
        date_cols = st.columns(2)
        today = date.today()
        with date_cols[0]:
            start_date = st.date_input("From", value=today - timedelta(days=30))
        with date_cols[1]:
            end_date = st.date_input(
                "Through (today)", value=today,
                key=f"hf_end_date_{today.isoformat()}", disabled=True,
            )
        ranking_label = st.radio(
            "Rank the five results by",
            options=["Most votes, then newest", "Latest"],
            horizontal=True,
        )
        search_submitted = st.form_submit_button("Search Hugging Face", type="primary")

    if search_submitted:
        if not keyword.strip():
            st.session_state.pop("hf_search_report", None)
            st.warning("Enter one or more topic keywords.")
        elif start_date > end_date:
            st.session_state.pop("hf_search_report", None)
            st.error("The start date must be on or before the end date.")
        else:
            with st.spinner("Fetching Daily Papers for the selected dates..."):
                try:
                    from src.ingestion.workflow import search_daily_papers

                    report = search_daily_papers(
                        start_date=start_date,
                        end_date=end_date,
                        keyword=keyword,
                        limit=5,
                        ranking="latest" if ranking_label == "Latest" else "votes",
                    )
                    st.session_state["hf_search_report"] = report
                    st.session_state["hf_selection_generation"] = (
                        st.session_state.get("hf_selection_generation", 0) + 1
                    )
                    _load_catalog.clear()
                except Exception as exc:
                    st.session_state.pop("hf_search_report", None)
                    st.error(f"Hugging Face search failed: {exc}")

    previous_status = st.session_state.pop("hf_ingestion_status", None)
    if previous_status:
        _show_timed_success(previous_status["message"])
        if previous_status["failures"]:
            st.warning("Some PDFs could not be parsed. Their abstracts were still eligible for indexing.")
            for paper_id, error in previous_status["failures"].items():
                st.caption(f"{paper_id}: {error}")

    report = st.session_state.get("hf_search_report")
    if report is None:
        return
    st.caption(
        f"Fetched {report.fetched_count} feed entries · "
        f"{report.matched_count} distinct papers matched · showing {len(report.papers)}"
    )
    if report.failed_dates:
        st.warning("Hugging Face could not be reached for: " + ", ".join(report.failed_dates))
    if not report.papers:
        st.info("No papers matched all topic terms in that date range.")
        return

    indexed_ids = set()
    try:
        from src.vectorstore import get_vectorstore

        indexed_ids = {chunk.paper_id for chunk in get_vectorstore().chunks}
    except (FileNotFoundError, RuntimeError):
        pass

    paper_ids = [paper.paper_id for paper in report.papers]
    generation = st.session_state.get("hf_selection_generation", 0)
    select_cols = st.columns([1, 1, 4])
    with select_cols[0]:
        if st.button("Select all results", key=f"select-all-{generation}"):
            for paper_id in paper_ids:
                st.session_state[f"hf-select-{generation}-{paper_id}"] = True
    with select_cols[1]:
        if st.button("Clear selection", key=f"clear-selection-{generation}"):
            for paper_id in paper_ids:
                st.session_state[f"hf-select-{generation}-{paper_id}"] = False

    st.subheader("Matching papers")
    selected_ids = []
    for paper in report.papers:
        state = "Already indexed" if paper.paper_id in indexed_ids else "Not indexed"
        with st.container(border=True):
            title_col, select_col = st.columns([9, 1])
            with title_col:
                st.markdown(f"### {paper.title}")
            with select_col:
                if st.checkbox(
                    "Select",
                    key=f"hf-select-{generation}-{paper.paper_id}",
                    help="Select this paper for downloading and indexing.",
                ):
                    selected_ids.append(paper.paper_id)
            st.caption(
                f"{paper.paper_id} · {paper.published_date} · "
                f"{paper.hf_votes} votes · {state}"
            )
            st.write(paper.abstract or "No abstract available.")
            if paper.pdf_url:
                st.link_button("PDF", str(paper.pdf_url), key=f"hf-pdf-{paper.paper_id}")

    if st.button(
        "Submit selection and build index",
        type="primary",
        disabled=not selected_ids,
        key=f"build-selected-{generation}",
    ):
        with st.spinner(
            f"Processing {len(selected_ids)} papers and building embeddings. "
            "PDF downloads may take a few seconds per paper..."
        ):
            try:
                from src.ingestion.workflow import ingest_selected_papers
                from src.indexing import build_index

                ingest_report = ingest_selected_papers(selected_ids)
                index_report = build_index(selected_ids)
                st.session_state["hf_ingestion_status"] = {
                    "message": (
                        f"Processed {len(selected_ids)} selected papers; "
                        f"added {index_report.added_chunks} chunks for "
                        f"{index_report.added_papers} new papers. "
                        f"The index now contains {index_report.total_papers} papers "
                        f"and {index_report.total_chunks} chunks."
                    ),
                    "failures": ingest_report.failures,
                }
                st.session_state["hf_selection_generation"] = generation + 1
                _load_catalog.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not ingest and index the selected papers: {exc}")


def main() -> None:
    st.sidebar.title("Research Paper Recommender")
    page = st.sidebar.radio(
        "Workspace",
        options=["Find papers on HF", "Paper workspace", "Discover"],
        key="active_page",
    )

    if st.sidebar.button("Reload paper index"):
        from src.vectorstore import clear_vectorstore_cache

        clear_vectorstore_cache()
        _load_catalog.clear()
        st.session_state.pop("search_results", None)
        st.rerun()

    if page == "Find papers on HF":
        _render_hf_ingestion()
        return

    try:
        catalog = _load_catalog()
        papers = _indexed_papers(catalog)
        st.sidebar.caption(f"{len(papers)} indexed papers")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.title("Research Paper Recommender")
        st.error(f"Could not load the paper index: {exc}")
        st.info("Build the index with `python scripts/build_index.py`, then reload this page.")
        return

    if page == "Discover":
        _render_discovery(papers)
    else:
        _render_paper_workspace(papers)


if __name__ == "__main__":
    main()
