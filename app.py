import logging
import os
import traceback

import streamlit as st

import config
from rag.observability import get_langfuse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tamubot")

st.set_page_config(
    page_title="TamuBot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 TamuBot — Texas A&M Academic Assistant")
st.markdown("Ask questions about courses, syllabi, degree requirements, and university policies.")

if config.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = f"TamuBot-{config.APP_MODE}"

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

USE_MONGODB = config.RETRIEVAL_BACKEND == "mongodb"

_session_manager = None

if USE_MONGODB:
    from rag import generator  # keep for format_context_xml fallback
    from rag.pipeline import generator_order, run_pipeline
    from rag.search_v3 import get_syllabus_urls
    if config.USE_V4_PIPELINE:
        from rag.v4.pipeline_v4 import run_pipeline_v4 as run_pipeline  # noqa: F811
        from rag.v4.pipeline_v4 import run_pipeline_v4_with_memory
        from rag.v4.session import SessionManager
        _session_manager = SessionManager()
else:
    from typing import Any, List

    import vertexai
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_google_vertexai import ChatVertexAI
    from vertexai.preview import rag


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    st.write(f"**Backend:** {'MongoDB Atlas' if USE_MONGODB else 'Vertex AI RAG'}")
    st.write(f"**Model:** {config.MODEL_NAME}")

    if USE_MONGODB:
        st.info("Using MongoDB Atlas hybrid search + Voyage AI reranking")
    else:
        st.write(f"**Project ID:** {config.PROJECT_ID}")
        st.write(f"**RAG Region:** {config.RETRIEVAL_REGION}")
        st.write(f"**LLM Region:** {config.GENERATION_REGION}")
        st.info("Using Vertex AI Managed RAG Service")


# ---------------------------------------------------------------------------
# Vertex AI legacy path (SYSTEM_PROMPT used only by Vertex)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are TamuBot, an academic assistant for Texas A&M University.
Answer the question based only on the following context. If the context does not contain
enough information, say so clearly rather than guessing.

Context:
{context}

Question: {question}
"""

if not USE_MONGODB:
    class VertexRagRetriever(BaseRetriever):
        project_id: str
        location: str
        rag_corpus_resource_name: str

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
            try:
                vertexai.init(project=self.project_id, location=self.location)
                response = rag.retrieval_query(
                    rag_resources=[rag.RagResource(rag_corpus=self.rag_corpus_resource_name)],
                    text=query,
                    similarity_top_k=5
                )
                documents = []
                if hasattr(response, 'contexts') and hasattr(response.contexts, 'contexts'):
                    for context in response.contexts.contexts:
                        documents.append(Document(
                            page_content=context.text,
                            metadata={"source": context.source_uri, "score": context.score}
                        ))
                return documents
            except Exception as e:
                st.error(f"Error retrieving documents: {e}")
                return []

    @st.cache_resource
    def get_rag_chain():
        try:
            llm = ChatVertexAI(
                model=config.MODEL_NAME,
                temperature=0.2,
                project=config.PROJECT_ID,
                location=config.GENERATION_REGION
            )
            retriever = VertexRagRetriever(
                project_id=config.PROJECT_ID,
                location=config.RETRIEVAL_REGION,
                rag_corpus_resource_name=config.RAG_CORPUS_RESOURCE_NAME
            )
            template = SYSTEM_PROMPT
            prompt = ChatPromptTemplate.from_template(template)
            generation_chain = prompt | llm
            return generation_chain, retriever
        except Exception as e:
            st.error(f"Error initializing RAG Chain: {e}")
            return None, None

    rag_chain, retriever = get_rag_chain()


# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about courses, syllabi, or degree requirements..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if USE_MONGODB:
            # --- MongoDB 3-stage pipeline: Route → Retrieve+Rerank → Generate ---

            # Create a parent Langfuse trace for this request
            lf = get_langfuse()
            lf_trace = None
            if lf is not None:
                try:
                    lf_trace = lf.trace(
                        name="TamuBot_Complete_Pipeline",
                        input=prompt,
                        metadata={"session_id": str(id(st.session_state))},
                    )
                except Exception:
                    lf_trace = None

            source_docs = []
            router_result = None
            data_gaps: list = []
            data_integrity = True
            conflicted_ids: list = []
            with st.spinner("Routing query and retrieving information..."):
                try:
                    if config.USE_V4_PIPELINE and _session_manager is not None:
                        thread_config = _session_manager.get_thread_config(str(id(st.session_state)))
                        result = run_pipeline_v4_with_memory(prompt, trace=lf_trace, thread_config=thread_config)
                    else:
                        result = run_pipeline(prompt, trace=lf_trace)
                    source_docs, router_result, data_gaps, data_integrity, conflicted_ids = result
                    logger.info(f"Router: function={router_result.function}, mode={router_result.retrieval_mode}, courses={router_result.course_ids}, docs={len(source_docs)}")
                except Exception as e:
                    logger.error(f"Retrieval failed: {traceback.format_exc()}")
                    st.error(f"Retrieval failed: {e}")

            answer = ""
            answer_placeholder = st.empty()
            try:
                logger.info("Starting generation (streaming)...")
                stream = generator_order(
                    recurrent=False,
                    chunks=source_docs,
                    query=prompt,
                    router_result=router_result,
                    data_gaps=data_gaps,
                    data_integrity=data_integrity,
                    conflicted_course_ids=conflicted_ids,
                    trace=lf_trace,
                ) if router_result is not None else iter([])
                for token in stream:
                    answer += token
                    answer_placeholder.markdown(answer + "▌")
                answer_placeholder.markdown(answer)
                logger.info(f"Generation complete, answer length: {len(answer)}")
            except Exception as e:
                logger.error(f"Generation failed: {traceback.format_exc()}")
                st.error(f"Generation failed: {e}")
                if source_docs:
                    context_xml = generator.format_context_xml(source_docs)
                    answer = "**Relevant documents found:**\n\n" + context_xml
                else:
                    answer = "No relevant information found in the knowledge base."
                answer_placeholder.markdown(answer)

            # Render syllabus links for all retrieved courses
            if source_docs:
                course_ids = list({doc["course_id"] for doc in source_docs if doc.get("course_id")})
                try:
                    url_map = get_syllabus_urls(course_ids)
                except Exception:
                    url_map = {}
                if url_map:
                    links = "  ".join(
                        f"[{cid} Syllabus]({url})" for cid, url in sorted(url_map.items())
                    )
                    answer_placeholder.markdown(answer + "\n\n---\n**Syllabi:** " + links)
                    answer += "\n\n---\n**Syllabi:** " + links

            # Close the parent trace and flush all buffered spans
            if lf_trace is not None:
                try:
                    lf_trace.update(output=answer)
                except Exception:
                    pass
                try:
                    lf.flush()
                except Exception:
                    pass

            if source_docs:
                with st.expander("View Source Documents", expanded=False):
                    if router_result:
                        mode_label = router_result.retrieval_mode
                        sem = f" | Intent: {router_result.intent_type}" if router_result.intent_type else ""
                        st.caption(
                            f"Function: **{router_result.function}** | "
                            f"Mode: {mode_label} | "
                            f"CatConf: {router_result.category_confidence:.2f}{sem} | "
                            f"Courses: {', '.join(router_result.course_ids) or 'none'}"
                        )
                    for i, doc in enumerate(source_docs):
                        label = doc.get("course_id", doc.get("policy_name", "Unknown"))
                        st.write(f"**Source {i+1}:** {label}")
                        if doc.get("category"):
                            st.write(f"*Category: {doc['category']}*")
                        content = doc.get("content", doc.get("policy_name", ""))
                        st.info(content[:500] + ("..." if len(content) > 500 else ""))
                        st.write("---")

            st.session_state.messages.append({"role": "assistant", "content": answer})

        else:
            # --- Vertex AI legacy path ---
            source_docs = []
            with st.spinner("Retrieving information..."):
                try:
                    source_docs = retriever.invoke(prompt)
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")

            answer = ""
            if rag_chain and source_docs:
                with st.spinner("Generating answer..."):
                    try:
                        vertexai.init(project=config.PROJECT_ID, location=config.GENERATION_REGION)
                        response = rag_chain.invoke({
                            "context": "\n\n".join([d.page_content for d in source_docs]),
                            "question": prompt
                        })
                        answer = response.content
                    except Exception as e:
                        st.warning("Generative model unavailable. Showing retrieved documents.")
                        st.caption(f"Error: {str(e)[:100]}...")
                        answer = "**Relevant documents found:**"
            elif not source_docs:
                answer = "No relevant information found in the knowledge base."

            st.markdown(answer)

            if source_docs:
                with st.expander("View Source Documents", expanded=False):
                    for i, doc in enumerate(source_docs):
                        st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"*Score: {doc.metadata.get('score', 'N/A')}*")
                        st.info(doc.page_content)
                        st.write("---")

            st.session_state.messages.append({"role": "assistant", "content": answer})
