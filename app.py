import streamlit as st
import os
import logging
import traceback
import config
from rag.observability import get_langfuse, run_ragas_background

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

if USE_MONGODB:
    from rag.router import route_retrieve_rerank
    from rag import generator
else:
    import vertexai
    from vertexai.preview import rag
    from langchain_google_vertexai import ChatVertexAI
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from typing import List, Any


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
            with st.spinner("Routing query and retrieving information..."):
                try:
                    source_docs, router_result = route_retrieve_rerank(prompt, trace=lf_trace)
                    logger.info(f"Router: function={router_result.function}, mode={router_result.retrieval_mode}, courses={router_result.course_ids}, docs={len(source_docs)}")
                except Exception as e:
                    logger.error(f"Retrieval failed: {traceback.format_exc()}")
                    st.error(f"Retrieval failed: {e}")

            answer = ""
            answer_placeholder = st.empty()
            try:
                logger.info("Starting generation (streaming)...")
                stream = generator.generate_stream(
                    results=source_docs,
                    question=prompt,
                    function=router_result.function if router_result else "semantic_general",
                    course_ids=router_result.course_ids if router_result else None,
                    semantic_type=router_result.semantic_type if router_result else None,
                    trace=lf_trace,
                )
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

            # Close the parent trace, flush all buffered spans, trigger RAGAS
            if lf_trace is not None:
                try:
                    lf_trace.update(output=answer)
                except Exception:
                    pass
                try:
                    lf.flush()
                except Exception:
                    pass
                if answer and source_docs:
                    contexts = [
                        doc.get("content") or doc.get("policy_name", "")
                        for doc in source_docs
                        if doc.get("content") or doc.get("policy_name")
                    ]
                    if contexts:
                        run_ragas_background(prompt, contexts, answer, lf_trace.id)

            if source_docs:
                with st.expander("View Source Documents", expanded=False):
                    if router_result:
                        mode_label = router_result.retrieval_mode
                        sem = f" | Semantic: {router_result.semantic_type}" if router_result.semantic_type else ""
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
