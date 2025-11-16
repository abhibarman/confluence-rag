"""
Confluence RAG (with Persistence)
- Uses google.genai client to call Gemini (gemini-2.0-flash)
- Saves and loads FAISS indexes and scraped content
- Persistent storage across sessions
"""

import os
import re
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Try to import the genai client
genai = None
try:
    from google import genai as _genai
    genai = _genai
except Exception:
    try:
        import genai as _genai
        genai = _genai
    except Exception:
        genai = None

# Streamlit page config
st.set_page_config(page_title="Confluence RAG", page_icon="📚", layout="wide")
st.title("📚 Confluence RAG — Gemini (gemini-2.0-flash)")

# Create data directory for persistence
DATA_DIR = Path("confluence_indexes")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Embeddings ----------
@st.cache_resource
def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Return a HuggingFaceEmbeddings object cached by Streamlit.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

# Session state defaults
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "scraped_pages" not in st.session_state:
    st.session_state.scraped_pages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_stats" not in st.session_state:
    st.session_state.index_stats = {}
if "current_index_name" not in st.session_state:
    st.session_state.current_index_name = None

# ---------- Persistence Functions ----------
def get_available_indexes() -> List[Dict]:
    """Get list of saved indexes with metadata"""
    indexes = []
    if not DATA_DIR.exists():
        return indexes
    for index_dir in DATA_DIR.iterdir():
        if index_dir.is_dir():
            metadata_file = index_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    indexes.append({
                        "name": index_dir.name,
                        "metadata": metadata,
                        "path": index_dir
                    })
                except Exception:
                    # ignore corrupt metadata
                    pass
    return sorted(indexes, key=lambda x: x["metadata"].get("created_at", ""), reverse=True)

def save_index(
    index_name: str,
    vector_store: FAISS,
    pages: List[Dict],
    stats: Dict
) -> bool:
    """Save FAISS index, pages, and metadata with debug logging."""
    st.write("### 🐞 DEBUG: Starting save_index()")

    index_path = DATA_DIR / index_name
    st.write(f"DEBUG: Target index folder = `{index_path}`")

    try:
        index_path.mkdir(parents=True, exist_ok=True)
        st.write("DEBUG: Created/verified index folder.")
    except Exception as e:
        st.error(f"❌ ERROR creating index folder: {e}")
        return False

    # FAISS directory
    faiss_dir = index_path / "faiss_index"
    st.write(f"DEBUG: FAISS save folder = `{faiss_dir}`")

    try:
        faiss_dir.mkdir(parents=True, exist_ok=True)
        st.write("DEBUG: Created/verified faiss_index folder.")
    except Exception as e:
        st.error(f"❌ ERROR creating faiss_index folder: {e}")
        return False

    # ---- SAVE FAISS INDEX ----
    st.write("DEBUG: Calling vector_store.save_local() ...")
    try:
        # Save to the faiss_dir
        vector_store.save_local(str(faiss_dir))
        st.success("DEBUG: FAISS index saved successfully.")
        # List files saved
        saved_files = list(faiss_dir.glob("*"))
        st.write(f"DEBUG: Files saved inside faiss_index: {[str(f) for f in saved_files]}")
    except Exception as e:
        st.error(f"❌ ERROR saving FAISS index: {e}")
        return False

    # ---- SAVE pages.pkl ----
    try:
        pages_file = index_path / "pages.pkl"
        with open(pages_file, "wb") as f:
            pickle.dump(pages, f)
        st.success(f"DEBUG: pages.pkl saved at `{pages_file}`")
    except Exception as e:
        st.error(f"❌ ERROR saving pages.pkl: {e}")
        return False

    # ---- SAVE metadata.json ----
    try:
        metadata = {
            "created_at": datetime.now().isoformat(),
            "stats": stats,
            "page_count": len(pages),
        }
        metadata_file = index_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        st.success(f"DEBUG: metadata.json saved at `{metadata_file}`")
    except Exception as e:
        st.error(f"❌ ERROR saving metadata.json: {e}")
        return False

    st.success("### ✅ DEBUG: save_index() completed successfully!")
    return True

def load_index(index_name: str, embeddings):
    """Load FAISS index, pages, and metadata from disk"""
    
    index_path = DATA_DIR / index_name

    if not index_path.exists():
        return None, None, None

    try:
        # Load FAISS index
        vector_store = FAISS.load_local(
            str(index_path / "faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Load scraped pages
        with open(index_path / "pages.pkl", 'rb') as f:
            pages = pickle.load(f)

        # Load metadata
        with open(index_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        return vector_store, pages, metadata
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None, None, None

# def delete_index(index_name: str):
#     """Delete saved index"""
#     import shutil
#     index_path = DATA_DIR / index_name
#     if index_path.exists():
#         shutil.rmtree(index_path)
#         return True
#     return False

def delete_index(index_name: str):
    """Force-delete an index directory even if FAISS files are locked (Windows safe)."""
    import gc
    import shutil
    import stat

    index_path = DATA_DIR / index_name

    st.write(f"DEBUG: Attempting to delete {index_path}")

    # --------- RELEASE FAISS FROM MEMORY ---------
    try:
        st.write("DEBUG: Clearing vector_store from session")
        st.session_state.vector_store = None
        gc.collect()
        time.sleep(0.3)
    except Exception as e:
        st.write(f"DEBUG: Error clearing vector_store: {e}")

    if not index_path.exists():
        st.warning("DEBUG: Index folder does not exist")
        return True

    # --------- WINDOWS FORCE DELETE HANDLER ---------
    def on_rm_error(func, path, exc_info):
        """Handle read-only or locked files"""
        st.write(f"DEBUG: on_rm_error triggered for {path}")

        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception as e:
            st.write(f"DEBUG: chmod failed: {e}")

        try:
            func(path)
        except Exception as e:
            st.write(f"DEBUG: func retry failed: {e}")

    # --------- FORCE DELETE DIRECTORY ---------
    try:
        shutil.rmtree(index_path, onerror=on_rm_error)
        st.success(f"DEBUG: Deleted index folder {index_path}")
        return True
    except Exception as e:
        st.error(f"DEBUG: FINAL DELETE FAILURE → {e}")
        return False


# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    gemini_api_key = st.text_input("Gemini API Key", type="password")

    st.markdown("---")
    st.markdown("### 💾 Saved Indexes")

    available_indexes = get_available_indexes()

    if available_indexes:
        index_options = ["<Create New Index>"] + [idx["name"] for idx in available_indexes]
        selected_index = st.selectbox(
            "Select Index",
            index_options,
            help="Choose an existing index or create a new one"
        )

        if selected_index != "<Create New Index>":
            # Show index info
            idx_info = next(idx for idx in available_indexes if idx["name"] == selected_index)
            st.info(f"📅 Created: {idx_info['metadata']['created_at'][:10]}")
            st.info(f"📄 Pages: {idx_info['metadata']['page_count']}")

            col1, col2 = st.columns(2)
            with col1:
                load_btn = st.button("📂 Load", use_container_width=True)
            with col2:
                delete_btn = st.button("🗑️ Delete", use_container_width=True)

            if load_btn:
                with st.spinner("Loading index..."):
                    # 🔥 IMPORTANT: clear previous index from memory
                    st.session_state.vector_store = None
                    st.session_state.scraped_pages = []
                    st.session_state.index_stats = {}
                    st.session_state.current_index_name = None
                    
                    embeddings = get_embeddings()
                    vector_store, pages, metadata = load_index(selected_index, embeddings)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.scraped_pages = pages
                        st.session_state.index_stats = metadata.get("stats", {})
                        st.session_state.current_index_name = selected_index
                        st.success(f"✅ Loaded index: {selected_index}")
                        st.rerun()
                    else:
                        st.error("Failed to load index")

            if delete_btn:
                if delete_index(selected_index):
                    st.success(f"🗑️ Deleted index: {selected_index}")
                    if st.session_state.current_index_name == selected_index:
                        st.session_state.vector_store = None
                        st.session_state.scraped_pages = []
                        st.session_state.index_stats = {}
                        st.session_state.current_index_name = None
                    st.rerun()
                else:
                    st.error("Failed to delete index")
    else:
        st.info("No saved indexes yet")

    st.markdown("---")
    st.markdown("### 🔧 Confluence scraping settings")
    confluence_start_url = st.text_input("Start page URL", placeholder="https://your-domain.atlassian.net/wiki/...")
    confluence_username = st.text_input("Username (optional)")
    confluence_token = st.text_input("API Token (optional)", type="password")
    max_pages = st.slider("Max pages to scrape", 1, 200, 20)
    max_chars = st.number_input("Max chars per page", 2000, 300000, 100000, step=1000)

    st.markdown("**Scraping method**")
    use_api = st.checkbox("Use Confluence REST API (recommended)", value=True)

    st.markdown("**Retrieval settings**")
    top_k = st.slider("Top K documents to retrieve", 1, 10, 4)
    chunk_size = st.number_input("Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", 0, 500, 200, step=50)

    st.markdown("---")
    if st.session_state.current_index_name:
        st.success(f"📍 Current: {st.session_state.current_index_name}")

    if st.session_state.index_stats:
        st.markdown("**Index Statistics**")
        st.json(st.session_state.index_stats)

scrape_clicked = st.sidebar.button("🔍 Scrape & Index Confluence", use_container_width=True)

# ---------- Utilities ----------
def normalize_url(base: str, link: str) -> Optional[str]:
    try:
        return urljoin(base, link)
    except Exception:
        return None

def same_origin(a: str, b: str) -> bool:
    try:
        pa, pb = urlparse(a), urlparse(b)
        return pa.scheme == pb.scheme and pa.netloc == pb.netloc
    except Exception:
        return False

def build_requests_session(username: Optional[str] = None, token: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    if username and token:
        s.auth = (username, token)
    s.headers.update({"User-Agent": "ConfluenceRAG/1.0"})
    return s

# ---------- Confluence Scraper ----------
class ConfluenceScraper:
    def __init__(self, start_url: str, session: requests.Session,
                 max_pages: int = 50, max_chars: int = 100000, use_api: bool = True):
        self.start_url = start_url
        self.session = session
        self.to_visit = [start_url]
        self.visited = set()
        self.pages: List[Dict] = []
        self.max_pages = max_pages
        self.max_chars = max_chars
        self.use_api = use_api

        parsed = urlparse(start_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.is_cloud = "atlassian.net" in parsed.netloc

    def _get(self, url: str):
        r = self.session.get(url, timeout=12)
        r.raise_for_status()
        return r

    def _get_page_id_from_url(self, url: str) -> Optional[str]:
        patterns = [
            r'/pages/(\d+)',
            r'pageId=(\d+)',
            r'/wiki/spaces/[^/]+/pages/(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _fetch_via_api(self, page_id: str) -> Optional[Dict]:
        try:
            api_url = f"{self.base_url}/wiki/rest/api/content/{page_id}?expand=body.storage,space"
            r = self._get(api_url)
            data = r.json()

            title = data.get("title", "Untitled")
            body = data.get("body", {}).get("storage", {}).get("value", "")

            soup = BeautifulSoup(body, "html.parser")
            text = soup.get_text("\n", strip=True)

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            text = "\n".join(lines)

            if len(text) > self.max_chars:
                text = text[:self.max_chars] + "\n\n[...] (truncated)"

            space_key = data.get('space', {}).get('key', 'unknown')
            return {
                "url": f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}",
                "title": title,
                "content": text,
            }
        except Exception as e:
            st.warning(f"⚠️ API fetch failed for page {page_id}: {e}")
            return None

    def _extract_links(self, soup: BeautifulSoup, base: str) -> List[str]:
        out = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full = normalize_url(base, href)
            if full and same_origin(self.start_url, full):
                out.append(full)
        return out

    def _extract_text_from_html(self, soup: BeautifulSoup) -> str:
        content = None
        selectors = [
            {"id": "main-content"},
            {"class": "wiki-content"},
            {"class": "confluence-content"},
            {"id": "content"},
            {"class": "page-content"},
            {"role": "main"},
        ]

        for selector in selectors:
            content = soup.find("div", selector)
            if content:
                break

        if not content:
            content = soup.find("main") or soup.find("article") or soup.body

        if content:
            for bad in content.find_all(["script", "style", "nav", "footer", "header", "form", "button", "noscript"]):
                bad.decompose()
            text = content.get_text("\n", strip=True)
        else:
            text = soup.get_text("\n", strip=True)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)

        noise_patterns = ["requestCorrelationId", "serverDuration", "Skip to main content", "Navigation menu"]
        for pattern in noise_patterns:
            if pattern in text and len(text) < 500:
                text = text.replace(pattern, "")

        if len(text) > self.max_chars:
            text = text[:self.max_chars] + "\n\n[...] (truncated)"

        return text.strip()

    def scrape(self, progress_callback=None) -> List[Dict]:
        count = 0
        while self.to_visit and count < self.max_pages:
            url = self.to_visit.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            page_added = False

            if self.use_api:
                page_id = self._get_page_id_from_url(url)
                if page_id:
                    page_data = self._fetch_via_api(page_id)
                    if page_data and len(page_data["content"]) > 100:
                        self.pages.append(page_data)
                        count += 1
                        page_added = True
                        if progress_callback:
                            progress_callback(count, page_data["title"], page_data["url"], len(page_data["content"]))

            if not page_added:
                try:
                    r = self._get(url)
                    soup = BeautifulSoup(r.content, "html.parser")
                    title = soup.title.get_text(strip=True) if soup.title else url
                    text = self._extract_text_from_html(soup)

                    if text and len(text.strip()) > 100:
                        if not (text.strip().startswith('{') and text.strip().endswith('}') and len(text) < 500):
                            self.pages.append({"url": url, "title": title, "content": text})
                            count += 1
                            page_added = True
                            if progress_callback:
                                progress_callback(count, title, url, len(text))
                        else:
                            st.warning(f"⚠️ Skipping {url[:80]}... - Only JSON metadata")
                    else:
                        st.warning(f"⚠️ Skipping {url[:80]}... - Content too short")
                except Exception as e:
                    st.warning(f"⚠️ Error: {e}")

            try:
                if not page_added or not self.use_api:
                    r = self._get(url)
                    soup = BeautifulSoup(r.content, "html.parser")
                    links = self._extract_links(soup, url)
                    for l in links:
                        if l not in self.visited and l not in self.to_visit and len(self.to_visit) < self.max_pages:
                            self.to_visit.append(l)
            except:
                pass

            time.sleep(0.25)

        return self.pages

def build_faiss_from_documents(documents: List[Document], embeddings, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = splitter.split_documents(documents)

    st.info(f"✂️ Split {len(documents)} documents into {len(splits)} chunks")

    with st.expander("📋 View Sample Chunks (first 3)"):
        for i, split in enumerate(splits[:3]):
            st.markdown(f"**Chunk {i+1}** (length: {len(split.page_content)})")
            st.text(split.page_content[:500] + "..." if len(split.page_content) > 500 else split.page_content)
            st.markdown(f"*Metadata: {split.metadata}*")
            st.markdown("---")

    store = FAISS.from_documents(splits, embeddings)
    return store, splits

def rag_answer(question: str, vector_store: FAISS, client, model_name: str = "gemini-2.0-flash", k: int = 4):
    docs = vector_store.similarity_search(question, k=k)

    with st.expander(f"🔍 Retrieved {len(docs)} documents", expanded=False):
        for i, d in enumerate(docs):
            st.markdown(f"**Doc {i+1}: {d.metadata.get('title', 'Untitled')}**")
            st.markdown(f"URL: {d.metadata.get('url', 'N/A')}")
            st.markdown(f"Content length: {len(d.page_content)} chars")
            st.text_area(f"Content preview {i+1}", d.page_content[:500], height=150, key=f"doc_{i}_{time.time()}")
            st.markdown("---")

    if not docs:
        return "No relevant documents found in the index.", []

    context_pieces = []
    for d in docs:
        title = d.metadata.get("title", "Untitled")
        url = d.metadata.get("url", "")
        content = d.page_content
        context_pieces.append(f"Title: {title}\nURL: {url}\n\n{content}")

    context_text = "\n\n---\n\n".join(context_pieces)

    prompt = f"""You are a helpful assistant. Use the following context from Confluence documentation to answer the question.
If the answer is not in the context, say "I don't have enough information in the provided context to answer that question."

Context:
{context_text}

Question: {question}

Answer (be specific and cite relevant information from the context):"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text, docs
    except Exception as e:
        return f"[Gemini Error] {e}", docs

# ---------- UI: Scrape & Index ----------
if scrape_clicked:

    # Reset because we are building a NEW index
    st.session_state.current_index_name = None

    if not confluence_start_url:
        st.error("❌ Please provide a Confluence Start Page URL.")
    elif not gemini_api_key:
        st.error("❌ Provide Gemini API Key in the sidebar.")
    else:
        if genai is None:
            st.error("❌ genai client not available. Install: `pip install google-generativeai`")
        else:
            if use_api and not (confluence_username and confluence_token):
                st.warning("⚠️ API mode enabled but no credentials provided. Will fall back to HTML scraping.")

            progress_box = st.empty()

            def progress_cb(count, title, url, text_len):
                progress_box.info(f"📄 Scraped {count}/{max_pages} — {title[:50]}... ({text_len:,} chars)")

            session = build_requests_session(confluence_username, confluence_token)
            with st.spinner("🔄 Scraping pages..."):
                scraper = ConfluenceScraper(
                    confluence_start_url,
                    session,
                    max_pages=max_pages,
                    max_chars=max_chars,
                    use_api=use_api
                )
                pages = scraper.scrape(progress_callback=progress_cb)

            if not pages:
                st.error("❌ No pages scraped. Check URL/auth/permissions.")
            else:
                st.success(f"✅ Successfully scraped {len(pages)} pages")

                with st.expander("📄 View Scraped Pages", expanded=False):
                    for i, page in enumerate(pages[:5]):
                        st.markdown(f"**{i+1}. {page['title']}**")
                        st.caption(f"URL: {page['url']}")
                        st.caption(f"Content length: {len(page['content']):,} chars")
                        st.text_area(f"Content preview", page['content'][:500], height=100, key=f"page_{i}")
                        st.markdown("---")

                # persist into session_state so the Save UI (outside this block) can use it
                st.session_state.scraped_pages = pages

                with st.spinner("🔄 Creating embeddings and building FAISS index..."):
                    embeddings = get_embeddings()
                    documents = [
                        Document(
                            page_content=p["content"],
                            metadata={"title": p["title"], "url": p["url"]}
                        ) for p in pages
                    ]

                    vector_store, splits = build_faiss_from_documents(
                        documents,
                        embeddings,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    st.session_state.vector_store = vector_store

                    stats = {
                        "total_pages": len(pages),
                        "total_chunks": len(splits),
                        "total_chars": sum(len(p["content"]) for p in pages),
                        "avg_chars_per_page": sum(len(p["content"]) for p in pages) // len(pages) if pages else 0,
                    }
                    st.session_state.index_stats = stats

                st.success(f"✅ Index created with {len(splits)} chunks!")

# -----------------------------------------------------------
# Show Save Index only if a NEW index was created (not a loaded one)
# -----------------------------------------------------------
if (
    st.session_state.vector_store is not None
    and len(st.session_state.scraped_pages) > 0
    and st.session_state.current_index_name is None
):
    st.markdown("### 💾 Save Index")

    index_name = st.text_input(
        "Index name (new index)",
        value=f"confluence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        key="save_index_name"
    )

    if st.button("💾 Save Index", key="save_index_button"):
        st.write("DEBUG: Save button clicked")

        ok = save_index(
            index_name,
            st.session_state.vector_store,
            st.session_state.scraped_pages,
            st.session_state.index_stats
        )
        if ok:
            st.success(f"Saved index: {index_name}")
            st.session_state.current_index_name = index_name
            st.rerun()
        else:
            st.error("Failed to save index")

# ---------- UI: Chat / Q&A ----------
st.subheader("💬 Ask Questions")

if genai is None:
    st.warning("⚠️ Install: `pip install google-generativeai`")
    st.stop()

if not gemini_api_key:
    st.info("ℹ️ Set your Gemini API key in the sidebar.")
    st.stop()

try:
    client = genai.Client(api_key=gemini_api_key)
except Exception as e:
    st.error(f"❌ Failed to create genai client: {e}")
    st.stop()

if st.session_state.vector_store is None:
    st.info("ℹ️ Load an existing index or create a new one (use Scrape & Index button).")
    st.stop()

for msg in st.session_state.chat_history:
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(msg.get("content", ""))

user_q = st.chat_input("Ask a question about your Confluence pages...")
if user_q:
    st.session_state.chat_history.append({"role": "user", "content": user_q})

    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            answer_text, source_docs = rag_answer(
                user_q,
                st.session_state.vector_store,
                client,
                model_name="gemini-2.0-flash",
                k=top_k
            )
            st.markdown(answer_text)

            if source_docs:
                st.markdown("### 📚 Sources")
                for d in source_docs:
                    title = d.metadata.get("title", "Untitled")
                    url = d.metadata.get("url", "")
                    excerpt = d.page_content[:300].replace("\n", " ")
                    st.markdown(f"- **[{title}]({url})**")
                    st.caption(excerpt + ("..." if len(d.page_content) > 300 else ""))

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_text,
            })

st.markdown("---")
st.markdown("<small>Built with Streamlit • LangChain • Google Gemini</small>", unsafe_allow_html=True)
