# 📚 Confluence RAG with Gemini

A powerful Retrieval-Augmented Generation (RAG) application that scrapes Confluence pages, builds searchable indexes with persistent storage, and enables intelligent Q&A using Google's Gemini AI.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![LangChain](https://img.shields.io/badge/langchain-latest-green)

## ✨ Features

- **🔍 Smart Content Extraction**: Supports both Confluence REST API and HTML scraping
- **💾 Persistent Storage**: Save and load FAISS indexes across sessions
- **🤖 AI-Powered Q&A**: Uses Google Gemini 2.0 Flash for accurate, contextual answers
- **📊 Index Management**: Create, load, and delete multiple indexes
- **🎯 Source Attribution**: Every answer includes links to source Confluence pages
- **🔄 Flexible Crawling**: Automatically discovers and indexes linked pages
- **📈 Real-time Progress**: Live feedback during scraping and indexing
- **🐛 Enhanced Debugging**: View scraped content, chunks, and retrieval results
- **⚙️ Configurable Parameters**: Adjust chunk sizes, retrieval settings, and scraping limits

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Confluence workspace access
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/abhibarman/confluence-rag.git
cd confluence-rag
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser** to `http://localhost:8501`

## 📦 Requirements

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
requests>=2.31.0
beautifulsoup4>=4.12.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.1
langchain-text-splitters>=0.0.1
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
google-generativeai>=0.3.0
```

## 📖 Usage Guide

### 1. Initial Setup

**Configure API Key**:
- Enter your Gemini API key in the sidebar
- Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey)

**Confluence Settings**:

**Option A: Using REST API (Recommended)**

For **Confluence Cloud**:
1. Go to [Atlassian Account Settings](https://id.atlassian.com/manage-profile/security/api-tokens)
2. Click "Create API token"
3. Copy the token
4. In the app sidebar, enter:
   - **Username**: Your Atlassian email
   - **API Token**: The token you created
5. Enable "Use Confluence REST API" checkbox

For **Confluence Server/Data Center**:
1. Use your regular username and password
2. Or create a Personal Access Token in Confluence settings

**Option B: Using HTML Scraping**
- Disable "Use Confluence REST API" checkbox
- No credentials required for public pages
- May not work well with JavaScript-heavy pages

**Start Page URL**:
```
https://your-domain.atlassian.net/wiki/spaces/YOURSPACE/pages/123456/Your+Page+Title
```

### 2. Configuration Parameters

#### Scraping Settings

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| **Max pages to scrape** | Number of pages to index | 20 | 1-200 |
| **Max chars per page** | Truncate pages longer than this | 100,000 | 2,000-300,000 |
| **Use Confluence REST API** | Use API vs HTML scraping | ✅ Enabled | - |

#### Retrieval Settings

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| **Top K documents** | Number of chunks to retrieve | 4 | 1-10 |
| **Chunk size** | Characters per chunk | 1000 | 500-2000 |
| **Chunk overlap** | Overlap between chunks for context | 200 | 0-500 |

### 3. Creating & Saving Indexes

1. **Configure settings** in the sidebar
2. Click **🔍 Scrape & Index Confluence**
3. Monitor real-time progress as pages are scraped
4. Review scraped pages and chunks in expandable sections
5. **Save** the index with a descriptive name (e.g., `engineering_docs_2024`)
6. Index statistics will be displayed automatically

**Index Statistics Example**:
```json
{
  "total_pages": 15,
  "total_chunks": 128,
  "total_chars": 245680,
  "avg_chars_per_page": 16378
}
```

### 4. Loading Existing Indexes

1. View **💾 Saved Indexes** in the sidebar
2. Select an index from the dropdown
3. View creation date and page count
4. Click **📂 Load** to activate the index
5. The index is now ready for Q&A

### 5. Asking Questions

1. Type your question in the chat input
2. The system will:
   - Search the index for relevant documents
   - Retrieve top-k most similar chunks
   - Generate a contextual answer using Gemini
3. **View retrieved documents**: Click the expandable section to see:
   - Document title and URL
   - Content preview (first 500 chars)
   - Relevance to your query
4. **Check sources**: Every answer includes links to original Confluence pages

#### Example Questions

- "What is our deployment process?"
- "How do I configure the authentication system?"
- "What are the API rate limits?"
- "Summarize the architecture overview"
- "What security measures are in place?"

## 🗂️ Index Management

### Saved Index Structure
```
confluence_indexes/
└── your_index_name/
    ├── faiss_index/
    │   ├── index.faiss
    │   └── index.pkl
    ├── pages.pkl
    └── metadata.json
```

### Index Operations

- **Create**: Scrape new Confluence pages and build index
- **Save**: Persist index to disk with metadata
- **Load**: Restore a previously saved index
- **Delete**: Remove index from disk (Windows-safe handling)

**Guidelines**:
- **Chunks per page**: Should be 5-15 (varies by page length)
- **Avg chars per page**: 5,000-50,000 is typical
- **Too few chunks**: Content might be too short or not extracted properly
- **Too many chunks**: Consider reducing max_chars or chunk_size

## 🏗️ Technical Architecture

### System Components

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
    ┌────▼─────────────────────────┐
    │   Confluence Scraper         │
    │   (API or HTML)              │
    └────┬─────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │   Text Splitting             │
    │   (RecursiveCharacterSplitter)│
    └────┬─────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │   Embeddings Generation      │
    │   (HuggingFace Sentence      │
    │    Transformers)             │
    └────┬─────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │   Vector Store (FAISS)       │
    │   - Semantic Search          │
    │   - Persistent Storage       │
    └────┬─────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │   RAG Pipeline               │
    │   1. Query → Vector Search   │
    │   2. Retrieve Top-K Chunks   │
    │   3. Build Context           │
    │   4. Generate Answer (Gemini)│
    └──────────────────────────────┘
```

### How It Works

**1. Content Extraction**

**REST API Mode** (Recommended):
- Extracts page ID from URL
- Calls `/wiki/rest/api/content/{id}?expand=body.storage`
- Parses HTML storage format
- Returns clean text content

**HTML Scraping Mode** (Fallback):
- Fetches page HTML via HTTP
- Finds content div using multiple selectors
- Removes scripts, styles, navigation elements
- Extracts clean text

**2. Text Chunking**

Uses LangChain's `RecursiveCharacterTextSplitter`:
- Splits on paragraph boundaries (`\n\n`)
- Falls back to sentence boundaries (`. `)
- Maintains chunk size limits
- Preserves overlapping context between chunks

**3. Embeddings**

Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`:
- 384-dimensional embeddings
- Optimized for semantic similarity
- Runs locally (no API calls)
- Fast inference time (~80MB model download on first run)
- Cached by Streamlit for performance

**4. Vector Search**

FAISS (Facebook AI Similarity Search):
- Efficient similarity search at scale
- Cosine similarity metric
- Returns top-K most relevant chunks
- Low memory footprint
- Persistent storage with pickle serialization

**5. Answer Generation**

Google Gemini 2.0 Flash:
- Receives question + retrieved context
- Generates grounded, contextual answer
- Cites sources when possible
- Admits when answer isn't in context

### Data Flow

```
Confluence Pages → Scraper → Text Chunks → Embeddings → FAISS Index → Persistent Storage
                                                              ↓
User Question → Similarity Search → Top-K Docs → Context → Gemini → Answer + Sources
```

## 🐛 Troubleshooting

### Issue: "Only 83 characters of JSON metadata retrieved"

**Cause**: JavaScript-rendered Confluence page, HTML scraping only gets metadata

**Solution**: 
1. Enable "Use Confluence REST API" checkbox
2. Provide valid Confluence credentials
3. Ensure credentials have page read permissions

### Issue: "No pages scraped"

**Causes & Solutions**:
- **Wrong URL format**: Use full page URL, not space URL
- **Authentication required**: Provide username/API token
- **Permissions**: Ensure account can read the pages
- **Network issues**: Check firewall/proxy settings
- **Invalid credentials**: Verify username and API token

### Issue: "I don't have enough information to answer"

**Causes & Solutions**:
- **Not indexed**: Verify pages were scraped successfully
- **Wrong search**: Check "Retrieved documents" expander to see what was found
- **Too specific**: Try broader questions first
- **Increase Top K**: Try retrieving more chunks (6-8) in sidebar settings
- **Adjust chunk size**: Smaller chunks for specific facts, larger for context

### Issue: "Gemini API error"

**Causes & Solutions**:
- **Invalid API key**: Verify key at [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Rate limit exceeded**: Wait a few seconds and retry
- **Context too long**: Reduce chunk size or Top K in sidebar
- **Network issues**: Check internet connection

### Issue: "Failed to save/load index"

**Causes & Solutions**:
- **Disk space**: Ensure sufficient disk space available
- **Permissions**: Check write permissions in `confluence_indexes/` directory
- **Windows file locks**: App handles this automatically with force-delete on Windows
- **Corrupt index**: Delete and recreate the index

### Issue: Slow indexing

**Solutions**:
- Reduce "Max pages to scrape" to start (10-20 for testing)
- Reduce "Max chars per page" if pages are very long
- Use faster internet connection
- First run downloads the embedding model (~80MB)
- Check system resources (CPU/memory usage)

### Issue: Index won't delete on Windows

**Solution**: 
- Application automatically handles file locks
- Clears vector store from memory before deletion
- Uses Windows-safe error handling with `on_rm_error`
- If still fails, close and restart the application

## ⚙️ Advanced Tips

### For Better Results

**Chunk Size Optimization**:
- **Smaller (500-800)**: Better for specific facts, technical details, API parameters
- **Larger (1200-2000)**: Better for broader context, explanations, overviews

**Top K Tuning**:
- **Lower (2-3)**: Faster responses, more focused answers
- **Higher (6-10)**: More comprehensive answers, better for complex queries

**Max Pages Strategy**:
- Start small (10-20) for testing and validation
- Increase gradually for comprehensive coverage
- Note: More pages = longer indexing time but better coverage

**API vs HTML Mode**:
- Always prefer REST API when credentials are available
- API provides cleaner content with better structure
- HTML mode as fallback for public pages only

### Performance Optimization

1. **First-time Setup**: 
   - Allow time for embedding model download (~80MB)
   - Model is cached for subsequent runs

2. **Large Indexes**:
   - Consider splitting into multiple smaller indexes by space/topic
   - Each index can be loaded independently

3. **Query Performance**:
   - Lower Top K values improve response time
   - Smaller chunk sizes reduce context processing time

## 🔒 Security & Privacy

### Data Storage

- **Local processing**: All data processed and stored locally
- **Persistent indexes**: Saved in `confluence_indexes/` directory
- **No cloud sync**: Indexes remain on your machine
- **Session management**: Chat history cleared when session ends

### API Keys

- **Client-side only**: Keys stored in session state
- **Not persisted**: Keys never saved to disk
- **Secure transmission**: Keys only sent to Google Gemini API via HTTPS
- **User responsibility**: Keep your API keys secure, rotate regularly

### Confluence Access

- **Read-only operations**: App only reads content, never writes
- **Credential security**: Uses HTTPS, credentials in session only
- **Minimum permissions**: Only needs page read access
- **API tokens**: Prefer API tokens over passwords (more secure, revocable)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

-  Add support for attachments (PDFs, Word docs)
-  Implement incremental indexing (only new/updated pages)
-  Add support for more LLMs (Claude, GPT-4, local models)
-  Multi-space indexing with filters
-  Export conversations to PDF/Markdown
-  Advanced filtering (by space, labels, date ranges)
-  Automatic index refresh/update scheduling
-  Support for Confluence comments and inline comments
-  Multi-language support for content

## 📝 License

This project is open source.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [LangChain](https://python.langchain.com/) - RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [HuggingFace](https://huggingface.co/) - Embeddings
- [Google Gemini](https://ai.google.dev/) - LLM
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing

## 📧 Support

For issues, questions, or suggestions:

1. **Check Documentation**: Review this README and the [Troubleshooting](#-troubleshooting) section
2. **Review Issues**: Check existing GitHub issues for similar problems
3. **Open New Issue**: Include:
   - Clear description of the problem
   - Steps to reproduce
   - Screenshots (if applicable)
   - Index statistics from sidebar
   - Error messages from Streamlit logs
4. **System Info**: Mention OS, Python version, and relevant package versions

---

**Built with** ❤️ **using Streamlit • LangChain • Google Gemini**

**Made for better documentation search and knowledge management**