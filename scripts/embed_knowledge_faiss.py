import os
import re
import pickle
import yaml
from pathlib import Path
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_START = "<!--chunk:start-->"
CHUNK_END = "<!--chunk:end-->"
MAX_CHUNK_LENGTH = 6000  # Max chars per chunk for embedding


def get_embedding(text):
    """Get embedding using OpenAI directly"""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        timeout=30
    )
    return response.data[0].embedding


def parse_file_frontmatter(content):
    """Extract YAML frontmatter from file if present."""
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return {}
    return {}


def parse_chunk_yaml(chunk_text):
    """Extract YAML metadata from a chunk's code fence."""
    match = re.search(r'```yaml\s*\n(.*?)```', chunk_text, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return {}
    return {}


def extract_semantic_chunks(content, file_meta):
    """Split content on <!--chunk:start--> / <!--chunk:end--> markers.
    Returns list of (text_for_embedding, metadata) tuples.
    """
    chunks = []

    # Split on chunk markers
    parts = content.split(CHUNK_START)

    for part in parts[1:]:  # Skip everything before first chunk marker
        if CHUNK_END in part:
            chunk_body = part.split(CHUNK_END)[0].strip()
        else:
            chunk_body = part.strip()

        if not chunk_body:
            continue

        # Extract per-chunk YAML metadata
        chunk_meta = parse_chunk_yaml(chunk_body)
        chunk_id = chunk_meta.get("chunk_id", "unknown")
        priority = chunk_meta.get("context_priority", "medium")
        weight = chunk_meta.get("embedding_weight", 1.0)

        # Strip the YAML code fence from the text to embed
        clean_text = re.sub(r'```yaml\s*\n.*?```\s*', '', chunk_body, flags=re.DOTALL).strip()

        if not clean_text:
            continue

        # Prepend metatag context for richer embeddings (RAG best practice)
        category = file_meta.get("category", "general")
        title = file_meta.get("title", "")
        context_prefix = f"[{category}] [{title}] [{chunk_id}]\n\n"
        text_for_embedding = context_prefix + clean_text

        # If still too long, split at paragraph boundaries
        if len(text_for_embedding) > MAX_CHUNK_LENGTH:
            sub_chunks = split_at_paragraphs(text_for_embedding, MAX_CHUNK_LENGTH)
            for idx, sub in enumerate(sub_chunks):
                chunks.append((sub, {
                    "chunk_id": f"{chunk_id}_part{idx}",
                    "context_priority": priority,
                    "embedding_weight": weight,
                    **file_meta
                }))
        else:
            chunks.append((text_for_embedding, {
                "chunk_id": chunk_id,
                "context_priority": priority,
                "embedding_weight": weight,
                **file_meta
            }))

    return chunks


def split_at_paragraphs(text, max_length):
    """Split text at paragraph boundaries (double newline) respecting max_length."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_length and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_file_without_markers(content, file_meta):
    """Fallback for files without chunk markers: split at headings or paragraphs."""
    category = file_meta.get("category", "general")
    title = file_meta.get("title", "")

    # Try splitting on ## headings first
    sections = re.split(r'\n(?=## )', content)
    chunks = []

    for idx, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        text = f"[{category}] [{title}]\n\n{section}"

        if len(text) > MAX_CHUNK_LENGTH:
            sub_chunks = split_at_paragraphs(text, MAX_CHUNK_LENGTH)
            for sub_idx, sub in enumerate(sub_chunks):
                chunks.append((sub, {**file_meta, "chunk_id": f"auto_{idx}_{sub_idx}"}))
        else:
            chunks.append((text, {**file_meta, "chunk_id": f"section_{idx}"}))

    return chunks if chunks else [(f"[{category}] [{title}]\n\n{content}", {**file_meta, "chunk_id": "full"})]


# ─── Main ───

knowledge_base_path = Path("knowledge-base")
documents = []
metadatas = []
embeddings_list = []

file_count = 0
skipped_count = 0

for md_file in sorted(knowledge_base_path.rglob("*.md")):
    # Skip index files
    if md_file.name == "PROJECT-FILES-INDEX":
        continue

    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.strip():
        print(f"  Skipped (empty): {md_file}")
        skipped_count += 1
        continue

    # Extract file-level metadata
    frontmatter = parse_file_frontmatter(content)
    file_meta = {
        "filename": str(md_file),
        "category": md_file.parent.name,
        "title": frontmatter.get("title", md_file.stem),
    }

    # Choose chunking strategy
    if CHUNK_START in content:
        chunks = extract_semantic_chunks(content, file_meta)
        strategy = "semantic"
    else:
        chunks = chunk_file_without_markers(content, file_meta)
        strategy = "auto"

    for text, meta in chunks:
        weight = meta.get("embedding_weight", 1.0)

        # Skip chunks with very low weight
        if weight < 0.3:
            continue

        documents.append(text)
        embeddings_list.append(get_embedding(text))
        metadatas.append(meta)

    print(f"  [{strategy}] {md_file.name}: {len(chunks)} chunks")
    file_count += 1

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings_list).astype('float32')

# Create FAISS index
dimension = len(embeddings_list[0])
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# Save everything
os.makedirs("faiss_db", exist_ok=True)
faiss.write_index(index, "faiss_db/resume.index")

with open("faiss_db/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open("faiss_db/metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print(f"\nSuccessfully embedded {len(documents)} chunks from {file_count} files ({skipped_count} skipped)")
print(f"Index saved to faiss_db/")
