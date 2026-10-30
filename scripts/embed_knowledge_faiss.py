"""Rebuilds the FAISS index from the knowledge base.

    python scripts/embed_knowledge_faiss.py                      # as before: KB -> faiss_db/
    python scripts/embed_knowledge_faiss.py --kb DIR --out DIR   # another KB or output folder
    python scripts/embed_knowledge_faiss.py --context --estimate # what a context build costs; calls nothing
    python scripts/embed_knowledge_faiss.py --context --sample growth-001,pay-002  # PAID: a few lines, no index
    python scripts/embed_knowledge_faiss.py --context --out DIR  # PAID: lines + embeddings, new index in DIR

--context (plan D.2, contextual retrieval): before embedding, a model writes a
short line placing each chunk in its file, and that line is embedded with the
chunk. It goes into the EMBEDDED text only. documents.pkl keeps the chunk text
exactly as before, so /search content, the source format and the prompt are
unchanged, and BM25 (built from documents.pkl) is unchanged too. The line is
kept in metadata['context'] so it can be read and audited.

Lines are cached on disk (CONTEXT_CACHE), keyed on model, prompt version, file
text and chunk text. A reindex only pays for chunks or files that changed.

Without --context nothing changes: same chunks, same text, same metadata.
"""
import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI

# Paths resolve against THIS FILE, not the working directory, same as
# api_server.py. The relative form wrote faiss_db/ into whatever folder the
# script was launched from. metadata['filename'] stays relative to the KB's
# parent folder (the repo root by default), byte-identical to what the
# relative form wrote.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAISS_DB = PROJECT_ROOT / "faiss_db"
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge-base"
# Outside faiss_db/ so a rebuilt or replaced index does not lose it. Gitignored:
# the lines are derived from the knowledge base, which is personal content.
CONTEXT_CACHE = PROJECT_ROOT / "context_cache" / "context_lines.json"

ENV_FILE = PROJECT_ROOT / ".env"

# Created on the first paid call, so --estimate and the free checks need no key.
_openai_client = None


def openai_client():
    global _openai_client
    if _openai_client is None:
        load_dotenv(ENV_FILE)
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

CHUNK_START = "<!--chunk:start-->"
CHUNK_END = "<!--chunk:end-->"
MAX_CHUNK_LENGTH = 6000  # Max chars per chunk for embedding

# Context lines (plan D.2). gpt-4o-mini: already in the stack for the follow-up
# rewrite, same key and SDK as the embeddings. Prices per 1M tokens, standard
# tier, checked at developers.openai.com/api/docs/pricing 2026-10-30.
CONTEXT_MODEL = "gpt-4o-mini"
CONTEXT_PRICE = (0.15, 0.60)
EMBED_PRICE = 0.10  # text-embedding-ada-002
CONTEXT_MAX_TOKENS = 120
# Bump when the prompt changes, so cached lines from the old prompt are not reused.
# ctx-v2 (2026-10-30): v1 sample wrote "innovative approach" for regimen-001, a
# word not in the chunk. Added the no-evaluative-adjectives rule.
CONTEXT_PROMPT_VERSION = "ctx-v2"
CONTEXT_PROMPT = """<document title="{title}">
{document}
</document>

Here is one chunk of that document:
<chunk>
{chunk}
</chunk>

Write one or two sentences, at most 50 words, that place this chunk within the document, so a search engine can match the chunk to questions about its topic. Say what the chunk is about and what part of the document it belongs to.

Rules:
- Add no facts. Do not state any number, amount, percentage, date, result, or name of a person, company or product unless it appears in the chunk itself.
- Do not judge or praise. Describe, do not sell. Use no evaluative adjectives (innovative, strong, successful, impressive, key) unless the chunk uses them.
- Answer with the sentences only."""


def get_embedding(text):
    """Get embedding using OpenAI directly"""
    response = openai_client().embeddings.create(
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


def build_chunks(knowledge_base_path):
    """Every chunk that goes into the index, in index order: a list of
    (text, metadata, file_text). No API calls."""
    out = []
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
            "filename": str(md_file.relative_to(knowledge_base_path.parent)),
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

            out.append((text, meta, content))

        print(f"  [{strategy}] {md_file.name}: {len(chunks)} chunks")
        file_count += 1

    print(f"{len(out)} chunks from {file_count} files ({skipped_count} skipped)")
    return out


# ─── Context lines (plan D.2) ───

def context_key(file_text, chunk_text):
    digest = hashlib.sha256()
    for part in (CONTEXT_MODEL, CONTEXT_PROMPT_VERSION, file_text, chunk_text):
        digest.update(part.encode("utf-8") + b"\0")
    return digest.hexdigest()


def context_prompt(meta, file_text, chunk_text):
    return CONTEXT_PROMPT.format(title=meta.get("title", ""), document=file_text, chunk=chunk_text)


def get_context_line(meta, file_text, chunk_text):
    """PAID: one gpt-4o-mini call. Returns (line, usage)."""
    response = openai_client().chat.completions.create(
        model=CONTEXT_MODEL,
        messages=[{"role": "user", "content": context_prompt(meta, file_text, chunk_text)}],
        temperature=0,
        max_tokens=CONTEXT_MAX_TOKENS,
        timeout=60,
    )
    line = " ".join(response.choices[0].message.content.split())
    return line, response.usage


def embedded_text(text, line):
    """The context line goes after the [category] [title] [chunk_id] prefix and
    before the chunk body. Only this string is embedded; documents.pkl keeps text."""
    if not line:
        return text
    head, sep, body = text.partition("\n\n")
    return f"{head}\n\n{line}\n\n{body}" if sep else f"{line}\n\n{text}"


# Words a line may use without them being in the chunk: sentence glue, and the
# subject of the whole knowledge base.
# Praise words a line must not add (ctx-v2 rule). Flagged when the chunk does not use them.
EVALUATIVE = {"innovative", "strong", "successful", "impressive", "key", "significant", "effective",
              "robust", "exceptional", "strategic", "critical", "valuable", "seamless", "cutting"}
FLAG_ALLOWED = {"this", "the", "it", "in", "a", "an", "jordanne", "jordanne's", "her", "she", "document", "chunk", "section"}


def unsupported_terms(line, text):
    """Numbers and capitalised words in a context line that do not appear in its
    chunk (case-insensitive). A check for added facts, not a proof: it cannot
    see a claim made in ordinary words. Every flagged line is printed for a person to read."""
    haystack = text.lower()
    flags = []
    for tok in re.findall(r"\$?\d[\d,.]*\s?[%MKBmkb+]*|\b[A-Z][\w'’.-]*", line):
        t = tok.strip().rstrip(".,")
        if not t or t.lower() in FLAG_ALLOWED:
            continue
        if t.lower() not in haystack:
            flags.append(t)
    for word in re.findall(r"[a-z]+", line.lower()):
        if word in EVALUATIVE and word not in haystack:
            flags.append(word)
    return flags


def load_cache(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def estimate(chunks, cache, use_context):
    """Prints what a build would call and roughly cost. Calls nothing. Tokens
    are counted as characters / 4, a rough English average, not a tokenizer."""
    embed_chars = sum(len(t) for t, _, _ in chunks)
    print(f"\n{len(chunks)} chunks, {embed_chars:,} characters of chunk text.")
    ctx_cost = 0.0
    if use_context:
        todo = [(t, m, f) for t, m, f in chunks if context_key(f, t) not in cache]
        in_chars = sum(len(context_prompt(m, f, t)) for t, m, f in todo)
        out_tokens = len(todo) * 70  # 50 words is about 70 tokens
        ctx_cost = in_chars / 4 / 1e6 * CONTEXT_PRICE[0] + out_tokens / 1e6 * CONTEXT_PRICE[1]
        print(f"Context lines: {len(todo)} to generate, {len(chunks) - len(todo)} cached. "
              f"{len(todo)} {CONTEXT_MODEL} calls, ~{in_chars // 4:,} input tokens (each call carries its whole file), "
              f"~{out_tokens:,} output tokens: ~${ctx_cost:.4f}.")
        embed_chars += len(chunks) * 300  # ~50 words per line
    embed_cost = embed_chars / 4 / 1e6 * EMBED_PRICE
    print(f"Embeddings: {len(chunks)} text-embedding-ada-002 calls, ~{embed_chars // 4:,} tokens: ~${embed_cost:.4f}.")
    print(f"Total ~${ctx_cost + embed_cost:.4f}. Nothing was called.")


def main():
    global ENV_FILE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kb", type=Path, default=KNOWLEDGE_BASE, help="knowledge-base folder (read only)")
    ap.add_argument("--out", type=Path, default=FAISS_DB, help="output folder for the three index files")
    ap.add_argument("--context", action="store_true", help="plan D.2: embed a context line with each chunk")
    ap.add_argument("--context-cache", type=Path, default=CONTEXT_CACHE)
    ap.add_argument("--estimate", action="store_true", help="print calls and cost, call nothing, write nothing")
    ap.add_argument("--sample", help="with --context: comma-separated chunk ids. PAID, a few calls. "
                                     "Generates and prints their lines, writes the cache, builds no index")
    ap.add_argument("--env", type=Path, default=ENV_FILE, help=".env holding OPENAI_API_KEY (default: the repo's)")
    args = ap.parse_args()
    ENV_FILE = args.env

    chunks = build_chunks(args.kb.resolve())
    cache = load_cache(args.context_cache) if args.context else {}

    if args.estimate:
        estimate(chunks, cache, args.context)
        return

    if args.sample:
        if not args.context:
            raise SystemExit("--sample needs --context")
        wanted = args.sample.split(",")
        chunks = [c for c in chunks if c[1]["chunk_id"] in wanted]
        missing = set(wanted) - {c[1]["chunk_id"] for c in chunks}
        if missing:
            raise SystemExit(f"not in the knowledge base: {sorted(missing)}")

    lines = []
    if args.context:
        spent_in = spent_out = 0
        for text, meta, file_text in chunks:
            key = context_key(file_text, text)
            if key not in cache:
                line, usage = get_context_line(meta, file_text, text)
                spent_in += usage.prompt_tokens
                spent_out += usage.completion_tokens
                cache[key] = {"chunk_id": meta["chunk_id"], "file": meta["filename"], "line": line,
                              "model": CONTEXT_MODEL, "prompt_version": CONTEXT_PROMPT_VERSION}
                save_cache(args.context_cache, cache)  # after every call, so a crash re-pays nothing
            lines.append(cache[key]["line"])
        cost = spent_in / 1e6 * CONTEXT_PRICE[0] + spent_out / 1e6 * CONTEXT_PRICE[1]
        print(f"\ncontext lines: {spent_in:,} input + {spent_out:,} output tokens this run, ${cost:.4f}")
        flagged = 0
        for (text, meta, _), line in zip(chunks, lines):
            flags = unsupported_terms(line, text)
            flagged += bool(flags)
            if args.sample or flags:
                print(f"\n[{meta['chunk_id']}] {line}")
                if flags:
                    print(f"   NOT IN THE CHUNK: {', '.join(flags)}")
        print(f"\n{flagged} of {len(lines)} lines carry a number or name not in their chunk.")
        if args.sample:
            print("Sample only: no embeddings, no index written.")
            return
    else:
        lines = [None] * len(chunks)

    documents, metadatas, embeddings_list = [], [], []
    for (text, meta, _), line in zip(chunks, lines):
        documents.append(text)
        embeddings_list.append(get_embedding(embedded_text(text, line)))
        metadatas.append({**meta, "context": line} if line else meta)

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings_list).astype('float32')

    # Create FAISS index
    dimension = len(embeddings_list[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # Save everything
    out = args.out
    os.makedirs(out, exist_ok=True)
    faiss.write_index(index, str(out / "resume.index"))

    with open(out / "documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    with open(out / "metadatas.pkl", "wb") as f:
        pickle.dump(metadatas, f)

    print(f"\nSuccessfully embedded {len(documents)} chunks")
    print(f"Index saved to {out}")


if __name__ == "__main__":
    main()
