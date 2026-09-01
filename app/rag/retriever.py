"""
Retrieval half of RAG. Deliberately simple: chunk policy docs by paragraph,
TF-IDF vectorize, retrieve by cosine similarity. No vector DB, no embedding
model download (this sandbox has no access to a model hub) - TF-IDF is a
transparent, dependency-light stand-in. Swap for a real embedding model +
vector store (e.g. sentence-transformers + FAISS/pgvector) in production;
the retrieval interface below (`retrieve(query, k)`) stays the same either way.
"""
import os
import glob
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "data", "external", "policies")


@dataclass
class Chunk:
    text: str
    source: str


class PolicyRetriever:
    def __init__(self, policy_dir: str = POLICY_DIR):
        self.chunks: list[Chunk] = []
        self._load(policy_dir)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks]) if self.chunks else None

    def _load(self, policy_dir: str):
        for path in sorted(glob.glob(os.path.join(policy_dir, "*.md"))):
            source = os.path.basename(path)
            with open(path) as f:
                content = f.read()
            # split on markdown headers/blank lines into paragraph-ish chunks
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
            for p in paragraphs:
                # skip header-only chunks (e.g. a lone "# Title" line) - no real content to retrieve
                is_header_only = all(line.strip().startswith("#") or not line.strip()
                                      for line in p.splitlines())
                if len(p) > 20 and not is_header_only:
                    self.chunks.append(Chunk(text=p, source=source))

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        if not self.chunks or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        # drop zero-similarity matches - "no relevant policy found" is a valid answer
        return [self.chunks[i] for i in top_idx if sims[i] > 0]


_retriever = None


def get_retriever() -> PolicyRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever
