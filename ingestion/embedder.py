"""
ingestion/embedder.py

Decides what gets embedded, builds the text for each entity, and runs
the embedding model.

Two responsibilities live here:

  1. Filtering — should_embed(fn) decides whether a FunctionNode is worth
     embedding at all. Functions that are too short, private, and undocumented
     add noise to similarity results without contributing retrieval value.

  2. Text construction — build_embed_text(fn) and build_class_embed_text(cls)
     construct the string that actually gets embedded. The quality of this
     text directly determines retrieval quality — more context is better.

  3. Embedding — Embedder loads the sentence-transformers model once and
     exposes a batched embed() method. Uses all-MiniLM-L6-v2 by default
     (fast, small, good general-purpose quality). OpenAI can be substituted
     via config if needed.

Embedding strategy:
  - Functions: embed if they pass should_embed(). Text = signature + docstring + body.
  - Classes: embed only if they have a docstring. Text = signature line + docstring.
  - Modules, packages, attributes: not embedded, graph only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import Config
from .models import ClassNode, EmbedDoc, FunctionNode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding filter
# ---------------------------------------------------------------------------

def should_embed(fn: FunctionNode) -> bool:
    """
    Return True if this function is worth embedding in the vector store.

    Skip if ALL THREE of these are true — each alone is acceptable,
    but together they signal a trivial helper with no retrieval value:
      - no docstring        (undocumented)
      - fewer than 5 lines  (too short to be semantically meaningful)
      - private name        (name starts with _ — internal implementation detail)

    Examples that get skipped:
        def _reset(self): self.x = None          # private, 1 line, no docstring
        def __repr__(self): return f"<{self.name}>"  # dunder, 1 line, no docstring

    Examples that get embedded:
        def process_payment(...)                 # public, has docstring
        def _validate_internal(self, ...):       # private but >5 lines OR has docstring
        def __init__(self, client, timeout):     # dunder but complex enough
    """
    line_count  = (fn.line_end - fn.line_start) + 1
    is_private  = fn.name.startswith("_")
    has_docstring = bool(fn.docstring)

    if not has_docstring and line_count < 5 and is_private:
        return False

    return True


# ---------------------------------------------------------------------------
# Text construction — functions
# ---------------------------------------------------------------------------

def build_embed_text(fn: FunctionNode) -> str:
    """
    Build the text that represents a function in the vector store.

    full_body already contains the complete function source — the def line,
    docstring, and body exactly as written in the source file. No reconstruction
    needed; just use it directly.
    """
    return fn.full_body


# ---------------------------------------------------------------------------
# Text construction — classes
# ---------------------------------------------------------------------------

def build_class_embed_text(cls: ClassNode) -> str:
    """
    Build the text that represents a class in the vector store.

    Only the signature line + docstring — no method bodies. We don't have
    the full class source, only the inventory (method_names, attribute_names).
    A synthetic summary of those would be shallow; better to embed just the
    docstring which is the genuine human-written description of what the class does.

    Example output:
        class PaymentProcessor(BaseProcessor):
            \"\"\"Handles payment processing for Stripe. Manages retries,
            idempotency keys, and webhook verification.\"\"\"

    Classes without a docstring are not embedded — callers should check
    should_embed_class() before calling this.
    """
    bases = f"({', '.join(cls.base_classes)})" if cls.base_classes else ""
    signature = f"class {cls.name}{bases}:"

    parts = [signature]
    if cls.docstring:
        doc = cls.docstring.strip()
        if '\n' in doc:
            indented = '\n'.join(f'    {line}' for line in doc.splitlines())
            parts.append(f'    """\n{indented}\n    """')
        else:
            parts.append(f'    """{doc}"""')

    return '\n'.join(parts)


def should_embed_class(cls: ClassNode) -> bool:
    """Only embed classes that have a docstring — synthetic summaries add noise."""
    return bool(cls.docstring)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """
    Loads a sentence-transformers model and embeds EmbedDocs in batches.

    The model is loaded once on construction and reused for all embed() calls.
    Loading takes a few seconds the first time (model download on first run,
    then cached in ~/.cache/huggingface/).

    Default model: all-MiniLM-L6-v2
      - 384 dimensions
      - ~80MB on disk
      - Fast on CPU, no GPU required
      - Good general-purpose semantic similarity

    To use OpenAI instead, set EMBED_MODEL=text-embedding-3-large in your
    environment — the embedder will switch to the OpenAI API automatically.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg        = cfg
        self._model     = None   # lazy-loaded on first embed() call
        self._use_openai = cfg.embed_model.startswith("text-embedding")

    def _load_model(self) -> None:
        """Load the model on first use."""
        if self._use_openai:
            # OpenAI client is lightweight — nothing to preload.
            logger.info("Using OpenAI embeddings: %s", self.cfg.embed_model)
        else:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.cfg.embed_model)
            self._model = SentenceTransformer(self.cfg.embed_model)
            logger.info("Model loaded.")

    def embed(self, docs: list[EmbedDoc]) -> list[list[float]]:
        """
        Embed a list of EmbedDocs and return a parallel list of float vectors.

        The returned list is the same length as docs — docs[i] corresponds to
        embeddings[i]. The caller (VectorWriter) uses this alignment to upsert
        into Chroma.

        Batching: processes cfg.embed_batch_size documents per API/model call
        to avoid memory spikes on large repos.
        """
        if not docs:
            return []

        if self._model is None and not self._use_openai:
            self._load_model()

        texts = [doc.text for doc in docs]

        if self._use_openai:
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """
        Embed using the local sentence-transformers model.
        encode() handles batching internally — batch_size controls memory usage.
        """
        if self._model is None:
            self._load_model()

        vectors = self._model.encode(
            texts,
            batch_size=self.cfg.embed_batch_size,
            show_progress_bar=len(texts) > 100,  # only show bar for large jobs
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """
        Embed using the OpenAI embeddings API, batched at cfg.embed_batch_size.
        Requires OPENAI_API_KEY in the environment.
        """
        from openai import OpenAI
        client  = OpenAI(api_key=self.cfg.openai_api_key)
        results = []

        for i in range(0, len(texts), self.cfg.embed_batch_size):
            batch    = texts[i : i + self.cfg.embed_batch_size]
            response = client.embeddings.create(model=self.cfg.embed_model, input=batch)
            results.extend([item.embedding for item in response.data])
            logger.debug("OpenAI embeddings: %d/%d", min(i + self.cfg.embed_batch_size, len(texts)), len(texts))

        return results


# ---------------------------------------------------------------------------
# EmbedDoc factories
# ---------------------------------------------------------------------------

def make_function_embed_doc(fn: FunctionNode) -> EmbedDoc:
    """Build an EmbedDoc for a function. Call only after should_embed() returns True."""
    return EmbedDoc(
        uuid=fn.uuid,
        text=build_embed_text(fn),
        entity_type="function",
        metadata={
            "name":           fn.name,
            "qualified_name": fn.qualified_name,
            "file_path":      fn.file_path,
            "line_start":     fn.line_start,
            "line_end":       fn.line_end,
            "language":       fn.language,
            "repo_id":        fn.repo_id,
            "is_method":      fn.is_method,
            "is_async":       fn.is_async,
        },
    )


def make_class_embed_doc(cls: ClassNode) -> EmbedDoc:
    """Build an EmbedDoc for a class. Call only after should_embed_class() returns True."""
    return EmbedDoc(
        uuid=cls.uuid,
        text=build_class_embed_text(cls),
        entity_type="class",
        metadata={
            "name":           cls.name,
            "qualified_name": cls.qualified_name,
            "file_path":      cls.file_path,
            "line_start":     cls.line_start,
            "line_end":       cls.line_end,
            "language":       cls.language,
            "repo_id":        cls.repo_id,
            "is_abstract":    cls.is_abstract,
            "is_protocol":    cls.is_protocol,
        },
    )
