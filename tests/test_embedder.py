"""
tests/test_embedder.py

Unit tests for ingestion/embedder.py.

Tests cover:
  - should_embed() filtering logic
  - should_embed_class() filtering logic
  - build_embed_text() uses full_body directly
  - build_class_embed_text() constructs correct text from ClassNode fields
  - make_function_embed_doc() produces correct EmbedDoc
  - make_class_embed_doc() produces correct EmbedDoc
"""

import pytest
from ingestion.embedder import (
    build_class_embed_text,
    build_embed_text,
    make_class_embed_doc,
    make_function_embed_doc,
    should_embed,
    should_embed_class,
)
from ingestion.models import ClassNode, FunctionNode, make_uuid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_fn(**overrides) -> FunctionNode:
    """Build a FunctionNode with sensible defaults, overridable per test."""
    defaults = dict(
        uuid=make_uuid("test", "a.py", "a.fn"),
        name="process_payment",
        qualified_name="payments.core.process_payment",
        file_path="payments/core.py",
        line_start=1,
        line_end=10,
        language="python",
        signature="def process_payment(amount: float) -> bool:",
        docstring="Process a payment via Stripe.",
        return_type="bool",
        is_async=False,
        is_method=False,
        is_property=False,
        is_classmethod=False,
        is_staticmethod=False,
        is_overload=False,
        decorators=[],
        body_preview="def process_payment",
        full_body="def process_payment(amount: float) -> bool:\n    \"\"\"Process a payment via Stripe.\"\"\"\n    return charge(amount).ok",
        complexity=2,
        repo_id="test",
    )
    return FunctionNode(**{**defaults, **overrides})


def make_cls(**overrides) -> ClassNode:
    """Build a ClassNode with sensible defaults, overridable per test."""
    defaults = dict(
        uuid=make_uuid("test", "a.py", "a.PaymentProcessor"),
        name="PaymentProcessor",
        qualified_name="payments.core.PaymentProcessor",
        file_path="payments/core.py",
        line_start=1,
        line_end=20,
        language="python",
        docstring="Handles payment processing for Stripe.",
        base_classes=["BaseProcessor"],
        decorators=[],
        is_abstract=False,
        is_protocol=False,
        is_dataclass=False,
        is_exception=False,
        method_names=["__init__", "process"],
        attribute_names=["self.client"],
        repo_id="test",
    )
    return ClassNode(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# should_embed — functions
# ---------------------------------------------------------------------------

class TestShouldEmbed:

    def test_public_function_always_embeds(self):
        fn = make_fn(name="process_payment", docstring=None, line_end=2)
        assert should_embed(fn) is True

    def test_private_short_no_docstring_is_skipped(self):
        # all three conditions met — should skip
        fn = make_fn(name="_reset", docstring=None, line_start=1, line_end=2)
        assert should_embed(fn) is False

    def test_private_with_docstring_embeds(self):
        # private + short but HAS docstring — should embed
        fn = make_fn(name="_reset", docstring="Resets internal state.", line_start=1, line_end=2)
        assert should_embed(fn) is True

    def test_private_long_no_docstring_embeds(self):
        # private + no docstring but long enough — should embed
        fn = make_fn(name="_validate", docstring=None, line_start=1, line_end=10)
        assert should_embed(fn) is True

    def test_dunder_short_no_docstring_is_skipped(self):
        fn = make_fn(name="__repr__", docstring=None, line_start=1, line_end=2)
        assert should_embed(fn) is False

    def test_dunder_with_docstring_embeds(self):
        fn = make_fn(name="__init__", docstring="Initialise the processor.", line_start=1, line_end=3)
        assert should_embed(fn) is True

    def test_exactly_5_lines_embeds(self):
        # boundary: line_count = 5 is NOT < 5, so it should embed even if private + no docstring
        fn = make_fn(name="_helper", docstring=None, line_start=1, line_end=5)
        assert should_embed(fn) is True

    def test_exactly_4_lines_private_no_docstring_skipped(self):
        fn = make_fn(name="_helper", docstring=None, line_start=1, line_end=4)
        assert should_embed(fn) is False


# ---------------------------------------------------------------------------
# should_embed_class
# ---------------------------------------------------------------------------

class TestShouldEmbedClass:

    def test_class_with_docstring_embeds(self):
        cls = make_cls(docstring="Handles payment processing.")
        assert should_embed_class(cls) is True

    def test_class_without_docstring_skipped(self):
        cls = make_cls(docstring=None)
        assert should_embed_class(cls) is False

    def test_empty_docstring_skipped(self):
        cls = make_cls(docstring="")
        assert should_embed_class(cls) is False


# ---------------------------------------------------------------------------
# build_embed_text — functions
# ---------------------------------------------------------------------------

class TestBuildEmbedText:

    def test_returns_full_body_directly(self):
        fn = make_fn(full_body="def process_payment():\n    return True")
        assert build_embed_text(fn) == "def process_payment():\n    return True"

    def test_does_not_reconstruct(self):
        # full_body is the source of truth — signature/docstring are not appended
        fn = make_fn(
            signature="def process_payment():",
            docstring="Some docstring.",
            full_body="def process_payment():\n    pass",
        )
        result = build_embed_text(fn)
        assert result == "def process_payment():\n    pass"
        assert result.count('"""') == 0  # docstring not re-injected


# ---------------------------------------------------------------------------
# build_class_embed_text
# ---------------------------------------------------------------------------

class TestBuildClassEmbedText:

    def test_single_line_docstring(self):
        cls = make_cls(
            name="PaymentProcessor",
            base_classes=["BaseProcessor"],
            docstring="Handles payment processing.",
        )
        result = build_class_embed_text(cls)
        assert result == 'class PaymentProcessor(BaseProcessor):\n    """Handles payment processing."""'

    def test_multiline_docstring(self):
        cls = make_cls(
            name="PaymentProcessor",
            base_classes=[],
            docstring="Handles payments.\n\nSupports retries.",
        )
        result = build_class_embed_text(cls)
        assert "Handles payments." in result
        assert "Supports retries." in result
        assert '"""' in result

    def test_no_base_classes(self):
        cls = make_cls(name="Standalone", base_classes=[], docstring="A standalone class.")
        result = build_class_embed_text(cls)
        assert result.startswith("class Standalone:")

    def test_multiple_base_classes(self):
        cls = make_cls(name="Foo", base_classes=["Bar", "Baz"], docstring="Foo class.")
        result = build_class_embed_text(cls)
        assert "class Foo(Bar, Baz):" in result

    def test_no_docstring_returns_signature_only(self):
        cls = make_cls(name="Bare", base_classes=[], docstring=None)
        result = build_class_embed_text(cls)
        assert result == "class Bare:"
        assert '"""' not in result


# ---------------------------------------------------------------------------
# make_function_embed_doc
# ---------------------------------------------------------------------------

class TestMakeFunctionEmbedDoc:

    def test_uuid_matches_function(self):
        fn = make_fn()
        doc = make_function_embed_doc(fn)
        assert doc.uuid == fn.uuid

    def test_text_is_full_body(self):
        fn = make_fn(full_body="def process():\n    pass")
        doc = make_function_embed_doc(fn)
        assert doc.text == "def process():\n    pass"

    def test_entity_type(self):
        doc = make_function_embed_doc(make_fn())
        assert doc.entity_type == "function"

    def test_metadata_fields(self):
        fn = make_fn()
        doc = make_function_embed_doc(fn)
        assert doc.metadata["name"]           == fn.name
        assert doc.metadata["qualified_name"] == fn.qualified_name
        assert doc.metadata["file_path"]      == fn.file_path
        assert doc.metadata["repo_id"]        == fn.repo_id
        assert doc.metadata["is_method"]      == fn.is_method
        assert doc.metadata["is_async"]       == fn.is_async


# ---------------------------------------------------------------------------
# make_class_embed_doc
# ---------------------------------------------------------------------------

class TestMakeClassEmbedDoc:

    def test_uuid_matches_class(self):
        cls = make_cls()
        doc = make_class_embed_doc(cls)
        assert doc.uuid == cls.uuid

    def test_entity_type(self):
        doc = make_class_embed_doc(make_cls())
        assert doc.entity_type == "class"

    def test_text_contains_class_name(self):
        cls = make_cls(name="PaymentProcessor", docstring="Handles payments.")
        doc = make_class_embed_doc(cls)
        assert "PaymentProcessor" in doc.text

    def test_metadata_fields(self):
        cls = make_cls()
        doc = make_class_embed_doc(cls)
        assert doc.metadata["name"]           == cls.name
        assert doc.metadata["qualified_name"] == cls.qualified_name
        assert doc.metadata["file_path"]      == cls.file_path
        assert doc.metadata["repo_id"]        == cls.repo_id
        assert doc.metadata["is_abstract"]    == cls.is_abstract
        assert doc.metadata["is_protocol"]    == cls.is_protocol
