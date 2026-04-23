from typing import Optional
from tree_sitter import Node

from .extractor import ExtractionContext
from ..models import (
    ModuleNode,
    PackageNode,
    ClassNode,
    FunctionNode,
    AttributeNode,
    ParameterNode,
    make_uuid,
)

class GoBuilder:
    def __init__(self, ctx: ExtractionContext):
        self.ctx = ctx
        self._module_name = ctx.file_path.replace("/", ".").removesuffix(".go")
        self.extracted_attributes: list[AttributeNode] = []
        self.extracted_parameters: list[ParameterNode] = []
        
    def build_module(self) -> ModuleNode:
        return ModuleNode(
            uuid=self.ctx.make_uuid(self._module_name),
            name=self._module_name,
            file_path=self.ctx.file_path,
            language=self.ctx.language,
            docstring=None,
            exported_names=[],
            imported_modules=[],
            is_init=False,
            repo_id=self.ctx.repo_id,
        )

    def build_package(self) -> PackageNode:
        # Go package name comes from the `package <name>` declaration in source,
        # not the directory. Real implementation: query (package_clause
        # (package_identifier) @pkg.name) from self.ctx.root. Returning a
        # directory-based approximation until Go queries are implemented.
        from pathlib import Path
        pkg_dir  = str(Path(self.ctx.file_path).parent)
        pkg_name = pkg_dir.replace("/", ".") if pkg_dir != "." else self.ctx.repo_id
        return PackageNode(
            uuid=make_uuid(self.ctx.repo_id, pkg_name),
            name=pkg_name,
            directory=pkg_dir,
            is_namespace=True,
            has_init=False,
            init_file=None,
            repo_id=self.ctx.repo_id,
        )

    def build_classes(self) -> list[ClassNode]:
        return []

    def build_functions(self) -> list[FunctionNode]:
        return []

    def build_calls(self) -> list[dict]:
        return []
