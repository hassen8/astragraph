from typing import Optional
from tree_sitter import Node

from .extractor import ExtractionContext
from ..models import (
    ModuleNode,
    ClassNode,
    FunctionNode,
    AttributeNode,
    ParameterNode,
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

    def build_classes(self) -> list[ClassNode]:
        return []

    def build_functions(self) -> list[FunctionNode]:
        return []

    def build_calls(self) -> list[dict]:
        return []
