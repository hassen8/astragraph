import textwrap
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

# ---------------------------------------------------------------------------
# PythonBuilder
# ---------------------------------------------------------------------------

class PythonBuilder:
    def __init__(self, ctx: ExtractionContext):
        self.ctx = ctx
        self._module_name = ctx.file_path.replace("/", ".").removesuffix(".py")
        self.extracted_attributes: list[AttributeNode] = []
        self.extracted_parameters: list[ParameterNode] = []
        # Cache for lookup during call extraction
        self._local_fn_lookup: dict[str, FunctionNode] = {}

    def build_module(self) -> ModuleNode:
        docstring = None
        for child in self.ctx.root.children:
            if child.type == "expression_statement" and child.child_count > 0:
                first = child.children[0]
                if first.type in ("string", "concatenated_string"):
                    docstring = _clean_python_docstring(self.ctx.src(first))
                    break
            elif child.type not in ("newline", "comment"):
                break

        fn_captures  = self.ctx.captures("function", self.ctx.root)
        cls_captures = self.ctx.captures("class", self.ctx.root)

        exported_names: list[str] = []

        for fn_def in fn_captures.get("fn.def", []):
            parent = fn_def.parent
            if parent and parent.type in ("module", "decorated_definition"):
                name_node = fn_def.child_by_field_name("name")
                if name_node:
                    exported_names.append(self.ctx.src(name_node))

        for cls_def in cls_captures.get("cls.def", []):
            parent = cls_def.parent
            if parent and parent.type in ("module", "decorated_definition"):
                name_node = cls_def.child_by_field_name("name")
                if name_node:
                    exported_names.append(self.ctx.src(name_node))

        import_captures = self.ctx.captures("import", self.ctx.root)
        imported_modules = [
            self.ctx.src(node).strip()
            for node in import_captures.get("import", [])
        ]

        # Is init?
        is_init = self.ctx.file_path.endswith("__init__.py")

        return ModuleNode(
            uuid=self.ctx.make_uuid(self._module_name),
            name=self._module_name,
            file_path=self.ctx.file_path,
            language=self.ctx.language,
            docstring=docstring,
            exported_names=exported_names,
            imported_modules=imported_modules,
            is_init=is_init,
            repo_id=self.ctx.repo_id,
        )

    def build_classes(self) -> list[ClassNode]:
        captures = self.ctx.captures("class", self.ctx.root)
        results: list[ClassNode] = []

        for cls_node in captures.get("cls.def", []):
            name_node = cls_node.child_by_field_name("name")
            body_node = cls_node.child_by_field_name("body")
            if not name_node:
                continue

            name = self.ctx.src(name_node)
            qname = f"{self._module_name}.{name}"
            class_uuid = self.ctx.make_uuid(qname)

            base_classes: list[str] = []
            bases_node = cls_node.child_by_field_name("superclasses")
            if bases_node:
                base_classes = [
                    self.ctx.src(child).strip()
                    for child in bases_node.children
                    if child.type not in (",", "(", ")", "keyword_argument")
                    and self.ctx.src(child).strip()
                ]

            method_names: list[str] = []
            if body_node:
                method_captures = self.ctx.captures("function", body_node)
                for mfn in method_captures.get("fn.def", []):
                    mname_node = mfn.child_by_field_name("name")
                    if mname_node:
                        method_names.append(self.ctx.src(mname_node))

            # Base properties
            is_abstract = any(b in ("ABC", "ABCMeta", "Abstract") for b in base_classes)
            is_protocol = any(b == "Protocol" or b == "typing.Protocol" for b in base_classes)
            is_exception = any(b in ("Exception", "BaseException") for b in base_classes)
            
            decorators = extract_decorators(cls_node, self.ctx)
            is_dataclass = any("dataclass" in d for d in decorators)

            # Build AttributeNodes for class variables and __init__ assignments
            attribute_names = self._build_and_collect_attributes(body_node, class_uuid, name)

            results.append(ClassNode(
                uuid=class_uuid,
                name=name,
                qualified_name=qname,
                file_path=self.ctx.file_path,
                line_start=cls_node.start_point[0] + 1,
                line_end=cls_node.end_point[0] + 1,
                language=self.ctx.language,
                docstring=extract_python_docstring(body_node, self.ctx),
                base_classes=base_classes,
                decorators=decorators,
                is_abstract=is_abstract,
                is_protocol=is_protocol,
                is_dataclass=is_dataclass,
                is_exception=is_exception,
                method_names=method_names,
                attribute_names=attribute_names,
                repo_id=self.ctx.repo_id,
            ))

        return results

    def _build_and_collect_attributes(self, class_body: Optional[Node], class_uuid: str, class_name: str) -> list[str]:
        if class_body is None:
            return []

        attr_names = []
        
        # 1. Look for __init__ attributes (self.x = ...)
        init_body: Optional[Node] = None
        fn_captures = self.ctx.captures("function", class_body)
        for fn_node in fn_captures.get("fn.def", []):
            name_node = fn_node.child_by_field_name("name")
            if name_node and self.ctx.src(name_node) == "__init__":
                init_body = fn_node.child_by_field_name("body")
                break

        if init_body is not None:
            ATTR_QUERY = """
                (assignment
                  left: (attribute
                    object: (identifier) @obj
                    attribute: (identifier) @attr)) @assign
            """
            try:
                from tree_sitter import Query, QueryCursor
                q = Query(self.ctx.lang_obj, ATTR_QUERY)
                cursor = QueryCursor(q)
                grouped = cursor.captures(init_body)
                
                obj_nodes  = grouped.get("obj",  [])
                attr_nodes = grouped.get("attr", [])
                
                for obj_node, attr_node in zip(obj_nodes, attr_nodes):
                    if self.ctx.src(obj_node) == "self":
                        name = self.ctx.src(attr_node)
                        attr_names.append(f"self.{name}")
                        
                        attr_uuid = self.ctx.make_uuid(class_uuid + "::attr_init::" + name)
                        
                        self.extracted_attributes.append(AttributeNode(
                            uuid=attr_uuid,
                            name=name,
                            full_name=f"self.{name}",
                            type_hint=None, # Typed assignments in __init__ are not usually straightforward to parse this way
                            default=None,
                            is_instance=True,
                            is_class_var=False,
                            line=attr_node.start_point[0] + 1,
                            parent_uuid=class_uuid,
                            repo_id=self.ctx.repo_id,
                        ))
            except Exception:
                pass
                
        # 2. Look for class variables
        # This requires more complex Tree-sitter querying, returning empty for class vars right now
        # until a python class_var query is added.
                
        return attr_names

    def build_functions(self) -> list[FunctionNode]:
        captures = self.ctx.captures("function", self.ctx.root)
        fn_defs  = captures.get("fn.def", [])

        seen_bytes: set[int] = set()
        results: list[FunctionNode] = []

        for fn_node in fn_defs:
            if fn_node.start_byte in seen_bytes:
                continue
            seen_bytes.add(fn_node.start_byte)

            node_obj = self._build_function_node(fn_node)
            results.append(node_obj)
            self._local_fn_lookup[node_obj.name] = node_obj
            self._local_fn_lookup[node_obj.qualified_name] = node_obj

        return results

    def _build_function_node(self, fn_node: Node) -> FunctionNode:
        # In tree-sitter-python, async def is a function_definition whose
        # first non-whitespace child is the "async" keyword token.
        is_async = any(c.type == "async" for c in fn_node.children)

        name_node = fn_node.child_by_field_name("name")
        body_node = fn_node.child_by_field_name("body")
        name      = self.ctx.src(name_node) if name_node else "<unknown>"

        class_name = _enclosing_class_name(fn_node, self.ctx)
        qname = (
            f"{self._module_name}"
            f"{'.' + class_name if class_name else ''}"
            f".{name}"
        )
        fn_uuid = self.ctx.make_uuid(qname)

        decorators = extract_decorators(fn_node, self.ctx)

        # Build ParameterNodes
        self._build_parameters(fn_node, fn_uuid)

        return FunctionNode(
            uuid=fn_uuid,
            name=name,
            qualified_name=qname,
            file_path=self.ctx.file_path,
            line_start=fn_node.start_point[0] + 1,
            line_end=fn_node.end_point[0] + 1,
            language=self.ctx.language,
            signature=build_python_signature(fn_node, self.ctx),
            docstring=extract_python_docstring(body_node, self.ctx),
            return_type=extract_python_return_type(fn_node, self.ctx),
            is_async=is_async,
            is_method=class_name is not None,
            is_property=any("property" in d for d in decorators),
            is_classmethod=any("classmethod" in d for d in decorators),
            is_staticmethod=any("staticmethod" in d for d in decorators),
            is_overload=any("overload" in d for d in decorators),
            decorators=decorators,
            body_preview=self.ctx.src(fn_node)[:300],
            full_body=self.ctx.src(fn_node),
            complexity=compute_complexity(fn_node),
            repo_id=self.ctx.repo_id,
        )

    def _build_parameters(self, actual_fn_node: Node, fn_uuid: str):
        params_node = actual_fn_node.child_by_field_name("parameters")
        if not params_node:
            return
            
        position = 0
        for child in params_node.children:
            if child.type in ("(", ")", ",", "comment"):
                continue
                
            param_name = None
            type_hint = None
            default_val = None
            is_variadic = False
            is_keyword = False
            
            # Very basic extraction (Tree-sitter parameter structure is nested depending on hints/defaults)
            # This covers identifiers, typed_parameters, default_parameters, etc.
            if child.type == "identifier":
                param_name = self.ctx.src(child)
            elif child.type == "typed_parameter":
                for sub in child.children:
                    if sub.type == "identifier":
                        param_name = self.ctx.src(sub)
                    elif sub.type == "type":
                        type_hint = self.ctx.src(sub)
            elif child.type == "default_parameter":
                for sub in child.children:
                    if sub.type == "identifier":
                        param_name = self.ctx.src(sub)
                    # not perfectly capturing type_hint & default together without deeper traversal, but covers basic
            elif child.type == "typed_default_parameter":
                # ... skip detailed traversal for now
                pass
            elif child.type == "list_splat_pattern": # *args
                is_variadic = True
            elif child.type == "dictionary_splat_pattern": # **kwargs
                is_keyword = True
                
            if not param_name and not is_variadic and not is_keyword:
                param_name = self.ctx.src(child) # fallback raw text
                
            p_uuid = self.ctx.make_uuid(fn_uuid + "::param::" + str(position))
            
            self.extracted_parameters.append(ParameterNode(
                uuid=p_uuid,
                name=param_name or "<unknown>",
                type_hint=type_hint,
                default=default_val,
                position=position,
                is_self=(param_name in ("self", "cls")),
                is_variadic=is_variadic,
                is_keyword=is_keyword,
                parent_uuid=fn_uuid,
                repo_id=self.ctx.repo_id,
            ))
            position += 1

    def build_calls(self) -> list[dict]:
        """
        Returns a list of raw call dicts:
        {"src_uuid": caller_uuid, "callee": text, "call_site_line": int, "is_conditional": bool}
        """
        calls_data: list[dict] = []
        
        # We need to iterate over all functions we just built.
        for qname, caller in self._local_fn_lookup.items():
            if qname != caller.qualified_name:
                continue # only process the canonical objects once
                
            fn_body_node = self._find_function_node_by_line(caller.line_start)
            if not fn_body_node:
                continue

            call_captures = self.ctx.captures("call", fn_body_node)
            call_sites    = call_captures.get("call.site", [])
            callee_names  = call_captures.get("call.name",   [])
            callee_attrs  = call_captures.get("call.attr",   [])

            call_pairs: list[tuple[Node, str]] = []

            for site, name_node in zip(call_sites, callee_names):
                call_pairs.append((site, self.ctx.src(name_node)))

            for site, attr_node in zip(call_sites[len(callee_names):], callee_attrs):
                call_pairs.append((site, self.ctx.src(attr_node)))

            for site_node, callee_text in call_pairs:
                calls_data.append({
                    "src_uuid": caller.uuid,
                    "callee": callee_text,
                    "call_site_line": site_node.start_point[0] + 1,
                    "is_conditional": _is_inside_branch(site_node),
                })
                
        return calls_data

    def _find_function_node_by_line(self, line_start: int) -> Optional[Node]:
        captures = self.ctx.captures("function", self.ctx.root)
        all_fn_nodes = captures.get("fn.def", []) + captures.get("fn.async_wrapper", [])
        for node in all_fn_nodes:
            if node.start_point[0] + 1 == line_start:
                return node
        return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_python_docstring(raw: str) -> str:
    for q in ('\"\"\"', "'''", '"', "'"):
        if raw.startswith(q) and raw.endswith(q) and len(raw) > 2 * len(q):
            raw = raw[len(q):-len(q)]
            break
    return textwrap.dedent(raw).strip()

def extract_python_docstring(body_node: Optional[Node], ctx: ExtractionContext) -> Optional[str]:
    if body_node is None:
        return None
    for child in body_node.children:
        if child.type == "expression_statement" and child.child_count > 0:
            first = child.children[0]
            if first.type in ("string", "concatenated_string"):
                return _clean_python_docstring(ctx.src(first))
        elif child.type not in ("newline", "comment", "pass_statement"):
            break
    return None

def build_python_signature(fn_node: Node, ctx: ExtractionContext) -> str:
    is_async    = any(c.type == "async" for c in fn_node.children)
    name_node   = fn_node.child_by_field_name("name")
    params_node = fn_node.child_by_field_name("parameters")
    ret_node    = fn_node.child_by_field_name("return_type")

    prefix = "async def" if is_async else "def"
    sig = f"{prefix} {ctx.src(name_node)}"
    if params_node:
        sig += ctx.src(params_node)
    if ret_node:
        sig += f" -> {ctx.src(ret_node)}"
    return sig

def extract_python_return_type(fn_node: Node, ctx: ExtractionContext) -> Optional[str]:
    ret_node = fn_node.child_by_field_name("return_type")
    return ctx.src(ret_node).lstrip("->").strip() if ret_node else None

def compute_complexity(fn_node: Node) -> int:
    BRANCH_TYPES = {
        "if_statement", "elif_clause", "for_statement",
        "while_statement", "except_clause", "with_statement",
        "conditional_expression", "boolean_operator",
    }
    def _count(node: Node) -> int:
        total = 1 if node.type in BRANCH_TYPES else 0
        for child in node.children:
            total += _count(child)
        return total
    return 1 + _count(fn_node)

def extract_decorators(fn_or_cls_node: Node, ctx: ExtractionContext) -> list[str]:
    parent = fn_or_cls_node.parent
    if parent is None or parent.type != "decorated_definition":
        return []
    return [
        ctx.src(child).strip()
        for child in parent.children
        if child.type == "decorator"
    ]

def _enclosing_class_name(node: Node, ctx: ExtractionContext) -> Optional[str]:
    parent = node.parent
    while parent:
        if parent.type == "class_definition":
            name_node = parent.child_by_field_name("name")
            return ctx.src(name_node) if name_node else None
        parent = parent.parent
    return None

def _is_inside_branch(node: Node) -> bool:
    BRANCH_TYPES = {
        "if_statement", "elif_clause",
        "while_statement", "for_statement",
        "try_statement", "except_clause",
    }
    parent = node.parent
    while parent:
        if parent.type in BRANCH_TYPES:
            return True
        parent = parent.parent
    return False
