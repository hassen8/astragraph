import os
import sys

# setup path to astragraph
sys.path.insert(0, "/home/hsali/projects/astragraph")

from tree_sitter import Language, Parser
from ingestion.extractors.extractor import extract_file

def get_python_language():
    # Attempting to load python tree-sitter language object
    try:
        import tree_sitter_python
        return Language(tree_sitter_python.language())
    except ImportError:
        print("Missing tree_sitter_python... please ensure it's installed.")
        return None

def test_fastapi_extraction():
    lang_obj = get_python_language()
    if not lang_obj:
        return

    parser = Parser(lang_obj)

    fastapi_dir = "/home/hsali/projects/fastapi/fastapi"
    
    if not os.path.exists(fastapi_dir):
        print(f"Directory {fastapi_dir} not found. Are you sure fastapi repo is cloned here?")
        return
        
    py_files = []
    for root, dirs, files in os.walk(fastapi_dir):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
                
    print(f"Found {len(py_files)} python files in {fastapi_dir}")
    
    total_modules = 0
    total_classes = 0
    total_functions = 0
    total_attributes = 0
    total_parameters = 0
    total_calls = 0

    for idx, filepath in enumerate(py_files):
        with open(filepath, "rb") as f:
            source = f.read()

        tree = parser.parse(source)
        
        try:
            mod, _pkg, cls, fn, attrs, params, calls = extract_file(
                file_path=filepath,
                language="python",
                root=tree.root_node,
                source=source,
                lang_obj=lang_obj,
                repo_id="fastapi-repo",
            )
            total_modules += 1
            total_classes += len(cls)
            total_functions += len(fn)
            total_attributes += len(attrs)
            total_parameters += len(params)
            total_calls += len(calls)
            
        except Exception as e:
            print(f"FAILED on {filepath}")
            import traceback
            traceback.print_exc()
            return
            
    print("SUCCESS: Full FastAPI repository node extraction successful!")
    print(f"Modules: {total_modules}")
    print(f"Classes: {total_classes}")
    print(f"Functions: {total_functions}")
    print(f"Attributes: {total_attributes}")
    print(f"Parameters: {total_parameters}")
    print(f"Calls: {total_calls}")

if __name__ == "__main__":
    test_fastapi_extraction()
