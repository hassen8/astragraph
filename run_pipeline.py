import os
from pathlib import Path
from config import Config
from ingestion.walker import walk_repo
from ingestion.models import ModuleNode, make_uuid

def run_pipeline(repo_path: str):
    print("=== Starting Pipeline Progression ===")
    
    # 1. Configuration
    config = Config()
    print(f"Loaded Configuration:")
    print(f"  Chroma Path: {config.chroma_path}")
    print(f"  Embed Model: {config.embed_model}")
    print(f"  Chunk Line Limit: {config.chunk_line_limit}\n")

    print(f"Scanning repository: {repo_path}")
    repo_id = os.path.basename(os.path.abspath(repo_path))
    
    # 2. Walker
    # Run the codebase traversal respecting .gitignore and checking for binary/generated files
    files_discovered = list(walk_repo(repo_path))
    print(f"Walker discovered {len(files_discovered)} valid source file(s) to process.")

    # 3. Models & UUID Generation
    # Since we have designed the extractors theoretically but haven't implemented them yet,
    # we simulate the extraction phase to demonstrate our data structures.
    for index, (rel_path, language) in enumerate(files_discovered, start=1):
        print(f"\n--- Processing File {index}: {rel_path} ({language}) ---")
        
        # Rough conversion of file path to a module qualified name
        qualified_name = rel_path.replace(os.sep, ".").replace(".py", "").replace(".ts", "")
        
        # Deterministic UUID generation
        module_uuid = make_uuid(repo_id, rel_path, qualified_name)
        print(f"-> Generated Deterministic UUID: {module_uuid}")
        
        # Showcase instantiating a ModuleNode graph structure
        module_node = ModuleNode(
            uuid=module_uuid,
            name=qualified_name,
            file_path=rel_path,
            language=language,
            docstring="[Simulated docstring extracted via Treesitter]",
            exported_names=["MyClass", "my_helper_function"],
            imported_modules=["sys", "os", "pathlib"],
            repo_id=repo_id
        )
        
        print(f"-> Created ModuleNode Data Structure:")
        print(f"   Name: {module_node.name}")
        print(f"   Language: {module_node.language}")
        print(f"   Docstring: {module_node.docstring}")
        print(f"   Exports: {module_node.exported_names}")

    print("\n=== Pipeline execution completed ===")

if __name__ == "__main__":
    # Point this to the dummy repo provided
    run_pipeline("/home/hsali/projects/fastapi")
