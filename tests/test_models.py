import pytest
from ingestion.models import make_uuid

def test_make_uuid_deterministic():
    uuid1 = make_uuid("repo1", "src/main.py", "main.MyClass.method")
    uuid2 = make_uuid("repo1", "src/main.py", "main.MyClass.method")
    assert uuid1 == uuid2

def test_make_uuid_distinct():
    uuid1 = make_uuid("repo1", "src/main.py", "main.MyClass.method")
    uuid2 = make_uuid("repo1", "src/main.py", "main.MyClass.method2")
    uuid3 = make_uuid("repo2", "src/main.py", "main.MyClass.method")
    
    assert uuid1 != uuid2
    assert uuid1 != uuid3
