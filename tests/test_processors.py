import numpy as np
import pytest

from src.data.processors import CodeProcessor, DiffImageProcessor, ContextProcessor
from src.config import config


def test_code_processor():
    processor = CodeProcessor()
    
    diff_text = """diff --git a/file.py b/file.py
index 1234567..89abcdef 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 def hello():
-    print("Hello")
+    print("Hello World")
+    return True"""
    
    processed = processor.process_diff(diff_text)
    assert "diff --git" not in processed
    assert "index" not in processed
    assert "--- a/" not in processed
    assert "+++ b/" not in processed
    assert "print(\"Hello World\")" in processed


def test_diff_image_processor():
    processor = DiffImageProcessor()
    
    #test image generation
    diff_text = """+def add(a, b):
+    return a + b
-def add(x, y):
-    return x + y"""
    
    image = processor.diff_to_image(diff_text)
    assert isinstance(image, np.ndarray)
    assert image.shape == (*config.data.image_size, 3)


def test_context_processor():
    processor = ContextProcessor()
    
    title = "Add new feature"
    description = "This PR adds a new feature"
    comments = ["LGTM!", "Please fix tests"]
    
    context = processor.process_context(title, description, comments)
    assert isinstance(context, str)
    assert title in context
    assert description in context
    assert all(comment in context for comment in comments)
