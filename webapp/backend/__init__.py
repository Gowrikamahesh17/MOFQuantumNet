"""Makes src/ (the core pipeline — config.py, data_ingestion.py, etc.) importable as
plain top-level modules from anywhere in webapp/backend/, without every single file
repeating its own sys.path.insert. Runs once, on first import of this package."""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
