"""
Brings the knowledge base up to date with one command:
  collect new/changed content -> clean + deduplicate -> chunk -> rebuild the indexes

Run from the project folder:
    python update.py
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent / "src"
STEPS = [
    ("collect new and changed content", "ingest.py"),
    ("clean and deduplicate", "dedup.py"),
    ("split into chunks", "chunk.py"),
    ("rebuild the search indexes", "index.py"),
]

start = time.time()
for i, (description, script) in enumerate(STEPS, start=1):
    print(f"\n=== step {i}/{len(STEPS)}: {description} ({script}) ===")
    # run each step with the same Python as this script (the one from .venv)
    result = subprocess.run([sys.executable, str(SRC / script)])
    if result.returncode != 0:
        sys.exit(f"\nstopped: {script} failed, the later steps were not run")

print(f"\nknowledge base up to date ({time.time() - start:.0f}s)")