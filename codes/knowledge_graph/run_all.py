"""Rebuild the whole knowledge graph in order. Run from the project root:
    python codes/knowledge_graph/run_all.py
"""
import runpy
from pathlib import Path

HERE = Path(__file__).parent
for step in ["extract_pdfs.py", "label_seasons.py", "build_graph.py", "build_html_viewer.py", "export_neo4j.py"]:
    print(f"\n=== {step} ===")
    runpy.run_path(str(HERE / step), run_name="__main__")
