#!/usr/bin/env python3
"""Integration test to verify that the query sanitisation logic correctly prevents FTS5 syntax errors."""
import os
import sys
import tempfile
from pathlib import Path

# Add project root and custom system site-packages to path if needed
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from edumem.core.beam import BeamMemory, _fts_search_working

def run_test():
    print("[*] Creating temporary database for FTS5 sanitisation test...")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        # Initialize our custom system
        beam = BeamMemory(db_path=db_path)
        
        # Ingest facts with hyphens
        print("[*] Ingesting test facts into the custom system...")
        fact_content = "Fact 1257. Salem-Keizer Volcanoes is associated with the sport of baseball."
        beam.remember(
            content=fact_content,
            source='conversation',
            metadata={'chunk_id': 1}
        )
        
        # Test query containing complex characters that historically crashed SQLite FTS5 MATCH
        test_queries = [
            "Salem-Keizer Volcanoes",
            "Which sport is Salem-Keizer associated with?",
            "Salem-Keizer",
            "Salem-Keizer: Volcanoes"
        ]
        
        for q in test_queries:
            print(f"[*] Calling _fts_search_working with query: '{q}'")
            # This will raise sqlite3.OperationalError directly if FTS5 syntax parser fails!
            results = _fts_search_working(beam.conn, q, k=5)
            print(f"    Raw FTS5 results: {results}")
            assert len(results) > 0, f"FTS5 returned no results for query '{q}'!"
            
        print("[+] SUCCESS: All FTS5 query sanitisation tests passed!")
        return 0

    except Exception as e:
        print(f"[-] FAILURE: Test raised an exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up temporary DB files
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass
            for suffix in ["-shm", "-wal", "-vec"]:
                comp = db_path.with_name(db_path.name + suffix)
                if comp.exists():
                    try:
                        comp.unlink()
                    except Exception:
                        pass

if __name__ == "__main__":
    sys.exit(run_test())