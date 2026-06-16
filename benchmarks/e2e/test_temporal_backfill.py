import os
import sys
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# We import through the edumem alias (symlink to edumem) to verify integration
from edumem.core.beam import BeamMemory

def run_test():
    print("[*] Creating temporary database for Temporal Backfill TDD test...")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        beam = BeamMemory(db_path=db_path)
        
        # Test Case 1: Ingest batch with historical timestamps
        # We must use a timestamp within the 7-day working memory retention window
        # otherwise the core retention manager will correctly delete it on commit.
        print("[*] Ingesting batch items with historical timestamps...")
        historical_dt = datetime.now() - timedelta(days=2)
        historical_timestamp = historical_dt.isoformat()
        
        items = [
            {
                "content": "Alice lives in Madrid.",
                "source": "conversation",
                "timestamp": historical_timestamp,
                "importance": 0.8
            }
        ]
        
        # This is where remember_batch is called
        beam.remember_batch(items)
        
        # Assertions for remember_batch
        cursor = beam.conn.cursor()
        cursor.execute("SELECT timestamp, id FROM working_memory WHERE content LIKE 'Alice%'")
        row = cursor.fetchone()
        assert row is not None, "Alice memory was not written to working_memory"
        
        stored_wm_ts = row[0]
        stored_wm_id = row[1]
        print(f"[*] Stored working_memory timestamp: {stored_wm_ts}")
        assert stored_wm_ts == historical_timestamp, f"Expected {historical_timestamp}, but got {stored_wm_ts}"
        
        # Assertions for episodic graph tables
        cursor.execute("SELECT timestamp FROM gists WHERE memory_id = ?", (stored_wm_id,))
        gist_row = cursor.fetchone()
        assert gist_row is not None, "Gist was not generated or stored"
        stored_gist_ts = gist_row[0]
        print(f"[*] Stored gist timestamp: {stored_gist_ts}")
        assert stored_gist_ts == historical_timestamp, f"Expected gist timestamp {historical_timestamp}, but got {stored_gist_ts}"

        cursor.execute("SELECT timestamp FROM graph_edges WHERE source = ?", (f"gist_{stored_wm_id}",))
        edge_row = cursor.fetchone()
        if edge_row:
            stored_edge_ts = edge_row[0]
            print(f"[*] Stored edge timestamp: {stored_edge_ts}")
            assert stored_edge_ts == historical_timestamp, f"Expected edge timestamp {historical_timestamp}, but got {stored_edge_ts}"
        
        # Test Case 2: Ingest single memory with explicit timestamp override
        print("[*] Ingesting single memory with custom timestamp...")
        single_dt = datetime.now() - timedelta(days=3)
        single_timestamp = single_dt.isoformat()
        
        # We pass explicit timestamp override
        single_id = beam.remember(
            content="Bob is an AI scientist.",
            source="conversation",
            timestamp=single_timestamp
        )
        
        cursor.execute("SELECT timestamp FROM working_memory WHERE id = ?", (single_id,))
        single_row = cursor.fetchone()
        assert single_row is not None, "Bob memory was not written to working_memory"
        stored_single_ts = single_row[0]
        print(f"[*] Stored single memory timestamp: {stored_single_ts}")
        assert stored_single_ts == single_timestamp, f"Expected single memory timestamp {single_timestamp}, but got {stored_single_ts}"
        
        print("[+] SUCCESS: All Temporal Backfill TDD tests passed!")
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
