#!/usr/bin/env python3
"""edumem Master Regression Benchmarking & Testing Suite.

Sequentially executes our four core official evaluation pipelines (LongMemEval, BEAM Benchmark,
PersonaMem-v2 128K, and MemoryAgentBench), parses their outputs/logs, and compiles a
unified Markdown Performance Report. Can be run in QUICK mode to verify regressions in under 60 seconds.
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path

# Paths to the different modules
EDUMEM_ROOT = Path("/opt/edumem")
LONGMEMEVAL_ROOT = Path("/opt/LongMemEval")
MAB_ROOT = Path("/opt/MemoryAgentBench")

PYTHON_ENV = EDUMEM_ROOT / ".venv" / "bin" / "python3"

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

def run_command(command, cwd, env_overrides=None):
    """Run a shell command, showing output in real-time."""
    print(f"\n[*] Executing: {' '.join(command)}")
    print(f"[*] Cwd: {cwd}")
    
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
        
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    output_lines = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        output_lines.append(line)
        
    process.stdout.close()
    return_code = process.wait()
    return return_code, "".join(output_lines)

def main():
    # Load all environment configurations
    load_env(EDUMEM_ROOT / ".env")
    load_env(Path("/root/.hermes/.env"))
    
    # Check if Quick Mode is requested (default to False unless QUICK=1 env var is set)
    quick_mode = os.getenv("QUICK", "0") == "1"
    
    limit_500 = "10" if quick_mode else "500"
    limit_70 = "10" if quick_mode else "70"
    limit_100 = "10" if quick_mode else "100"
    
    scales_arg = "100K" if quick_mode else "100K,500K"
    sample_arg = "1" if quick_mode else "3"
    
    print("======================================================================")
    print("🚀 STARTING EDUMEM MASTER REGRESSION SUITE")
    print(f"[*] Mode: {'QUICK TEST (Subsets)' if quick_mode else 'HIGH-FIDELITY (Full Runs)'}")
    print("======================================================================\n")
    
    # 1. Run Official LongMemEval Benchmark
    print("--- STEP 1: Running Official LongMemEval ---")
    longmem_env = {
        "LIMIT": limit_500,
        "PYTHONPATH": str(LONGMEMEVAL_ROOT)
    }
    longmem_cmd = [str(PYTHON_ENV), "run_longmemeval_full.py"]
    t0 = time.time()
    code_longmem, out_longmem = run_command(longmem_cmd, LONGMEMEVAL_ROOT, longmem_env)
    t_longmem = time.time() - t0
    
    longmem_accuracy = "N/A"
    if code_longmem == 0:
        for line in out_longmem.splitlines():
            if "Overall Accuracy:" in line:
                longmem_accuracy = line.split("Overall Accuracy:")[1].strip()
                
    # 2. Run Official BEAM End-to-End Benchmark
    print("\n--- STEP 2: Running Official BEAM End-to-End ---")
    beam_env = {
        "OPENROUTER_API_KEY": os.getenv("NAN_APY_KEY") or os.getenv("OPENAI_API_KEY", ""),
        "OPENROUTER_BASE_URL": os.getenv("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1"),
        "PYTHONPATH": str(EDUMEM_ROOT)
    }
    beam_cmd = [
        str(PYTHON_ENV),
        str(EDUMEM_ROOT / ".venv/lib/python3.13/site-packages/tools/evaluate_beam_end_to_end.py"),
        "--scales", scales_arg,
        "--sample", sample_arg,
        "--model", "qwen3.6",
        "--pure-recall"
    ]
    t0 = time.time()
    code_beam, out_beam = run_command(beam_cmd, EDUMEM_ROOT, beam_env)
    t_beam = time.time() - t0
    
    # Parse BEAM Accuracy from output JSON
    beam_accuracy = "N/A"
    if code_beam == 0:
        try:
            summary_path = EDUMEM_ROOT / "results/beam_e2e_summary.json"
            if not summary_path.exists():
                fallback = EDUMEM_ROOT / ".venv/lib/python3.13/site-packages/results/beam_e2e_summary.json"
                if fallback.exists():
                    summary_path = fallback
            summary_data = json.loads(summary_path.read_text())
            ab_summary = summary_data.get("ability_summary", {})
            
            # Print average of the tested scales
            scale_scores = []
            for sc in scales_arg.split(","):
                if sc in ab_summary:
                    overall_score = ab_summary[sc].get("OVERALL", {}).get("avg_score", 0.0)
                    scale_scores.append(overall_score * 100)
            if scale_scores:
                avg_beam = sum(scale_scores) / len(scale_scores)
                beam_accuracy = f"{avg_beam:.2f}% (Scales: {scales_arg})"
        except Exception as e:
            print(f"[!] Error parsing BEAM summary file: {e}")

    # 3. Run Official PersonaMem-v2 (128K Context)
    print("\n--- STEP 3: Running Official PersonaMem-v2 (128K) ---")
    pm_env = {
        "LIMIT": limit_70,
        "PYTHONPATH": str(EDUMEM_ROOT)
    }
    pm_128k_cmd = [str(PYTHON_ENV), "benchmarks/personamem_v2_128k.py"]
    t0 = time.time()
    code_128k, out_128k = run_command(pm_128k_cmd, EDUMEM_ROOT, pm_env)
    t_128k = time.time() - t0
    
    # Parse 128K Accuracy from results JSONL
    pm_128k_accuracy = "N/A"
    try:
        jsonl_path_128k = EDUMEM_ROOT / "results/personamem_v2_128k_results.jsonl"
        lines_128k = jsonl_path_128k.read_text().splitlines()
        correct = 0
        total = 0
        for line in lines_128k[-int(limit_70):]:
            data = json.loads(line)
            correct += data.get("ok", 0)
            total += 1
        pm_128k_accuracy = f"{correct}/{total} ({correct/total*100:.2f}%)"
    except Exception as e:
        print(f"[!] Error parsing 128K results file: {e}")

    # 4. Run Official MemoryAgentBench (Conflict Resolution)
    print("\n--- STEP 4: Running Official MemoryAgentBench (FactConsolidation) ---")
    mab_env = {
        "PYTHONPATH": str(MAB_ROOT),
        "LIMIT": limit_100,
        "OPENAI_API_KEY": os.getenv("NAN_APY_KEY") or os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_BASE_URL": os.getenv("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1")
    }
    mab_cmd = [
        str(PYTHON_ENV), "main.py",
        "--agent_config", "configs/agent_conf/RAG_Agents/gpt-4o-mini/Simple_rag_qwen3.6_edumem.yaml",
        "--dataset_config", "configs/data_conf/Conflict_Resolution/Factconsolidation_sh_32k.yaml",
        "--max_test_queries_ablation", limit_100,
        "--force"
    ]
    t0 = time.time()
    code_mab, out_mab = run_command(mab_cmd, MAB_ROOT, mab_env)
    t_mab = time.time() - t0
    
    # Parse MAB Accuracy from output JSON
    mab_accuracy = "N/A"
    try:
        res_path = MAB_ROOT / "outputs/qwen3.6-edumem/Conflict_Resolution/factconsolidation_sh_32k_unknown_in32768_size10_shots0_max_samples1_k10_chunk4096_results.json"
        data = json.loads(res_path.read_text())
        avg = data.get("averaged_metrics", {})
        mab_accuracy = f"{avg.get('exact_match', 0.0):.2f}% Exact Match / {avg.get('f1', 0.0):.2f}% F1"
    except Exception as e:
        print(f"[!] Error parsing MemoryAgentBench results file: {e}")

    # Compile Unified Performance Report
    report = f"""# edumem Master Performance Regression Report

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
Run Configuration: `QUICK_MODE={quick_mode}`

## 📊 Summary Performance Ledger

| Evaluation Suite | Dataset Context | Metric Mode | Target Goal | **Current edumem Score** | Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LongMemEval** | Oracle Sessions | LLM-as-judge | `95.40%` (OMEGA) | **{longmem_accuracy}** | {t_longmem:.1f}s |
| **BEAM Benchmark** | 100K/500K Scales | LLM-as-judge | `64.10%` (API Client)| **{beam_accuracy}** | {t_beam:.1f}s |
| **PersonaMem-v2 (128K)** | 128K Histories | Multi-Option QA | `80.00%` (Milestone) | **{pm_128k_accuracy}** | {t_128k:.1f}s |
| **MemoryAgentBench** | 32K Fact-Consol | Substring Match | `54.00%` (HippoRAG) | **{mab_accuracy}** | {t_mab:.1f}s |

---

*Note: In QUICK mode, accuracy is parsed over the subset of evaluated samples. High-fidelity runs must be executed with QUICK=0 for full certification.*
"""
    
    print("\n======================================================================")
    print("📋 UNIFIED EDUMEM PERFORMANCE REPORT")
    print("======================================================================")
    print(report)
    print("======================================================================\n")
    
    # Save Report to file
    report_dir = EDUMEM_ROOT / "results"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "regression_report.md").write_text(report)
    print(f"[+] Master Performance Report saved successfully to: {report_dir}/regression_report.md")

if __name__ == "__main__":
    main()
