#!/usr/bin/env python3
"""
scripts/make_performance_report.py

Generates performance tables and graphs based on real experimental results in results/.
Exits with clear messages for missing inputs and never fabricates data.
"""

import sys
import json
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    repo_dir = Path(__file__).resolve().parent.parent
    results_dir = repo_dir / "results"

    print("=== CROP YIELD FL PERFORMANCE REPORT GENERATOR ===")
    print(f"Reading results from: {results_dir}")

    # 1. Audit input files
    baseline_path = results_dir / "baseline_metrics.json"
    lstm_log_path = results_dir / "Sreenidhi_LSTM_run_log.csv"
    encryption_path = results_dir / "encryption_overhead.csv"
    fl_rounds_path = results_dir / "fl_round_metrics.csv"  # hypothetical multi-round output

    has_baseline = baseline_path.exists()
    has_lstm = lstm_log_path.exists()
    has_encryption = encryption_path.exists()
    has_fl_rounds = fl_rounds_path.exists()

    print(f"[-] baseline_metrics.json found: {has_baseline}")
    print(f"[-] Sreenidhi_LSTM_run_log.csv found: {has_lstm}")
    print(f"[-] encryption_overhead.csv found: {has_encryption}")
    print(f"[-] Multi-round FL metrics found: {has_fl_rounds}")

    # 2. Process Baseline Metrics
    baseline_data = {}
    if has_baseline:
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)

    # 3. Process Centralized LSTM Metrics
    lstm_data = {}
    if has_lstm:
        with open(lstm_log_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                lstm_data = rows[0]

    # 4. Process Encryption Overhead
    enc_rows = []
    if has_encryption:
        with open(encryption_path, "r") as f:
            reader = csv.DictReader(f)
            enc_rows = list(reader)

    # 5. Generate Model Comparison Tables (Task C)
    print("\n--- MODEL METRICS TABLES ---")
    print("\nTable C1: TEST Split Evaluation (Real Data Only)")
    print("| Model Architecture | Test MAE (kg/ha) | Test RMSE (kg/ha) | Test R² | Data Split / Notes |")
    print("|---|---:|---:|---:|---|")
    
    # Mean baseline
    print("| Mean Baseline | N/A | N/A | N/A | Not provided in handoff |")
    # Linear Regression
    print("| Linear Regression | N/A | N/A | N/A | Not provided in handoff |")
    
    # Random Forest Baseline
    if has_baseline:
        tm = baseline_data.get("test_metrics", {})
        print(f"| Random Forest (Official) | {tm.get('MAE', 'N/A'):.4f} | {tm.get('RMSE', 'N/A'):.4f} | {tm.get('R2', 'N/A'):.4f} | Frozen test set (75 samples) |")
    else:
        print("| Random Forest (Official) | N/A | N/A | N/A | Missing baseline_metrics.json |")

    # Sreenidhi LSTM
    if has_lstm:
        t_mae = float(lstm_data["test_mae"]) if "test_mae" in lstm_data else "N/A"
        t_rmse = float(lstm_data["test_rmse"]) if "test_rmse" in lstm_data else "N/A"
        t_r2 = float(lstm_data["test_r2"]) if "test_r2" in lstm_data else "N/A"
        print(f"| Single-Timestep LSTM (Sreenidhi) | {t_mae:.4f} | {t_rmse:.4f} | {t_r2:.4f} | Frozen test set (1x38 shape, no temporal claim) |")
    else:
        print("| Single-Timestep LSTM (Sreenidhi) | N/A | N/A | N/A | Missing Sreenidhi_LSTM_run_log.csv |")

    # Older LSTM reference
    print("| Older LSTM (Unfrozen Run) | 4099.0000 | 4275.0000 | -11.3700 | non-frozen data, not comparable |")
    
    # Federated LSTM
    print("| Federated LSTM (FedAvg) | N/A | N/A | N/A | Multi-round FL experiment pending |")

    print("\nTable C2: VALIDATION Split Evaluation (Separate Labeled Table)")
    print("| Model Architecture | Val MAE (kg/ha) | Val RMSE (kg/ha) | Val R² | Notes |")
    print("|---|---:|---:|---:|---|")
    if has_baseline:
        vm = baseline_data.get("validation_metrics", {})
        print(f"| Random Forest (Official) | {vm.get('MAE', 'N/A'):.4f} | {vm.get('RMSE', 'N/A'):.4f} | {vm.get('R2', 'N/A'):.4f} | Frozen val set (75 samples) |")
    if has_lstm:
        v_mae = float(lstm_data["val_mae_at_best_epoch"]) if "val_mae_at_best_epoch" in lstm_data else "N/A"
        v_mse = float(lstm_data["val_mse_at_best_epoch"]) if "val_mse_at_best_epoch" in lstm_data else None
        v_rmse = math.sqrt(v_mse) if v_mse is not None else "N/A"
        print(f"| Single-Timestep LSTM (Sreenidhi) | {v_mae:.4f} | {v_rmse:.4f} | N/A | Frozen val set (best epoch {lstm_data.get('best_epoch','100')}) |")

    # 6. Generate Round Table (Task B)
    print("\n--- TASK B: FEDERATED ROUND TABLE ---")
    if not has_fl_rounds:
        print("NOTICE: Multi-round Flower experiment results are NOT present.")
        print("All per-round metrics (Round Time, Encryption/Decryption, FedAvg Time, Update Size, Client Training Time) are reported as N/A.")
    
    # 7. Generate Graphs (PNG)
    print("\n--- GENERATING GRAPHS ---")
    
    # Graph 1 & 2: Model vs MAE, Model vs RMSE (backed by real test metrics)
    if has_baseline and has_lstm:
        models = ["Random Forest", "Sreenidhi LSTM"]
        maes = [baseline_data["test_metrics"]["MAE"], float(lstm_data["test_mae"])]
        rmses = [baseline_data["test_metrics"]["RMSE"], float(lstm_data["test_rmse"])]

        # Model vs MAE
        plt.figure(figsize=(7, 5))
        bars = plt.bar(models, maes, color=['#2b5c8f', '#d95f02'])
        plt.title("Model vs Test MAE (Lower is Better)")
        plt.ylabel("MAE (kg/ha)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{yval:.1f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(results_dir / "model_vs_mae.png", dpi=300)
        plt.close()
        print(f"[+] Saved: {results_dir / 'model_vs_mae.png'}")

        # Model vs RMSE
        plt.figure(figsize=(7, 5))
        bars = plt.bar(models, rmses, color=['#2b5c8f', '#d95f02'])
        plt.title("Model vs Test RMSE (Lower is Better)")
        plt.ylabel("RMSE (kg/ha)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{yval:.1f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(results_dir / "model_vs_rmse.png", dpi=300)
        plt.close()
        print(f"[+] Saved: {results_dir / 'model_vs_rmse.png'}")

    # Graph 3: Encryption & Decryption Overhead across Benchmark Runs
    if enc_rows:
        runs = [int(r["run"]) for r in enc_rows]
        enc_times = [float(r["encryption_time_ms"]) for r in enc_rows]
        dec_times = [float(r["decryption_time_ms"]) for r in enc_rows]

        plt.figure(figsize=(9, 5))
        plt.plot(runs, enc_times, marker='o', label='Encryption Time (ms)', color='#1b9e77')
        plt.plot(runs, dec_times, marker='s', label='Decryption Time (ms)', color='#7570b3')
        plt.title("AES-256 Encryption & Decryption Overhead per Benchmark Run")
        plt.xlabel("Benchmark Run #")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(results_dir / "round_vs_encryption_overhead.png", dpi=300)
        plt.close()
        print(f"[+] Saved: {results_dir / 'round_vs_encryption_overhead.png'}")

    # Round vs MAE, Round vs RMSE, Round vs Total Round Time
    if not has_fl_rounds:
        print("[!] Round vs MAE, Round vs RMSE, and Round vs Total Round Time graphs skipped: multi-round FL data missing.")
        print("[!] Script completed safely without generating fake/mock data.")

if __name__ == "__main__":
    main()
