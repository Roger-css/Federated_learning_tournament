import json
import os
import sys
from typing import Dict

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
import numpy as np
import torch

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

try:
    from config import (
        ALPHA, CLIENT_MAP, DATA_DIR, D_MODEL, FL_EPOCHS, FL_LR,
        LOCAL_EPOCHS, LR, N_HEADS, N_LAYERS, NUM_ROUNDS, RESULTS_PATH,
    )
except ImportError:
    from pytorchexample.config import (
        ALPHA, CLIENT_MAP, DATA_DIR, D_MODEL, FL_EPOCHS, FL_LR,
        LOCAL_EPOCHS, LR, N_HEADS, N_LAYERS, NUM_ROUNDS, RESULTS_PATH,
    )

try:
    from task import (
        DEVICE, GlobalModel, build_sensor_map, evaluate, evaluate_detailed,
        fedavg_global_equal_stable, load_all_clients_data, load_client_data,
        prepare_client_data, train_model,
    )
except ImportError:
    from pytorchexample.task import (
        DEVICE, GlobalModel, build_sensor_map, evaluate, evaluate_detailed,
        fedavg_global_equal_stable, load_all_clients_data, load_client_data,
        prepare_client_data, train_model,
    )

# ---------------------------------------------------------------------------
# Backend API reporting
# ---------------------------------------------------------------------------

FL_API_BASE = "http://localhost:8080/api/fl"

def report_local_baseline_to_backend(local_results: dict):
    payload = {
        "clients": [
            {
                "clientId": cid,
                "trainF1": r["train_f1"],
                "testF1": r["test_f1"],
                "accuracy": r["accuracy"],
                "numExamples": r["num_examples"],
            }
            for cid, r in local_results.items()
        ]
    }
    try:
        resp = requests.post(f"{FL_API_BASE}/local-baseline", json=payload, timeout=5)
        resp.raise_for_status()
        print(f"  [Backend] Local baseline reported successfully (status {resp.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"  [Backend] WARNING: failed to report local baseline to backend: {e}")

def report_round_to_backend(round_number: int, round_results: dict):
    payload = {
        "roundNumber": round_number,
        "clients": [
            {
                "clientId": cid,
                "trainF1": r["train_f1"],
                "testF1": r["test_f1"],
                "accuracy": r["accuracy"],
                "numExamples": r["num_examples"],
            }
            for cid, r in round_results.items()
        ]
    }
    try:
        resp = requests.post(f"{FL_API_BASE}/rounds", json=payload, timeout=5)
        resp.raise_for_status()
        print(f"  [Backend] Round {round_number} reported successfully (status {resp.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"  [Backend] WARNING: failed to report round {round_number} to backend: {e}")

# ---------------------------------------------------------------------------
# ServerApp entrypoint (for flwr run)
# ---------------------------------------------------------------------------

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds   = context.run_config.get("num-server-rounds", NUM_ROUNDS)

    def fit_config(server_round: int) -> Dict:
        return {"local_epochs": FL_EPOCHS, "lr": FL_LR}

    class SimpleFedAvg:
        def __init__(self):
            self.round_metrics = []
            self.best_avg_f1 = 0.0

        def aggregate_fit(self, server_round, results, failures):
            from flwr.common import parameters_to_ndarrays
            import pickle
            payloads = {}
            fit_metrics = {}
            for _, fit_res in results:
                arr = parameters_to_ndarrays(fit_res.parameters)[0]
                payload = pickle.loads(bytes(arr.tobytes()))
                m = fit_res.metrics or {}
                cid = m.get("client_id", "unknown")
                payloads[cid] = payload["weights"]
                fit_metrics[cid] = {
                    "train_f1": float(m.get("train_f1", 0.0)),
                    "test_f1": float(m.get("test_f1", 0.0)),
                    "num_examples": fit_res.num_examples,
                }
            clients_data = load_all_clients_data()
            global_weights = fedavg_global_equal_stable(
                payloads, clients_data, None, alpha=1.0
            )
            agg_payload = {"weights": global_weights}
            agg_arr = np.frombuffer(pickle.dumps(agg_payload), dtype=np.uint8).copy()
            from flwr.common import ndarrays_to_parameters
            parameters = ndarrays_to_parameters([agg_arr])

            avg_f1 = sum(v["test_f1"] for v in fit_metrics.values()) / max(len(fit_metrics), 1)
            improvement = avg_f1 - self.best_avg_f1
            self.best_avg_f1 = max(self.best_avg_f1, avg_f1)

            print(f"\n{'='*65}")
            print(f"  Round {server_round}/{num_rounds} — Fit Results")
            print(f"  {'Client':<14} {'Train F1':>10} {'Test F1':>10} {'Samples':>10}")
            print(f"  {'─'*46}")
            for cid, m in sorted(fit_metrics.items()):
                print(f"  {cid:<14} {m['train_f1']:>10.4f} {m['test_f1']:>10.4f} {m['num_examples']:>10,}")
            print(f"  {'─'*46}")
            print(f"  {'Avg test F1':<14} {avg_f1:>10.4f}   improvement: {improvement:+.4f}")
            print(f"{'='*65}")

            self.round_metrics.append({"round": server_round, "fit": fit_metrics, "evaluate": {}})
            return parameters, {}

        def aggregate_evaluate(self, server_round, results, failures):
            if not results:
                return None, {}
            eval_metrics = {}
            for _, eval_res in results:
                m = eval_res.metrics or {}
                cid = m.get("client_id", "unknown")
                raw_preds = m.get("predictions", "[]")
                raw_confs = m.get("confidences", "[]")
                eval_metrics[cid] = {
                    "test_f1": float(m.get("test_f1", 0.0)),
                    "accuracy": float(m.get("accuracy", 0.0)),
                    "num_examples": eval_res.num_examples,
                    "predictions": json.loads(raw_preds) if isinstance(raw_preds, str) else raw_preds,
                    "confidences": json.loads(raw_confs) if isinstance(raw_confs, str) else raw_confs,
                }
            for entry in self.round_metrics:
                if entry["round"] == server_round:
                    entry["evaluate"] = eval_metrics
                    break
            try:
                write_results(self.round_metrics, RESULTS_PATH)
            except Exception as e:
                print(f"Warning: Failed to auto-write results: {e}")

            print(f"\n{'='*65}")
            print(f"  Round {server_round}/{num_rounds} — Evaluate Results")
            print(f"  {'Client':<14} {'Test F1':>10} {'Accuracy':>10} {'Samples':>10}")
            print(f"  {'─'*46}")
            for cid, m in sorted(eval_metrics.items()):
                print(f"  {cid:<14} {m['test_f1']:>10.4f} {m['accuracy']:>10.4f} {m['num_examples']:>10,}")
            print(f"{'='*65}\n")
            return 0.0, {}

    strategy = SimpleFedAvg()
    strategy.aggregate_fit = strategy.aggregate_fit
    strategy.aggregate_evaluate = strategy.aggregate_evaluate
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)


# ---------------------------------------------------------------------------
# Results writer (shared between Flower path and direct simulation)
# ---------------------------------------------------------------------------

def write_results(round_metrics, path, local_baseline=None):
    rounds_out = []
    for entry in round_metrics:
        fit_m  = entry.get("fit", {})
        eval_m = entry.get("evaluate", {})
        all_cids = sorted(set(fit_m.keys()) | set(eval_m.keys()))
        clients = []
        for cid in all_cids:
            fm = fit_m.get(cid, {})
            em = eval_m.get(cid, {})
            clients.append({
                "client_id": cid,
                "train_f1": fm.get("train_f1", 0.0),
                "test_f1": em.get("test_f1", fm.get("test_f1", 0.0)),
                "accuracy": em.get("accuracy", 0.0),
                "num_examples": fm.get("num_examples", em.get("num_examples", 0)),
                "predictions": em.get("predictions", []),
                "confidences": em.get("confidences", []),
            })
        rounds_out.append({"round": entry["round"], "clients": clients})

    output = {"rounds": rounds_out}
    if local_baseline is not None:
        output["local_baseline"] = local_baseline

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {path}")


# ---------------------------------------------------------------------------
# Simulation entry point — Phase 1 (full local training) + Phase 2 (FL rounds)
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  DHSV Fault Detection — Two-Phase Federated Learning")
    print("=" * 65)
    print(f"  Clients      : {len(CLIENT_MAP)}  ({', '.join(CLIENT_MAP.values())})")
    print(f"  Data dir     : {DATA_DIR}")
    print(f"  Results path : {RESULTS_PATH}")
    print(f"  Phase 1      : {LOCAL_EPOCHS} local epochs, LR={LR}")
    print(f"  Phase 2      : {NUM_ROUNDS} rounds x {FL_EPOCHS} epochs, LR={FL_LR}, alpha={ALPHA}")
    print("=" * 65 + "\n")

    # -- Pre-load all client data once ------------------------------------
    clients_data = load_all_clients_data()

    # -- Build sensor map from ALL clients --------------------------------
    sensor_map = build_sensor_map(clients_data)
    all_sensors = sensor_map["all_sensors"]
    print(f"  Global sensor universe ({len(all_sensors)} sensors): {all_sensors}\n")

    for cid, data in clients_data.items():
        print(f"  {cid}: sensors={data['sensor_names']} ({len(data['sensor_names'])}/{len(all_sensors)})")

    # =====================================================================
    # Phase 1 — Full local training (no aggregation, no weight sharing)
    # =====================================================================
    print(f"\n{'#'*65}")
    print(f"  {'#'*61}")
    print(f"  {'PHASE 1':^61}")
    print(f"  {'Full Local Training — each client trains independently':^61}")
    print(f"  {'#'*61}")
    print(f"{'#'*65}")

    local_results = {}
    client_loaders = {}

    for client_id in sorted(clients_data.keys()):
        data = clients_data[client_id]
        model = GlobalModel(
            all_sensors=all_sensors,
            window_size=data["window_size"],
        ).to(DEVICE)

        tr_loader, tr_eval, te_loader, class_w, sensor_mask = prepare_client_data(data, sensor_map)

        res, _ = train_model(
            model, tr_loader, tr_eval, te_loader, class_w, sensor_mask,
            n_epochs=LOCAL_EPOCHS, lr=LR,
            client_id=f"{client_id} (Phase 1)",
        )

        client_loaders[client_id] = (tr_loader, tr_eval, te_loader, class_w, sensor_mask, model)
        local_results[client_id] = res

    local_avg_f1 = np.mean([v["test"]["f1"] for v in local_results.values()])
    print(f"\n{'='*65}")
    print(f"  Phase 1 Complete — Local Training Results")
    print(f"  {'Client':<14} {'Train F1':>10} {'Test F1':>10}")
    print(f"  {'─'*36}")
    for cid in sorted(local_results.keys()):
        r = local_results[cid]
        print(f"  {cid:<14} {r['train']['f1']:>10.4f} {r['test']['f1']:>10.4f}")
    print(f"  {'─'*36}")
    print(f"  {'Avg Test F1':<24} {local_avg_f1:>10.4f}")
    print(f"{'='*65}")

    # =====================================================================
    # Phase 2 — FL rounds (lightweight fine-tuning + aggregation)
    # =====================================================================
    print(f"\n{'#'*65}")
    print(f"  {'#'*61}")
    print(f"  {'PHASE 2':^61}")
    print(f"  {'Federated Fine-Tuning — {NUM_ROUNDS} rounds':^61}")
    print(f"  {'#'*61}")
    print(f"{'#'*65}")

    global_weights = None
    round_metrics = []

    for fl_round in range(1, NUM_ROUNDS + 1):
        round_payloads = {}
        round_results = {}

        for client_id in sorted(clients_data.keys()):
            tr_loader, tr_eval, te_loader, class_w, sensor_mask, model = client_loaders[client_id]

            if global_weights is not None:
                model.load_weights(global_weights)

            res, _ = train_model(
                model, tr_loader, tr_eval, te_loader, class_w, sensor_mask,
                n_epochs=FL_EPOCHS, lr=FL_LR,
                use_early_stop=False,
                client_id=f"{client_id} (FL Round {fl_round})",
            )

            client_loaders[client_id] = (tr_loader, tr_eval, te_loader, class_w, sensor_mask, model)
            round_payloads[client_id] = model.get_weights()
            round_results[client_id] = res

        global_weights = fedavg_global_equal_stable(
            round_payloads, clients_data, global_weights, alpha=ALPHA,
        )

        # -- Log round metrics --------------------------------------------
        fit_metrics = {}
        eval_metrics = {}
        for cid, res in round_results.items():
            fit_metrics[cid] = {
                "train_f1": float(res["train"]["f1"]),
                "test_f1":  float(res["test"]["f1"]),
            }
            tr_loader, tr_eval, te_loader, class_w, sensor_mask, model = client_loaders[cid]
            ev = evaluate_detailed(model, te_loader, sensor_mask)
            eval_metrics[cid] = ev

        avg_f1 = np.mean([v["test"]["f1"] for v in round_results.values()])

        print(f"\n{'='*65}")
        print(f"  FL Round {fl_round}/{NUM_ROUNDS} — Results")
        print(f"  {'Client':<14} {'Train F1':>10} {'Test F1':>10} {'Accuracy':>10} {'Samples':>8} {'Preds':>6} {'Confs':>6}")
        print(f"  {'─'*66}")
        for cid in sorted(fit_metrics.keys()):
            fm = fit_metrics[cid]
            em = eval_metrics[cid]
            print(f"  {cid:<14} {fm['train_f1']:>10.4f} {fm['test_f1']:>10.4f} {em['accuracy']:>10.4f} {em['num_examples']:>8} {len(em['predictions']):>6} {len(em['confidences']):>6}")
        print(f"  {'─'*66}")
        print(f"  {'Avg test F1':<24} {avg_f1:>10.4f}")
        print(f"{'='*65}")

        # Build merged per-client dict for backend reporting (no predictions/confidences)
        round_report = {}
        for cid in sorted(fit_metrics.keys()):
            fm = fit_metrics[cid]
            em = eval_metrics[cid]
            round_report[cid] = {
                "train_f1": fm["train_f1"],
                "test_f1": em["test_f1"],
                "accuracy": em["accuracy"],
                "num_examples": em["num_examples"],
            }
        report_round_to_backend(fl_round, round_report)

        round_metrics.append({
            "round": fl_round,
            "fit": fit_metrics,
            "evaluate": eval_metrics,
        })

    # =====================================================================
    # Final report
    # =====================================================================
    print(f"\n{'#'*65}")
    print(f"  {'FINAL COMPARISON':^61}")
    print(f"{'#'*65}")
    print(f"  {'Client':<14} {'Local Test F1':>14} {'FL Test F1':>14} {'Change':>10}")
    print(f"  {'─'*54}")

    final_fl_results = round_metrics[-1]["fit"]
    total_local = 0.0
    total_fl = 0.0
    for cid in sorted(local_results.keys()):
        local_f1 = local_results[cid]["test"]["f1"]
        fl_f1 = final_fl_results.get(cid, {}).get("test_f1", 0.0)
        change = fl_f1 - local_f1
        total_local += local_f1
        total_fl += fl_f1
        print(f"  {cid:<14} {local_f1:>14.4f} {fl_f1:>14.4f} {change:>+10.4f}")

    n = len(local_results)
    print(f"  {'─'*54}")
    print(f"  {'Average':<14} {total_local/n:>14.4f} {total_fl/n:>14.4f} {(total_fl-total_local)/n:>+10.4f}")
    print(f"{'#'*65}")

    # -- Build local baseline for results JSON ----------------------------
    local_baseline = {}
    for cid in sorted(local_results.keys()):
        tr_loader, tr_eval, te_loader, class_w, sensor_mask, model = client_loaders[cid]
        ev = evaluate_detailed(model, te_loader, sensor_mask)
        r = local_results[cid]
        local_baseline[cid] = {
            "test_f1":      float(r["test"]["f1"]),
            "train_f1":     float(r["train"]["f1"]),
            "accuracy":     float(ev["accuracy"]),
            "predictions":  ev["predictions"],
            "confidences":  ev["confidences"],
            "num_examples": ev["num_examples"],
        }

    report_local_baseline_to_backend(local_baseline)
    write_results(round_metrics, RESULTS_PATH, local_baseline=local_baseline)


if __name__ == "__main__":
    main()
