"""DHSV Fault Detection — Flower Server: Strategy + Simulation entry point.

Run from anywhere:
    python -m pytorchexample.server_app
or
    python Federated/pytorchexample/server_app.py
"""

import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Force UTF-8 stdout/stderr so Windows cp1252 consoles don't choke on the
# box-drawing characters (─, etc.) printed by model.fit().
# ---------------------------------------------------------------------------
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math

import numpy as np

from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import start_simulation
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from client_app import FlowerClient, client_fn
from config import CLIENT_MAP, DATA_DIR, LOCAL_EPOCHS, NUM_ROUNDS, RESULTS_PATH
from task import DEVICE, iTransformerGlobal

# ---------------------------------------------------------------------------
# Custom strategy
# ---------------------------------------------------------------------------


class DHSVFedAvg(FedAvg):
    """FedAvg variant for heterogeneous sensor payloads.

    Each client transmits a pickled dict with:
        weights      – shared model parameters (dict[str, Tensor])
        sensor_names – ordered list of this client's sensor names
        n_sensors    – number of sensors

    Aggregation:
    • Deserialize each client payload.
    • Weighted average (by num_examples) for every key present in ALL
      clients and with identical shapes.  pos_embed is never present
      (model.get_shared_params() already excludes it).
    • Re-serialize the aggregated payload and return as Parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._round_metrics: List[Dict] = []   # one entry per round
        self._best_avg_f1: float = 0.0         # tracks global improvement

    # ------------------------------------------------------------------
    # aggregate_fit
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:

        if not results:
            return None, {}

        # --- Deserialize each client payload ----------------------------
        payloads_with_n: List[Tuple[Dict, int]] = []
        for _, fit_res in results:
            arr     = parameters_to_ndarrays(fit_res.parameters)[0]
            payload = pickle.loads(bytes(arr.tobytes()))
            payloads_with_n.append((payload, fit_res.num_examples))

        total = sum(n for _, n in payloads_with_n)

        # --- Log-scaled weights (reduce client_2's dominance) ----------
        log_weights = [math.log(n + 1) for _, n in payloads_with_n]
        log_total = sum(log_weights)
        normalized_weights = [w / log_total for w in log_weights]

        print(f"[Aggregation] Linear weights: {[n/total for _, n in payloads_with_n]}")
        print(f"[Aggregation] Log-scaled weights: {normalized_weights}")

        # --- Find keys common to every client ---------------------------
        common_keys = set.intersection(
            *[set(p["weights"].keys()) for p, _ in payloads_with_n]
        )
        print(f"\n[Aggregation] Common keys count: {len(common_keys)}")
        print(f"[Aggregation] Keys excluded (not in all clients): {set.union(*[set(p['weights'].keys()) for p, _ in payloads_with_n]) - common_keys}")
        # --- Weighted average (matching shapes only) --------------------
        aggregated: Dict = {}
        for key in common_keys:
            shapes = [p["weights"][key].shape for p, _ in payloads_with_n]
            if len(set(shapes)) == 1:                      # shapes match
                aggregated[key] = sum(
                    p["weights"][key].cpu().clone().float() * normalized_weights[i]
                    for i, (p, n) in enumerate(payloads_with_n)
                )

        # --- Build and serialize aggregated payload ---------------------
        agg_payload = {
            "weights"     : aggregated,
            "sensor_names": payloads_with_n[0][0]["sensor_names"],
            "n_sensors"   : payloads_with_n[0][0]["n_sensors"],
        }
        agg_arr    = np.frombuffer(pickle.dumps(agg_payload), dtype=np.uint8).copy()
        parameters = ndarrays_to_parameters([agg_arr])

        # --- Collect fit metrics per client -----------------------------
        fit_metrics: Dict[str, Dict] = {}
        for _, fit_res in results:
            m   = fit_res.metrics or {}
            cid = m.get("client_id", "unknown")
            fit_metrics[cid] = {
                "train_f1"   : float(m.get("train_f1", 0.0)),
                "test_f1"    : float(m.get("test_f1",  0.0)),
                "num_examples": fit_res.num_examples,
            }

        # --- Console logging --------------------------------------------
        avg_f1      = sum(v["test_f1"] for v in fit_metrics.values()) / max(len(fit_metrics), 1)
        improvement = avg_f1 - self._best_avg_f1
        self._best_avg_f1 = max(self._best_avg_f1, avg_f1)

        print(f"\n{'='*65}")
        print(f"  Round {server_round}/{NUM_ROUNDS} — Fit Results")
        print(f"  {'Client':<14} {'Train F1':>10} {'Test F1':>10} {'Samples':>10}")
        print(f"  {'─'*46}")
        for cid, m in sorted(fit_metrics.items()):
            print(
                f"  {cid:<14} {m['train_f1']:>10.4f} "
                f"{m['test_f1']:>10.4f} {m['num_examples']:>10,}"
            )
        print(f"  {'─'*46}")
        print(f"  {'Avg test F1':<14} {avg_f1:>10.4f}   improvement: {improvement:+.4f}")
        print(f"{'='*65}")

        # --- Store for JSON output (evaluate will merge later) ----------
        self._round_metrics.append({
            "round"   : server_round,
            "fit"     : fit_metrics,
            "evaluate": {},
        })

        return parameters, {}

    # ------------------------------------------------------------------
    # aggregate_evaluate
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict]:

        if not results:
            return None, {}

        # --- Collect evaluate metrics per client ------------------------
        eval_metrics: Dict[str, Dict] = {}
        for _, eval_res in results:
            m           = eval_res.metrics or {}
            cid         = m.get("client_id", "unknown")
            raw_preds   = m.get("predictions", "[]")
            raw_confs   = m.get("confidences", "[]")
            eval_metrics[cid] = {
                "test_f1"    : float(m.get("test_f1",  0.0)),
                "accuracy"   : float(m.get("accuracy", 0.0)),
                "num_examples": eval_res.num_examples,
                "predictions": json.loads(raw_preds) if isinstance(raw_preds, str) else raw_preds,
                "confidences": json.loads(raw_confs) if isinstance(raw_confs, str) else raw_confs,
            }

        # --- Merge into the matching round entry ------------------------
        for entry in self._round_metrics:
            if entry["round"] == server_round:
                entry["evaluate"] = eval_metrics
                break

        try:
            self.write_results(RESULTS_PATH)
        except Exception as e:
            print(f"Warning: Failed to auto-write results during round {server_round}: {e}")

        # --- Console logging --------------------------------------------
        print(f"\n{'='*65}")
        print(f"  Round {server_round}/{NUM_ROUNDS} — Evaluate Results")
        print(f"  {'Client':<14} {'Test F1':>10} {'Accuracy':>10} {'Samples':>10}")
        print(f"  {'─'*46}")
        for cid, m in sorted(eval_metrics.items()):
            print(
                f"  {cid:<14} {m['test_f1']:>10.4f} "
                f"{m['accuracy']:>10.4f} {m['num_examples']:>10,}"
            )
        print(f"{'='*65}\n")

        # Loss placeholder (0.0 used by all clients)
        return 0.0, {}

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def write_results(self, path: str) -> None:
        """Merge fit + evaluate metrics per client per round → fl_results.json."""
        rounds_out = []
        for entry in self._round_metrics:
            fit_m  = entry.get("fit",      {})
            eval_m = entry.get("evaluate", {})
            all_cids = sorted(set(fit_m.keys()) | set(eval_m.keys()))

            clients = []
            for cid in all_cids:
                fm = fit_m.get(cid,  {})
                em = eval_m.get(cid, {})
                clients.append({
                    "client_id"  : cid,
                    "train_f1"   : fm.get("train_f1",    0.0),
                    "test_f1"    : em.get("test_f1",     fm.get("test_f1", 0.0)),
                    "accuracy"   : em.get("accuracy",    0.0),
                    "num_examples": fm.get("num_examples", em.get("num_examples", 0)),
                    "predictions": em.get("predictions", []),
                    "confidences": em.get("confidences", []),
                })

            rounds_out.append({"round": entry["round"], "clients": clients})

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rounds": rounds_out}, f, indent=2)
        print(f"\nResults written to: {path}")


# ---------------------------------------------------------------------------
# ServerApp entrypoint & server components factory
# ---------------------------------------------------------------------------


def server_fn(context: Context) -> ServerAppComponents:
    """Construct strategy and config for the ServerApp."""
    num_rounds   = context.run_config.get("num-server-rounds", NUM_ROUNDS)
    local_epochs = context.run_config.get("local-epochs", LOCAL_EPOCHS)

    def fit_config(server_round: int) -> Dict:
        return {"local_epochs": local_epochs}

    strategy = DHSVFedAvg(
        fraction_fit          = 1.0,
        fraction_evaluate     = 1.0,
        min_fit_clients       = 4,
        min_evaluate_clients  = 4,
        min_available_clients = 4,
        on_fit_config_fn      = fit_config,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# App entrypoint for flwr run
app = ServerApp(server_fn=server_fn)


# ---------------------------------------------------------------------------
# Simulation entry point
# ---------------------------------------------------------------------------


def load_client(name: str) -> FlowerClient:
    """Load data and build a raw FlowerClient (without .to_client() wrapper)."""
    data    = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    import pandas as pd
    sensors_csv  = os.path.join(DATA_DIR, f"{name}_sensors.csv")
    sensor_names = pd.read_csv(sensors_csv)["sensor_name"].tolist()
    window_size  = X_train.shape[1]

    model = iTransformerGlobal(sensor_names=sensor_names, window_size=window_size).to(DEVICE)
    return FlowerClient(model, X_train, y_train, X_test, y_test, name)


def main() -> None:
    print("\n" + "="*65)
    print("  DHSV Fault Detection — Federated Learning Simulation")
    print("="*65)
    print(f"  Rounds       : {NUM_ROUNDS}")
    print(f"  Local epochs : {LOCAL_EPOCHS} per round")
    print(f"  Clients      : {len(CLIENT_MAP)}  ({', '.join(CLIENT_MAP.values())})")
    print(f"  Data dir     : {DATA_DIR}")
    print(f"  Results path : {RESULTS_PATH}")
    print("="*65 + "\n")

    def fit_config(server_round: int) -> Dict:
        return {"local_epochs": LOCAL_EPOCHS}

    strategy = DHSVFedAvg(
        fraction_fit          = 1.0,
        fraction_evaluate     = 1.0,
        min_fit_clients       = 4,
        min_evaluate_clients  = 4,
        min_available_clients = 4,
        on_fit_config_fn      = fit_config,
    )

    start_simulation(
        client_fn         = client_fn,
        num_clients       = len(CLIENT_MAP),
        config            = ServerConfig(num_rounds=NUM_ROUNDS),
        strategy          = strategy,
        client_resources  = {"num_cpus": 1, "num_gpus": 0.25},
    )

    strategy.write_results(RESULTS_PATH)


if __name__ == "__main__":
    main()
