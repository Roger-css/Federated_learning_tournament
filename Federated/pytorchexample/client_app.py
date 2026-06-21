import json
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.client import ClientApp, NumPyClient

try:
    from config import CLIENT_MAP
except ImportError:
    from pytorchexample.config import CLIENT_MAP

try:
    from task import (
        DEVICE, GlobalModel, build_sensor_map, evaluate,
        load_all_clients_data, load_client_data, prepare_client_data,
    )
except ImportError:
    from pytorchexample.task import (
        DEVICE, GlobalModel, build_sensor_map, evaluate,
        load_all_clients_data, load_client_data, prepare_client_data,
    )

sensor_map = None

class FlowerClient(NumPyClient):
    def __init__(self, model, tr_loader, tr_eval, te_loader, class_w,
                 sensor_mask, client_id):
        self.model       = model.to(DEVICE)
        self.tr_loader   = tr_loader
        self.tr_eval     = tr_eval
        self.te_loader   = te_loader
        self.class_w     = class_w
        self.sensor_mask = sensor_mask
        self.client_id   = client_id

    def get_parameters(self, config):
        weights = self.model.get_weights()
        payload = {"weights": weights}
        arr = np.frombuffer(pickle.dumps(payload), dtype=np.uint8).copy()
        return [arr]

    def set_parameters(self, parameters):
        arr = parameters[0]
        payload = pickle.loads(bytes(arr.tobytes()))
        self.model.load_weights(payload["weights"])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", 5))

        best_res, _ = train_model(
            self.model, self.tr_loader, self.tr_eval, self.te_loader,
            self.class_w, self.sensor_mask,
            n_epochs=local_epochs, lr=float(config.get("lr", 1e-4)),
            use_early_stop=False, client_id=self.client_id,
        )
        metrics = {
            "train_f1":  float(best_res["train"]["f1"]),
            "test_f1":   float(best_res["test"]["f1"]),
            "client_id": self.client_id,
        }
        return self.get_parameters(config={}), len(self.tr_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        n = len(self.te_loader.dataset)

        eval_res = evaluate(self.model, self.te_loader, self.sensor_mask)

        self.model.eval()
        logits_list = []
        with torch.no_grad():
            for X, _ in self.te_loader:
                logits_list.append(self.model(X.to(DEVICE), self.sensor_mask).cpu())

        logits      = torch.cat(logits_list, dim=0)
        probs       = F.softmax(logits, dim=1)
        predictions = probs.argmax(dim=1).tolist()
        confidences = probs.max(dim=1).values.tolist()

        metrics = {
            "test_f1":     float(eval_res["f1"]),
            "accuracy":    float(eval_res["accuracy"]),
            "client_id":   self.client_id,
            "predictions": json.dumps(predictions),
            "confidences": json.dumps(confidences),
        }
        return 0.0, n, metrics


def client_fn(cid_or_context):
    global sensor_map
    if sensor_map is None:
        clients_data = load_all_clients_data()
        sensor_map = build_sensor_map(clients_data)

    if hasattr(cid_or_context, "node_config"):
        node_config = cid_or_context.node_config
        partition_id = node_config.get("partition-id", node_config.get("partition_id", 0))
        cid = str(partition_id)
    else:
        cid = str(cid_or_context)

    client_name = CLIENT_MAP[cid]

    data = load_client_data(client_name)
    model = GlobalModel(all_sensors=sensor_map["all_sensors"],
                        window_size=data["window_size"]).to(DEVICE)
    tr_loader, tr_eval, te_loader, class_w, sensor_mask = prepare_client_data(data, sensor_map)

    return FlowerClient(
        model=model, tr_loader=tr_loader, tr_eval=tr_eval,
        te_loader=te_loader, class_w=class_w,
        sensor_mask=sensor_mask, client_id=client_name,
    ).to_client()


try:
    from task import train_model
except ImportError:
    from pytorchexample.task import train_model

app = ClientApp(client_fn=client_fn)
