from pathlib import Path

_HERE         = Path(__file__).resolve().parent
_FEDERATED    = _HERE.parent
_PROJECT_ROOT = _FEDERATED.parent

NUM_ROUNDS   = 10
LOCAL_EPOCHS = 50
FL_EPOCHS    = 15
FL_LR        = 1e-4
ALPHA        = 0.7
D_MODEL      = 32
N_HEADS      = 2
N_LAYERS     = 1
DROPOUT      = 0.3
LR           = 5e-4
WEIGHT_DECAY = 0.01
PATIENCE     = 20
MIN_DELTA    = 0.003

DATA_DIR     = str(_PROJECT_ROOT / "data")
RESULTS_PATH = str(_FEDERATED / "fl_results.json")

CLIENT_MAP = {
    "0": "client_1",
    "1": "client_2",
    "2": "client_3",
    "3": "client_13",
}
