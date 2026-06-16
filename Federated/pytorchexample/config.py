"""DHSV Fault Detection — FL Configuration."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout (resolved relative to this file — works regardless of CWD)
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent          # pytorchexample/
_FEDERATED    = _HERE.parent                             # Federated/
_PROJECT_ROOT = _FEDERATED.parent                        # project root

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
NUM_ROUNDS   = 10
LOCAL_EPOCHS = 5

# ---------------------------------------------------------------------------
# Paths and Client Map
# ---------------------------------------------------------------------------
DATA_DIR     = str(_PROJECT_ROOT / "data")               # .npz + _sensors.csv files
RESULTS_PATH = str(_FEDERATED / "fl_results.json")       # output JSON

CLIENT_MAP = {
    "0": "client_1",
    "1": "client_2",
    "2": "client_3",
    "3": "client_13",
}
