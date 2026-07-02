# Project Overview — Full Documentation (Excluding Neural Network Logic)

This document describes every file in the project **except** the neural network model architectures (iTransformer, TransformerBlock, GlobalModel, FocalLoss, EarlyStopping, forward-pass logic, weight initialization, etc.). Everything else — FL logic, backend, frontend, data, infrastructure, configs — is documented below.

---

## Project Structure

```
Federated_learning_tournament/
├── .git/
├── .github/workflows/          # GitHub Actions (empty)
├── .gitignore
├── .venv/                      # Python virtual environment
├── data/                       # Oil well sensor datasets
├── Federated/                  # Flower FL Python project
│   ├── fl_results.json         # Training results (20 rounds)
│   ├── pyproject.toml          # Project metadata & dependencies
│   ├── README.md               # Flower quickstart template
│   └── pytorchexample/         # FL client/server/task logic
├── FL_backend/                 # C# .NET 10 Web API + SignalR
├── FL_frontend/                # Next.js 15 dashboard
├── fllld.ipynb                 # Colab notebook (FL pipeline)
└── Untitled90.ipynb            # Colab notebook (FL pipeline, Arabic)
```

---

## 1. Root-Level

### `.gitignore` (535 lines)

Comprehensive ignore rules covering:

- **Python**: `__pycache__/`, `*.pyc`, `*.so`, `env/`, `venv/`, `.venv/`
- **PyTorch/ML**: `*.pt`, `*.pth`, `*.h5`, `*.ckpt`, `*.weights`, `model_checkpoints/`
- **Data**: `/data/`, `*.zip`, `*.csv`, `*.npy`, `*.npz`, `fl_results.*`, `datasets/`
- **FL logs**: `flower_logs/`, `*.log`, `state.db`, `ray_results/`, `.ray/`
- **Jupyter**: `.ipynb_checkpoints/`, `.matplotlib/`, `*.png`, `*.jpg`, `*.pdf`
- **IDE/OS**: `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`, `.env`
- **.NET**: build outputs, NuGet, VS cache (extensive section from `dotnet new gitignore`)

### `.github/workflows/` — **Empty**

No CI/CD workflows defined yet.

### `.venv/`

Python virtual environment (gitignored).

---

## 2. `data/` — Oil Well Sensor Dataset

Preprocessed time-series data from real oil wells for DHSV (Down Hole Safety Valve) fault detection into 3 classes: Normal (0), Steady-DHSV (1), Transient-DHSV (2).

### File Inventory

| File                                  | Description                                                   |
| ------------------------------------- | ------------------------------------------------------------- |
| `common_sensors.csv`                  | 3 sensors common to all clients: `P-ANULAR`, `P-TPT`, `T-TPT` |
| `wells_4clients_with_sensors (1).zip` | Original raw data ZIP archive                                 |
| `client_{1,2,3,13}.npz`               | NumPy archives: `X_train`, `X_test`, `y_train`, `y_test`      |
| `client_{1,2,3,13}_sensors.csv`       | Sensor index-to-name mapping per client                       |
| `client_{1,2,3,13}_README.txt`        | Per-client metadata in Arabic                                 |

### Per-Client Summary

| Client    | Well                   | Sessions | Sensors                                                                  | Train N/S/T             | Test N/S/T             |
| --------- | ---------------------- | -------- | ------------------------------------------------------------------------ | ----------------------- | ---------------------- |
| client_1  | WELL-00003 (2014+2018) | 2        | 5: P-ANULAR, P-MON-CKP, P-PDG, P-TPT, T-TPT                              | 4,786 / 1,961 / 3,780   | 2,041 / 844 / 1,619    |
| client_2  | WELL-00011 (13 dates)  | 13       | 5: P-ANULAR, P-PDG, P-TPT, T-PDG, T-TPT                                  | 16,499 / 8,371 / 30,933 | 7,013 / 3,576 / 13,282 |
| client_3  | WELL-00012 (2 dates)   | 2        | 8: ABER-CKP, ESTADO-PXO, P-ANULAR, P-MON-CKP, P-PDG, P-TPT, T-PDG, T-TPT | 5,107 / 1,398 / 861     | 2,185 / 596 / 370      |
| client_13 | WELL-00013 (2017)      | 1        | 3: P-ANULAR, P-TPT, T-TPT                                                | 2,525 / 257 / 2,046     | 1,078 / 111 / 877      |

**Preprocessing:** Window Size = 16, Window Step = 2, Train Ratio = 0.7

### `data/.client_states/`

Local PyTorch model state dict checkpoints saved after Phase 1 (local training):

- `client_1_local.pt`, `client_2_local.pt`, `client_3_local.pt`, `client_13_local.pt`

---

## 3. `Federated/` — Flower FL Python Project

### `pyproject.toml` (31 lines)

- **Package:** `dhsv-fl` v1.0.0 — "DHSV Fault Detection — iTransformer + Flower Federated Learning"
- **Build:** Hatchling (`hatchling.build`)
- **License:** Apache-2.0
- **Dependencies:** `flwr[simulation]>=1.26.0`, `torch==2.8.0`, `torchvision==0.23.0`, `scikit-learn>=1.3.0`, `pandas>=2.0.0`, `numpy>=1.24.0`
- **Flower config:** `num-server-rounds = 20`, `local-epochs = 5` (overridden by code to actual values)
- **Components:** `server_app = pytorchexample.server_app:app`, `client_app = pytorchexample.client_app:app`

### `README.md` (75 lines)

Generated template from `flwr new @flwrlabs/quickstart-pytorch`. Instructions for running with Simulation Engine (`flwr run .`) or Deployment Engine, with links to Flower docs.

### `fl_results.json` (~694K lines)

JSON output of FL training containing per-round (20 rounds) per-client metrics:

- `train_f1`, `test_f1`, `accuracy`, `num_examples`
- Full `predictions` array (class predictions for every test sample)
- Used by the backend API when no live SignalR data is available

---

### 3a. `pytorchexample/` — FL Logic (No Neural Network Details)

#### `__init__.py`

Empty package marker (`"""pytorchexample."""`)

#### `config.py` (29 lines)

Central configuration:

- `NUM_ROUNDS = 10`, `LOCAL_EPOCHS = 30`, `FL_EPOCHS = 15`, `FL_LR = 1e-4`
- `ALPHA = 0.7` (FedAvg momentum), `SEED = 42`
- `DATA_DIR` = `.../data/`, `RESULTS_PATH` = `.../Federated/fl_results.json`
- `CLIENT_MAP` = `{"0": "client_1", "1": "client_2", "2": "client_3"}` (3 clients in use)
- Neural network hyperparams: `D_MODEL=16`, `N_HEADS=2`, `N_LAYERS=1`, `DROPOUT=0.5`, `LR=5e-4`, `WEIGHT_DECAY=0.05`, `PATIENCE=20`, `MIN_DELTA=0.003`

#### `task.py` (462 lines) — FL Logic

Key functions (excluding NN model class details):

| Function                                                                             | Purpose                                                                                                                          |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `load_client_data(client_name)`                                                      | Loads `.npz` + `_sensors.csv` for a client; returns X_train, X_test, y_train, y_test, sensor list, class counts                  |
| `load_all_clients_data()`                                                            | Loads all clients via `CLIENT_MAP`, builds `sensor_map` (union of all sensors across clients)                                    |
| `build_sensor_map(all_clients_data)`                                                 | Creates global sensor universe with index mapping; saves `common_sensors.csv`                                                    |
| `make_loaders(X, y, batch_size, ...)`                                                | Creates DataLoaders with `WeightedRandomSampler` for class imbalance                                                             |
| `prepare_client_data(client_name, data)`                                             | Zeros-pads sensor data to full global sensor universe, builds `sensor_mask` for missing sensors                                  |
| `detect_largest_client(all_clients_data)`                                            | Identifies client with most training examples                                                                                    |
| `detect_weak_client(results)`                                                        | Calculates `need_score = data_size_ratio*0.6 + (1-f1)*0.4` to find weakest client                                                |
| `train_model(...)`                                                                   | Training loop with FocalLoss, AdamW, ReduceLROnPlateau, gradient clipping, early stopping                                        |
| `evaluate(model, loader, device)`                                                    | Returns loss, F1, accuracy, predictions, confidences                                                                             |
| `evaluate_detailed(model, loader, device)`                                           | Additional classification report                                                                                                 |
| `fedavg_global_equal_stable(global_model, client_models, client_train_loaders, ...)` | Equal-weight FedAvg with momentum (`alpha`): updates global model as `alpha*global + (1-alpha)*client_avg`, then broadcasts back |

#### `client_app.py` (129 lines) — Flower ClientApp

- `FlowerClient(NumPyClient)` wraps the PyTorch model:
  - `get_parameters()` — returns pickle-serialized state dict
  - `set_parameters(params)` — deserializes and loads state dict
  - `fit(config)` — calls `train_model()` with config hyperparameters
  - `evaluate(config)` — calls `evaluate()`, returns loss + metrics with softmax predictions
- `client_fn(context)` — lazily builds sensor_map from partitioned ID, creates model + loaders, returns `FlowerClient`
- `app = ClientApp(client_fn=client_fn)` — Flower entry point

#### `server_app.py` (465 lines) — Flower ServerApp (Two-Phase FL)

| Function                                         | Purpose                                                                                                                                                                          |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `set_seed(seed)`                                 | Deterministic seeding (Python, NumPy, PyTorch, CUDA)                                                                                                                             |
| `report_local_baseline_to_backend(clients_data)` | POSTs baseline results to .NET backend at `localhost:8080/api/fl/local-baseline`                                                                                                 |
| `report_round_to_backend(round_num, results)`    | POSTs round results to .NET backend at `localhost:8080/api/fl/rounds`                                                                                                            |
| `server_fn(context)`                             | Creates `ServerAppComponents` with custom `SimpleFedAvg` strategy                                                                                                                |
| `write_results(all_results, ...)`                | Writes `fl_results.json` with full prediction arrays and classification reports                                                                                                  |
| **`main()`**                                     | **Two-phase FL orchestration:**                                                                                                                                                  |
|                                                  | **Phase 1:** Full local training per client (largest client gets 1.5x epochs = 45)                                                                                               |
|                                                  | **Phase 2:** 10 FL rounds, each client fine-tunes for 15 epochs (weak client gets 30), equal-weight FedAvg with alpha=0.7 momentum. Reports results to backend after each round. |
| `app = ServerApp(server_fn=server_fn)`           | Flower entry point                                                                                                                                                               |

---

## 4. `FL_backend/` — C# .NET 10 Backend API

ASP.NET Core Web API with SignalR real-time broadcasting, Entity Framework Core with PostgreSQL.

### Infrastructure Files

| File                          | Lines | Description                                                                        |
| ----------------------------- | ----- | ---------------------------------------------------------------------------------- |
| `FL_backend.slnx`             | 6     | .NET solution file referencing `docker-compose.dcproj` + `FL_backend.csproj`       |
| `docker-compose.yml`          | 36    | Defines 3 services: `fl_backend` (.NET), `postgres:17.2`, `pgadmin` (with volumes) |
| `docker-compose.override.yml` | 15    | Dev overrides: env vars, port mappings (5036:8080), PG connection string           |
| `launchSettings.json`         | 11    | VS Docker Compose profile — `fl_backend` with `StartDebugging`                     |
| `.dockerignore`               | 30    | Docker exclusions (bin, obj, .git, node_modules, etc.)                             |

### `FL_backend/FL_backend/`

#### `FL_backend.csproj` (28 lines)

- .NET 10.0, `net10.0`
- Packages: `Npgsql.EntityFrameworkCore.PostgreSQL` (9.0.4), `Microsoft.AspNetCore.SignalR` (1.2.0), `Swashbuckle.AspNetCore` (7.3.1), `Microsoft.EntityFrameworkCore.Tools` (9.0.4)
- Docker support with Container Rapid Build

#### `Program.cs` (47 lines)

- Service registration: EF Core DbContext with Npgsql, SignalR, CORS (allow `http://localhost:3000`), Swagger, Controllers
- Maps `/hubs/fl` to `FlHub`
- On startup: auto-creates DB schema via `context.Database.EnsureCreated()`
- Uses `app.UseCors("AllowFrontend")`

#### `Controllers/FlController.cs` (161 lines) — REST API

| Endpoint                 | Method | Purpose                                                                                           |
| ------------------------ | ------ | ------------------------------------------------------------------------------------------------- |
| `/api/fl/local-baseline` | GET    | Returns stored local baseline metrics                                                             |
| `/api/fl/local-baseline` | POST   | Accepts `LocalBaselineRequest`, saves to DB, broadcasts via SignalR `LocalBaselineUpdated`        |
| `/api/fl/rounds`         | GET    | Returns all rounds with client results (ordered by round number)                                  |
| `/api/fl/rounds`         | POST   | Accepts `RoundRequest`, saves round + client results to DB, broadcasts `RoundUpdated` via SignalR |

#### `Hubs/FlHub.cs` (7 lines)

Empty SignalR hub class `FlHub : Hub` — the server pushes updates via `Clients.All.SendAsync` from the controller rather than handling client-to-server hub methods.

#### `Data/FLContext.cs` (46 lines)

EF Core DbContext with schema `fl`:

- `DbSet<FlLocalBaseline> LocalBaselines` → table `fl_local_baseline`
- `DbSet<FlRound> Rounds` → table `fl_round`
- `DbSet<FlClientResult> ClientResults` → table `fl_client_result`
- `OnModelCreating` configures relationships, FKs, cascade deletes

#### `Data/Migrations/` — EF Core Migrations

| File                                         | Description                            |
| -------------------------------------------- | -------------------------------------- |
| `20260621195515_InitialFlSchema.cs`          | Creates all 3 tables in `fl` schema    |
| `20260621195515_InitialFlSchema.Designer.cs` | Auto-generated migration designer file |
| `FLContextModelSnapshot.cs`                  | Current model snapshot                 |

Table structure:

- **`fl_local_baseline`**: Id (int PK), ClientId (text), TrainF1, TestF1, Accuracy (double), NumExamples (int), CreatedAt (timestamp)
- **`fl_round`**: Id (int PK), RoundNumber (int unique), CreatedAt (timestamp)
- **`fl_client_result`**: Id (int PK), FlRoundId (int FK→fl_round), ClientId (text), TrainF1, TestF1, Accuracy (double), NumExamples (int)

#### `Models/` — Entity Classes & DTOs

| File                 | Description                                                                                                                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `FlLocalBaseline.cs` | Entity: `Id`, `ClientId`, `TrainF1`, `TestF1`, `Accuracy`, `NumExamples`, `CreatedAt`                                                                                                                              |
| `FlRound.cs`         | Entity: `Id`, `RoundNumber`, `CreatedAt`, nav property `ClientResults`                                                                                                                                             |
| `FlClientResult.cs`  | Entity: `Id`, `FlRoundId` (FK), `ClientId`, `TrainF1`, `TestF1`, `Accuracy`, `NumExamples`                                                                                                                         |
| `FlDto.cs`           | DTOs with `JsonPropertyName` attributes: `LocalBaselineRequest` (list of `ClientMetric`), `RoundRequest` (RoundNumber + list of `ClientMetric`), `ClientMetric` (clientId, trainF1, testF1, accuracy, numExamples) |

#### Config Files

| File                             | Content                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| `appsettings.json`               | PostgreSQL connection string to `localhost:5432`, database `fl_dashboard`, user `fl_user`  |
| `appsettings.Development.json`   | Logging: Debug/Information levels                                                          |
| `Properties/launchSettings.json` | 3 profiles: http (5000), https (5001), Docker (8080)                                       |
| `Dockerfile`                     | Multi-stage: `mcr.microsoft.com/dotnet/sdk:10.0-noble` build → `aspnet:10.0-noble` runtime |
| `Dockerfile.original`            | Same but uses `aspnet:10.0` (non-noble)                                                    |
| `FL_backend.http`                | Test request: `GET https://localhost:5001/weatherforecast`                                 |

---

## 5. `FL_frontend/` — Next.js 15 Dashboard

### Configuration Files

| File                 | Purpose                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `package.json`       | Next.js 15.3, React 19, Recharts 2.15, SignalR 8.2, Tailwind v4, shadcn/ui, clsx, tw-merge |
| `tsconfig.json`      | TypeScript strict mode, path alias `@/*` → `./*`                                           |
| `next.config.mjs`    | Next.js config (minimal, no special settings)                                              |
| `postcss.config.mjs` | PostCSS with Tailwind CSS v4                                                               |
| `components.json`    | shadcn/ui registry config                                                                  |
| `.env`               | `NEXT_PUBLIC_FL_API_BASE=http://localhost:5036`                                            |
| `.gitignore`         | Next.js standard + `.env`                                                                  |

### `lib/` — Shared Utilities

#### `fl-constants.ts` (49 lines)

- `FL_API_BASE` from env var (default `http://localhost:5036`)
- `FL_HUB_URL` = `{FL_API_BASE}/hubs/fl`
- `CLIENT_IDS = ['client_1', 'client_2', 'client_3']` (only 3 in UI, not 4)
- `CLIENT_LABELS`: `client_1` → "Well 1", `client_2` → "Well 2", `client_3` → "Well 3"
- `CLIENT_COLORS`: blue-500, amber-500, emerald-500
- `HEALTH_THRESHOLDS`: green ≥ 0.70, yellow ≥ 0.40, red < 0.40
- `getHealthStatus(testF1)` → `'green' | 'yellow' | 'red'`
- Types: `ClientMetrics` (clientId, trainF1, testF1, accuracy, numExamples), `RoundResult` (roundNumber, clients)

#### `utils.ts` (6 lines)

Minimal `cn()` utility using `clsx` + `tailwind-merge` for conditional class merging.

### `hooks/` — React Hooks

#### `use-fl-socket.ts` (60 lines)

SignalR connection hook:

- Builds connection via `HubConnectionBuilder` with auto-reconnect
- Listens for `LocalBaselineUpdated` (sets baseline state) and `RoundUpdated` (merges/replaces round data sorted by roundNumber)
- Tracks `connectionState`: `connecting` → `connected` | `error` | `disconnected`
- Cleanup on unmount

### `app/` — Pages (Next.js App Router)

#### `layout.tsx` (35 lines)

Root layout: Geist/Geist Mono fonts, metadata "FL Monitor — DHSV Fault Detection", viewport with light/dark theme color, sidebar nav + scrollable main content area.

#### `page.tsx` — Dashboard (154 lines)

FL Dashboard home page:

- Initial REST fetch of `/api/fl/rounds` for historical data
- Merges live SignalR updates on top
- Calculates best round by average F1
- States: loading, empty ("No training data yet"), error ("Cannot connect to FL backend")
- Displays: F1 Score chart, Accuracy chart (Recharts line charts), Best Round Summary table
- Connection badge in header

#### `fault-detection/page.tsx` — Fault Detection (105 lines)

Per-well health monitoring:

- Finds the best round (highest avg F1) from all available data
- Maps each client's metrics to a well card
- 2×2 grid of `WellCard` components
- States: loading, empty, error

#### `local-vs-global/page.tsx` — Local vs Global (249 lines)

Comparison between Phase 1 (local-only) and Phase 2 (federated):

- Fetches both `/api/fl/local-baseline` and `/api/fl/rounds` in parallel
- Builds comparison rows with delta (global - local)
- Shows 3 summary cards: Local Avg F1, Global Avg F1, Change (+/-)
- Bar chart (Recharts) comparing local vs global F1 per client
- Per-client breakdown table with delta indicators (TrendingUp/TrendingDown/Minus icons)
- Color-coded deltas: emerald (improved), rose (hurt), muted (unchanged)

#### `globals.css` (169 lines)

Tailwind CSS v4 with CSS theme variables:

- Light/dark mode via `.dark` class and `prefers-color-scheme` media query
- shadcn/ui style variables (background, foreground, card, border, sidebar, chart colors)
- Custom radii, font families

### `components/`

#### `sidebar-nav.tsx` (65 lines)

Left sidebar (224px):

- Logo + title "FL Monitor / DHSV Fault Detection"
- 3 nav links with icons: FL Dashboard (`/`), Fault Detection (`/fault-detection`), Local vs Global (`/local-vs-global`)
- Active state highlighting (primary background)
- Footer shows backend URL from env var

#### `connection-badge.tsx` (41 lines)

Live connection indicator:

- States: `connected` (green dot + "Live"), `connecting` (amber pulsing), `disconnected` (rose), `error` (rose + "No live updates")
- Uses CSS classes for colors

#### `latest-round-table.tsx` (59 lines)

Best round results table:

- Columns: Client (with color dot), Train F1, Test F1, Accuracy, Examples
- Sorted by canonical CLIENT_IDS order
- Empty state returns null

#### `well-card.tsx` (102 lines)

Individual well health card:

- Color accent bar at top (per-client color)
- Header: colored dot + well name + client ID + health badge (green/yellow/red)
- Health thresholds: F1 ≥ 0.70 → Healthy, 0.40–0.70 → Degraded, < 0.40 → Fault Risk
- 2×2 metrics grid: Test F1, Accuracy, Train F1, Examples
- Empty state: "Waiting for first training round…"
- Threshold legend in footer

### `components/charts/`

#### `chart-utils.ts` (22 lines)

`buildRoundRows(rounds, metric)` — transforms `RoundResult[]` into Recharts-compatible row array where each row is `{ round: number, client_1: number, client_2: number, client_3: number }`.

#### `rounds-f1-chart.tsx` (73 lines)

Recharts `LineChart` — F1 score per round, one line per client (3 lines). Monotone interpolation, no dots, active dot on hover. Y-axis 0–1, X-axis round number, legend maps to client labels.

#### `rounds-accuracy-chart.tsx` (73 lines)

Same layout as F1 chart but for accuracy metric.

#### `local-global-chart.tsx` (90 lines)

Recharts `BarChart` — side-by-side bars per client: gray = Local (Phase 1), blue = Global/FL (Phase 2). Y-axis 0–1 with Test F1 label.

### `components/ui/`

#### `button.tsx` (58 lines)

shadcn/ui-style button with Base UI `ButtonPrimitive` + `class-variance-authority`:

- Variants: default, outline, secondary, ghost, destructive, link
- Sizes: default, xs, sm, lg, icon, icon-xs, icon-sm, icon-lg
- Proper `aria-invalid`, `focus-visible`, disabled states

### `public/` — Static Assets

- `icon.svg`, `icon-dark-32x32.png`, `icon-light-32x32.png`, `apple-icon.png` — Favicons
- `placeholder-logo.png`, `placeholder-logo.svg`, `placeholder-user.jpg`, `placeholder.jpg`, `placeholder.svg` — Placeholder images

---

## 6. Jupyter Notebook

### `fllld.ipynb` (1,714 lines, ~99 KB)

Self-contained Colab notebook implementing the full two-phase FL pipeline in a single code cell:

- Upload ZIP → extract → load `.npz` data per client
- Phase 1: Local training (30 epochs, largest client gets 45)
- Weak client detection (need_score based on data size + F1)
- Phase 2: 10 FL rounds with equal-weight FedAvg + alpha=0.7 momentum
- Final comparison tables + classification reports
- Hyperparams: D_MODEL=16, N_HEADS=2, N_LAYERS=1, DROPOUT=0.5, LR=5e-4, WD=0.05

---

## 7. Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Python FL (Federated/pytorchexample/server_app.py)       │
│    Phase 1: Local training → POST /api/fl/local-baseline    │
│    Phase 2: FL rounds → POST /api/fl/rounds (per round)     │
│              → writes fl_results.json (final)               │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP POST
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. C# Backend (FL_backend)                                  │
│    FlController: saves to PostgreSQL + broadcasts via        │
│    SignalR hub (/hubs/fl)                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket / SignalR
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Next.js Frontend (FL_frontend)                           │
│    useFlSocket hook → live updates                          │
│    Dashboard / Fault Detection / Local vs Global pages      │
└─────────────────────────────────────────────────────────────┘
```
