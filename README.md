# Maia Chess REST API (FastAPI)

Policy-network inference API for Maia-style human-like move prediction.

## Run API From Terminal (Complete)

### A. Linux or macOS

1. Move to project folder:

```bash
cd /path/to/project
```

2. Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Prepare model and mapping folder:

```bash
mkdir -p models
```

5. Put your Maia ONNX model file in `models/maia.onnx`.

6. Generate valid Lc0-compatible 1858 mapping JSON:

```bash
python scripts/generate_policy_index_to_uci.py --output models/policy_index_to_uci.json
```

7. Optional: validate existing mapping file only:

```bash
python scripts/generate_policy_index_to_uci.py --output models/policy_index_to_uci.json --validate-only
```

8. Set environment variables:

```bash
export MAIA_MODEL_PATH=models/maia.onnx
export MAIA_POLICY_MAP_PATH=models/policy_index_to_uci.json
```

9. Run API server:

```bash
uvicorn maia_api.api:app --host 0.0.0.0 --port 8000
```

10. Check health endpoint:

```bash
curl -s http://localhost:8000/health
```

### B. Windows PowerShell

1. Move to project folder:

```powershell
cd C:\path\to\project
```

2. Create and activate virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Prepare model and mapping folder:

```powershell
New-Item -ItemType Directory -Path models -Force
```

5. Put your Maia ONNX model file in `models/maia.onnx`.

6. Generate valid Lc0-compatible 1858 mapping JSON:

```powershell
python scripts/generate_policy_index_to_uci.py --output models/policy_index_to_uci.json
```

7. Optional: validate existing mapping file only:

```powershell
python scripts/generate_policy_index_to_uci.py --output models/policy_index_to_uci.json --validate-only
```

8. Set environment variables:

```powershell
$env:MAIA_MODEL_PATH = "models/maia.onnx"
$env:MAIA_POLICY_MAP_PATH = "models/policy_index_to_uci.json"
```

9. Run API server:

```powershell
uvicorn maia_api.api:app --host 0.0.0.0 --port 8000
```

10. Check health endpoint:

```powershell
curl http://localhost:8000/health
```

For policy size `1858`, the API only accepts mapping loaded from
`MAIA_POLICY_MAP_PATH` and validates it against canonical Lc0 order.

## Lc0 Policy Order Notes

- Mapping is generated algorithmically, not manually typed.
- Order follows Lc0 1858 convention:
  - 1792 base moves (queen-like rays + knight jumps) in from-square and to-square scan order.
  - 66 promotion entries using promotion piece order: rook, bishop, queen.
- Knight promotions use the base move index (without suffix) in Lc0 convention.
- Mapping validation checks include: exact 1858 length, no duplicate moves,
  valid UCI format, and exact canonical order.

Validate an existing mapping file without regenerating:

`python scripts/generate_policy_index_to_uci.py --output models/policy_index_to_uci.json --validate-only`

## Analyze Endpoint

`POST /analyze`

Request:

```json
{
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 3",
  "top_k": 5,
  "temperature": 1.0
}
```

Response:

```json
{
  "moves": [
    { "uci": "g8f6", "prob": 0.32 },
    { "uci": "d7d5", "prob": 0.21 }
  ]
}
```

## Policy Map Endpoints

`GET /policy-map/export`

- Exports the already loaded mapping from `MAIA_POLICY_MAP_PATH`.
- Does not generate mapping dynamically.
- Returns clear error when mapping is missing/invalid.

Success example:

```bash
curl -s http://localhost:8000/policy-map/export
```

```json
{
  "policy_size": 1858,
  "source_path": "models/policy_index_to_uci.json",
  "mapping": [
    "a1b1",
    "a1c1",
    "a1d1",
    "...",
    "h7h8q"
  ]
}
```

Error example when `MAIA_POLICY_MAP_PATH` is not set or invalid:

```bash
curl -s http://localhost:8000/policy-map/export
```

```json
{
  "detail": "Policy size 1858 detected but MAIA_POLICY_MAP_PATH is not set. Provide a valid Lc0 policy_index_to_uci.json file."
}
```

`GET /policy-map/move/{uci}`

- Returns index for a specific UCI move from the loaded policy map.
- Useful for debugging move->index mapping.

Success example:

```bash
curl -s http://localhost:8000/policy-map/move/a1b1
```

```json
{
  "uci": "a1b1",
  "index": 0
}
```

Error example (UCI not found in mapping):

```bash
curl -s http://localhost:8000/policy-map/move/e7e8n
```

```json
{
  "detail": "Move not found in loaded policy map: e7e8n"
}
```

`GET /policy-map/index/{idx}`

- Reverse lookup index->UCI from loaded mapping.
- Validates `idx` in range `[0, policy_size-1]`.
- Returns 404 when index is missing in loaded map.

Success example:

```bash
curl -s http://localhost:8000/policy-map/index/0
```

```json
{
  "index": 0,
  "uci": "a1b1"
}
```

Error example (index out of range):

```bash
curl -s http://localhost:8000/policy-map/index/3000
```

```json
{
  "detail": "Index out of range: 3000. Valid range is [0, 1857]"
}
```

Error example when policy is 4672 (implicit AlphaZero mapping):

```bash
curl -s http://localhost:8000/policy-map/index/123
```

```json
{
  "detail": "Explicit index-to-move lookup is unavailable for policy size 4672 (implicit AlphaZero mapping)"
}
```

For policy size `4672` (AlphaZero implicit mapping), policy-map debug endpoints return
an explicit error because there is no explicit JSON mapping to export.

## Quick Bidirectional Validation

Use these steps to verify index <-> UCI consistency:

1. Get mapping and choose one index:

```bash
curl -s http://localhost:8000/policy-map/export
```

2. Query the index to get UCI:

```bash
curl -s http://localhost:8000/policy-map/index/0
```

Expected example:

```json
{
  "index": 0,
  "uci": "a1b1"
}
```

3. Query the returned UCI back to index:

```bash
curl -s http://localhost:8000/policy-map/move/a1b1
```

Expected example:

```json
{
  "uci": "a1b1",
  "index": 0
}
```

If both indices are equal, mapping is consistent in both directions.
