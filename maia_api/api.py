from __future__ import annotations

import os

import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import MaiaInferenceEngine


class AnalyzeRequest(BaseModel):
    fen: str = Field(..., description="FEN position to analyze")
    top_k: int = Field(default=5, ge=1, le=50)
    temperature: float = Field(default=1.0, gt=0.0, le=5.0)


class MovePrediction(BaseModel):
    uci: str
    prob: float


class AnalyzeResponse(BaseModel):
    moves: list[MovePrediction]


class PolicyMapExportResponse(BaseModel):
    policy_size: int
    source_path: str
    mapping: list[str]


class PolicyMapMoveResponse(BaseModel):
    uci: str
    index: int


class PolicyMapIndexResponse(BaseModel):
    index: int
    uci: str


MODEL_PATH = os.getenv("MAIA_MODEL_PATH", "models/maia.onnx")
POLICY_MAP_PATH = os.getenv("MAIA_POLICY_MAP_PATH")

app = FastAPI(title="Maia Chess REST API", version="1.0.0")
engine: MaiaInferenceEngine | None = None
engine_load_error: str | None = None


@app.on_event("startup")
def load_engine() -> None:
    global engine, engine_load_error
    try:
        engine = MaiaInferenceEngine.from_onnx(MODEL_PATH, policy_map_path=POLICY_MAP_PATH)
        engine_load_error = None
    except Exception as error:
        engine = None
        engine_load_error = str(error)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    if engine is None:
        return {"status": "error", "detail": engine_load_error or "Model not loaded"}
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=engine_load_error or "Model not loaded",
        )

    try:
        moves = engine.analyze(
            fen=payload.fen,
            top_k=payload.top_k,
            temperature=payload.temperature,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error

    return AnalyzeResponse(moves=[MovePrediction(**move) for move in moves])


@app.get("/policy-map/export", response_model=PolicyMapExportResponse)
def export_policy_map() -> PolicyMapExportResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail=engine_load_error or "Model not loaded")

    mapper = engine.mapper
    if mapper.policy_size != 1858:
        raise HTTPException(
            status_code=400,
            detail="Explicit mapping export is unavailable for policy size 4672 (implicit AlphaZero mapping)",
        )
    if not mapper.loaded_from_file:
        raise HTTPException(
            status_code=400,
            detail="Policy mapping must be loaded from MAIA_POLICY_MAP_PATH",
        )

    try:
        mapping = mapper.export_loaded_mapping()
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Policy mapping not available. For policy size 1858, set MAIA_POLICY_MAP_PATH "
                "to a valid Lc0-compatible JSON mapping file."
            ),
        ) from error

    return PolicyMapExportResponse(
        policy_size=mapper.policy_size,
        source_path=mapper.mapping_path or "",
        mapping=mapping,
    )


@app.get("/policy-map/move/{uci}", response_model=PolicyMapMoveResponse)
def policy_map_move_index(uci: str) -> PolicyMapMoveResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail=engine_load_error or "Model not loaded")

    mapper = engine.mapper
    if mapper.policy_size != 1858:
        raise HTTPException(
            status_code=400,
            detail="Explicit move-to-index lookup is unavailable for policy size 4672 (implicit AlphaZero mapping)",
        )
    if not mapper.loaded_from_file:
        raise HTTPException(
            status_code=400,
            detail="Policy mapping must be loaded from MAIA_POLICY_MAP_PATH",
        )

    try:
        chess.Move.from_uci(uci)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"Invalid UCI move format: {uci}") from error

    index = mapper.index_for_uci(uci)
    if index is None:
        raise HTTPException(status_code=404, detail=f"Move not found in loaded policy map: {uci}")

    return PolicyMapMoveResponse(uci=uci, index=index)


@app.get("/policy-map/index/{idx}", response_model=PolicyMapIndexResponse)
def policy_map_index_lookup(idx: int) -> PolicyMapIndexResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail=engine_load_error or "Model not loaded")

    mapper = engine.mapper
    if mapper.policy_size != 1858:
        raise HTTPException(
            status_code=400,
            detail="Explicit index-to-move lookup is unavailable for policy size 4672 (implicit AlphaZero mapping)",
        )
    if not mapper.loaded_from_file:
        raise HTTPException(
            status_code=400,
            detail="Policy mapping must be loaded from MAIA_POLICY_MAP_PATH",
        )

    if idx < 0 or idx >= mapper.policy_size:
        raise HTTPException(
            status_code=400,
            detail=f"Index out of range: {idx}. Valid range is [0, {mapper.policy_size - 1}]",
        )

    uci = mapper.uci_for_index(idx)
    if uci is None:
        raise HTTPException(status_code=404, detail=f"Index not found in loaded policy map: {idx}")

    round_trip = mapper.index_for_uci(uci)
    if round_trip != idx:
        raise HTTPException(
            status_code=500,
            detail=(
                "Loaded mapping is inconsistent for reverse lookup "
                f"(index {idx} -> {uci} -> {round_trip})"
            ),
        )

    return PolicyMapIndexResponse(index=idx, uci=uci)
