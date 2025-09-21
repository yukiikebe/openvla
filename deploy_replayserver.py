#!/usr/bin/env python3
"""
ReplayServer compatible with deploy_real_robot.py client.

- POST /act     -> returns next ground-truth action: {"action": <np.ndarray>}
- POST /init    -> (re)load first episode from TFDS export
- POST /reset   -> reset step index to 0 or to provided idx
- GET  /status  -> report initialized, idx, n_steps, source

The server accepts the same payload keys your client sends:
    {"image": <HWC uint8>, "wrist_image": <HWC uint8>, "prompt": str, "unnorm_key": str}
but ignores them for replay (actions come from the dataset).

Requires:
    pip install fastapi uvicorn json-numpy tensorflow tensorflow-datasets
"""

import threading
from pathlib import Path
from typing import List, Optional

import json_numpy
json_numpy.patch()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# ====== EDIT THESE TO MATCH YOUR EXPORT ======
OUT_DIR = "./rlds_ur10e_task1_action_eef"      # must match your exporter --out_dir
DATASET_NAME = "real_ur10e"       # must match your dataset name
VERSION = "1.0.0"                 # must match your version
EPISODE_INDEX = 0                 # which episode to replay (0 = first)
# =============================================

app = FastAPI()


class ReplayState:
    def __init__(self):
        self.lock = threading.Lock()
        self.actions: List = []
        self.idx: int = 0
        self.source: Optional[str] = None

REPLAY = ReplayState()


def _load_episode_actions(out_dir: str, dataset_name: str, version: str, ep_index: int = 0):
    """Load actions for episode `ep_index` from TFDS export."""
    ds_dir = Path(out_dir) / dataset_name / version
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

    # Import TF before TFDS
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"TensorFlow not importable: {e}")
    try:
        import tensorflow_datasets as tfds
    except Exception as e:
        raise RuntimeError(f"tensorflow-datasets not importable: {e}")

    # Build dataset
    if hasattr(tfds, "builder_from_directory"):
        builder = tfds.builder_from_directory(str(ds_dir))
        ds = builder.as_dataset(split="train")
    else:
        ds, _ = tfds.load(f"{dataset_name}:{version}", split="train",
                          data_dir=out_dir, with_info=True)

    # Pick the requested episode
    it = iter(ds)
    for _ in range(ep_index + 1):
        try:
            episode = next(it)
        except StopIteration:
            raise RuntimeError(f"Requested episode index {ep_index} not found.")

    tf_steps = episode["steps"]
    actions = []
    for step_np in tf_steps.as_numpy_iterator():
        action = step_np.get("action", None)
        if action is None:
            raise KeyError("Step is missing 'action' key. Check your export schema.")
        actions.append(action)

    if not actions:
        raise RuntimeError("Episode had zero steps (no actions).")

    return actions, str(ds_dir)


@app.post("/init")
@app.post("/replay/init")
async def replay_init():
    """(Re)load episode actions and reset pointer."""
    with REPLAY.lock:
        try:
            actions, src = _load_episode_actions(OUT_DIR, DATASET_NAME, VERSION, EPISODE_INDEX)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load RLDS: {e}")
        REPLAY.actions = actions
        REPLAY.idx = 0
        REPLAY.source = src
        return {"status": "ok", "n_steps": len(actions), "idx": 0, "source": REPLAY.source}


@app.post("/reset")
@app.post("/replay/reset")
async def replay_reset(request: Request):
    """Reset index to 0 or to provided 'idx' (JSON body)."""
    body = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
    new_idx = int(body.get("idx", 0))

    with REPLAY.lock:
        if not REPLAY.actions:
            raise HTTPException(status_code=400, detail="Replay not initialized; call /init.")
        if not (0 <= new_idx < len(REPLAY.actions)):
            raise HTTPException(status_code=400, detail=f"idx must be in [0, {len(REPLAY.actions)-1}]")
        REPLAY.idx = new_idx
        return {"status": "ok", "idx": REPLAY.idx}


@app.get("/status")
@app.get("/replay/status")
async def replay_status():
    with REPLAY.lock:
        return {
            "initialized": bool(REPLAY.actions),
            "idx": REPLAY.idx,
            "n_steps": len(REPLAY.actions),
            "source": REPLAY.source,
        }


@app.post("/act")
async def act(request: Request):
    """
    Match the client's RestClientPolicy.infer(...) call:
      - Accepts JSON with keys {"image", "wrist_image", "prompt", "unnorm_key"}.
      - Returns {"action": <np.ndarray>} where array is the next ground-truth step.

    Images/prompt/unnorm_key are ignored for replay; kept only for compatibility.
    """
    # Parse/ignore incoming payload (we don't need it for replay)
    if request.headers.get("content-type", "").startswith("application/json"):
        _ = await request.json()

    with REPLAY.lock:
        if not REPLAY.actions:
            actions, src = _load_episode_actions(OUT_DIR, DATASET_NAME, VERSION, EPISODE_INDEX)
            REPLAY.actions, REPLAY.idx, REPLAY.source = actions, 0, src

        if REPLAY.idx >= len(REPLAY.actions):
            REPLAY.idx = len(REPLAY.actions) - 1 

        action = REPLAY.actions[REPLAY.idx]
        REPLAY.idx += 1

    # json_numpy patches JSONResponse so numpy arrays are serialized automatically
    return JSONResponse({"action": action})


if __name__ == "__main__":
    import uvicorn
    # Default host/port align with your client defaults (127.0.0.1:18000 in your Args)
    uvicorn.run(app, host="0.0.0.0", port=8000)
