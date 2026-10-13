"""Fresh neural members for the four-month, one-month-buffer validation.

This module has no executable training entry point. The user-launched outer
runner supplies a freshly prepared, fingerprinted fold. No historical fold
matrix, fitted encoder, checkpoint or held-out label is consumed here.

Three seeds of each family are fitted separately. All folds use 21 epochs of
the 24-epoch N09 schedule; this prospectively fixed stopping policy replaces
historical month-specific stopping. The callsign pool has three members rather
than the historical fifteen. Each family uses a logical batch of 2048 and
512-row microbatches, including callsign PLE; the latter changes its historical
dropout execution while preserving the architecture and optimizer recipe.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "VALIDATION"), str(ROOT / "pipeline"), str(ROOT / "VALIDATION/tools")]

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from records import sha256

SEEDS = [42, 1042, 2042]
FAMILIES = ["calls_ple", "physical_ple", "public_tabm"]
TRACK_DIRECTORY = ROOT / "data/derived/sparse_track_sequence_20261010"
EPOCHS = 21
SCHEDULE_EPOCHS = 24
BATCH = 2048
MICROBATCH = 512
PREDICTION_CHUNK = 512
N09_CONFIG = dict(hidden=[512, 512, 256], dropout=.1, lr=.002, weight_decay=1e-5,
                  batch=2048, epochs=24, bins=48, d_ple=16, label_clip=10800.)
CONFIG = dict(
    families=FAMILIES, seeds=SEEDS, epochs=EPOCHS, schedule_epochs=SCHEDULE_EPOCHS,
    effective_batch=BATCH, microbatch=MICROBATCH, prediction_chunk=PREDICTION_CHUNK,
    label_clip=10800.0, optimizer="AdamW", learning_rate=0.002,
    original_n09_configuration=N09_CONFIG,
    weight_decay=dict(calls_ple=1e-5, physical_ple=1e-5, public_tabm=3e-4),
    ordinary_training_population="fresh fold builder, Rome-substituted rows excluded",
    encoding="train-only nn_jan_jul.Encoder and 48 train-quantile PLE bins",
    calls_ple="original 512/512/256 SiLU PLE, combo156 plus six callsign inputs",
    physical_ple="original 156-input PLE plus physical sparse-track CNN",
    public_tabm="original 156-input encoder plus 12 fixed public transforms and physical CNN; K32 two512-block TabM",
    selection="no held labels, early stopping, seed selection or parameter search",
    cache_scope="input-only complete2025 sparse tracks, never old fold matrices or weights",
    pool_size_caveat="callsign3 differs from historical callsign15",
    schedule_caveat="all21 epochs differs from historical July17 and historical per-group stops",
    execution_caveat="512-row microbatch dropout stream differs from historical callsign2048",
    inference_backend="scoped deterministic CuDNN, benchmarkFalse; training flags restored",
)


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def _verify(record):
    path = ROOT / record["path"]
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Artifact changed: {path}")
    return path


def _modules():
    # These imports define architectures/helpers; none calls an old fit runner.
    import nn_residual_jan_jul as shared
    import chunked_direct_data as chunked
    import nn_track_physical_scales as physical
    import nn_public_tabm as tabm
    import sparse_track_tensors as track_tensors
    # This is the literal N09 configuration asserted by shared.recipe(). Apply
    # it directly, so a new validation never depends on reading old HPO scores.
    shared.hpo.apply_nn(N09_CONFIG, shared.nnj, shared.nni)
    return shared, chunked, physical, tabm, track_tensors


def source_files():
    """Sources to include in the outer immutable run identity, without data IO."""
    shared, chunked, physical, tabm, track_tensors = _modules()
    modules = [shared, chunked, physical, physical.parent, tabm, tabm.control,
               tabm.control.public, tabm.features, track_tensors, track_tensors.seq,
               shared.nnj, shared.nni, shared.hpo]
    return sorted({str(Path(__file__).relative_to(ROOT))} |
                  {str(Path(m.__file__).resolve().relative_to(ROOT)) for m in modules})


def _input_identity(data):
    value = data.get("input_identity")
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("Outer runner must supply a verified full-fold pickle input_identity")
    _verify(value)
    ids = np.asarray(data["train_ids"], dtype=np.int64)
    held_ids = np.asarray(data["ids"], dtype=np.int64)
    if len(ids) != len(data["X"]) or len(ids) != len(data["y"]) or len(held_ids) != len(data["V"]):
        raise ValueError("Neural input row lengths disagree")
    if len(np.unique(ids)) != len(ids) or len(np.unique(held_ids)) != len(held_ids):
        raise ValueError("Duplicate neural movement IDs")
    if np.intersect1d(ids, held_ids).size:
        raise ValueError("Training and validation movements overlap")
    if not np.isfinite(np.asarray(data["y"], dtype=float)).all():
        raise ValueError("Nonfinite training target")
    return value


def _encoding(data, out, group, shared, chunked):
    """Persist a train-only encoder once, shared across seeds and physical arms."""
    folder = Path(out) / "neural_encoding" / group
    record_path = folder / "complete.json"
    names = list(data["feature_groups"][group])
    expected = 162 if group == "calls162" else 156
    if len(names) != expected or len(set(names)) != expected:
        raise ValueError(f"Unexpected {group} feature list")
    identity = dict(input=data["input_identity"], features=names,
                    configuration=CONFIG, source=_artifact(__file__))
    if record_path.exists():
        record = _read(record_path)
        if record["identity"] != identity:
            raise ValueError("Neural encoding identity changed")
        for entry in record["files"]:
            _verify(entry)
        with _verify(record["preprocessing"]).open("rb") as handle:
            return folder, pickle.load(handle), record
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Incomplete neural encoding retained for inspection: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    _json(folder / "started.json", identity)
    X, V = data["X"].loc[:, names], data["V"].loc[:, names]
    encoder = shared.nnj.Encoder(X)
    # Only X fits categories, quantiles, imputation/scaling and missing flags.
    cat, num = chunked.encode(encoder, X, folder, "train")
    vcat, vnum = chunked.encode(encoder, V, folder, "held")
    bins, ple, plain = chunked.layout(encoder, num)
    target = np.minimum(np.asarray(data["y"], dtype=float), CONFIG["label_clip"])
    mu, sd = float(target.mean()), float(target.std())
    if not sd > 0 or not np.isfinite(sd):
        raise ValueError("Invalid training target scale")
    state = dict(encoder=encoder, columns=names, cards=encoder.cards, bins=bins,
                 ple=ple, plain=plain, mu=mu, sd=sd, input=data["input_identity"])
    prepath = folder / "preprocessing.pkl"
    with prepath.open("wb") as handle:
        pickle.dump(state, handle, protocol=5)
    for value in [cat, num, vcat, vnum]:
        value.flush()
        value._mmap.close()
    files = [folder / f"{part}_{kind}.npy" for part in ["train", "held"] for kind in ["cat", "num"]]
    record = dict(identity=identity, preprocessing=_artifact(prepath), files=[_artifact(p) for p in files],
                  train_rows=len(X), held_rows=len(V), held_labels_read=False)
    _json(record_path, record)
    return folder, state, record


def _load_tensors(folder, state, data, family, chunked, tabm, device):
    result = []
    for part in ["train", "held"]:
        cat = np.load(folder / f"{part}_cat.npy", mmap_mode="r", allow_pickle=False)
        num = np.load(folder / f"{part}_num.npy", mmap_mode="r", allow_pickle=False)
        tensors = (chunked.tensor_columns(cat, None, device),
                   chunked.tensor_columns(num, state["ple"], device),
                   chunked.tensor_columns(num, state["plain"], device))
        if family == "public_tabm":
            frame = data["X" if part == "train" else "V"]
            extra = torch.from_numpy(tabm.transform(frame)).to(device)
            tensors = (*tensors[:2], torch.cat([tensors[2], extra], dim=1))
        result.append(tensors)
        cat._mmap.close()
        num._mmap.close()
    return result


_TRACK_CHECKS = {}


def _tracks(data, track_tensors, device):
    """Read only the audited input cache; explicitly reject changed bytes."""
    manifest_path = TRACK_DIRECTORY / "full.json"
    manifest = _read(manifest_path)
    settings = manifest["settings"]
    if (settings.get("labels_used") is not False or
            settings.get("original_sequence") != track_tensors.seq.SETTINGS or
            len(settings.get("full_days", [])) != 365 or
            settings["full_days"][0] != "2025-01-01" or
            settings["full_days"][-1] != "2025-12-31"):
        raise ValueError("Sparse track cache is not the frozen input-only2025 representation")
    record = _artifact(manifest_path)
    paths = [TRACK_DIRECTORY / name for name in ["full_index.parquet", "full_points.bin"]]
    signature = (record["sha256"], *[(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths])
    if _TRACK_CHECKS.get(str(TRACK_DIRECTORY)) != signature:
        if set(manifest["outputs"]) != {p.name for p in paths}:
            raise ValueError("Unexpected sparse track cache outputs")
        for p in paths:
            if sha256(p) != manifest["outputs"][p.name]:
                raise ValueError(f"Sparse track cache changed: {p}")
        _TRACK_CHECKS[str(TRACK_DIRECTORY)] = signature
    index = pd.read_parquet(paths[0])
    points = np.memmap(paths[1], dtype=np.dtype("<i4"), mode="r").reshape(-1, 6)
    if len(index) != manifest["rows"] or len(points) != manifest["stored_points"]:
        raise ValueError("Sparse track cache shape disagrees with manifest")
    tracks = track_tensors.RaggedTracks(index, points, device)
    train_rows, held_rows = tracks.align(data["train_ids"]), tracks.align(data["ids"])
    if int(index["count"].max()) > track_tensors.seq.MAX_POINTS:
        raise ValueError("Sparse track cache exceeds frozen maximum length")
    return tracks, train_rows, held_rows, record


def _make(family, state, plain_width, shared, physical, tabm, device):
    constructor = dict(calls_ple=shared.nni.PleNet, physical_ple=physical.SparseFusion,
                       public_tabm=tabm.PublicTabM)[family]
    return constructor(state["cards"], state["bins"], plain_width).to(device)


def _forward(model, values, rows, tracks, track_rows):
    tab = tuple(value[rows] for value in values)
    return model(*tab) if tracks is None else model(*tab, *tracks.batch(track_rows))


def _loss(output, target):
    if output.ndim != 2 or target.ndim != 1 or len(output) != len(target):
        raise ValueError("Wrong neural output/target shape")
    return torch.nn.functional.mse_loss(output, target[:, None].expand_as(output))


def _prediction(model, values, state, tracks, rows):
    model.eval()
    output = []
    backend_before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    with torch.no_grad(), torch.backends.cudnn.flags(deterministic=True, benchmark=False):
        for start in range(0, len(values[0]), PREDICTION_CHUNK):
            stop = min(start + PREDICTION_CHUNK, len(values[0]))
            tr = None if rows is None else rows[start:stop]
            batch = _forward(model, values, slice(start, stop), tracks, tr)
            output.append(batch.mean(dim=1).cpu().numpy())
    if backend_before != (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark):
        raise ValueError("Prediction failed to restore the fitting backend flags")
    result = np.maximum(np.concatenate(output).astype(float) * state["sd"] + state["mu"], 0)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite neural predictions")
    return result


def _torch_save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def _execution(device, threads):
    return dict(host=platform.node(), device=str(device), gpu=torch.cuda.get_device_name(device),
                torch=str(torch.__version__), numpy=np.__version__, threads=threads,
                tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()))


def fit_neural_member(family, data, out, seed=42, threads=2, device="cuda"):
    """Fit/recover one member and return (full-held float prediction, receipt).

    Complete members reload their validated predictions. An interrupted fit
    resumes from its last committed epoch including all optimizer/scheduler/RNG
    state; committed epochs are never repeated. A hardware/source/input
    mismatch fails instead of silently reusing or restarting a fit.
    """
    if family not in FAMILIES or seed not in SEEDS:
        raise ValueError("Family/seed is outside the frozen four-month design")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The frozen neural validation requires a CUDA GPU")
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    shared, chunked, physical, tabm, track_tensors = _modules()
    input_identity = _input_identity(data)
    source_identity = {name: sha256(ROOT / name) for name in source_files()}
    folder = Path(out) / "neural" / family / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(family=family, seed=seed, config=CONFIG, input=input_identity,
                    sources=source_identity, execution=_execution(device, threads))
    ipath, complete = folder / "identity.json", folder / "complete.json"
    if ipath.exists():
        if _read(ipath) != identity:
            raise ValueError(f"Neural run identity changed: {folder}")
    else:
        if any(folder.iterdir()):
            raise ValueError(f"Unrecorded neural artifacts: {folder}")
        _json(ipath, identity)
    if complete.exists():
        receipt = _read(complete)
        if receipt["identity"] != identity:
            raise ValueError("Completed member identity changed")
        for key in ["model", "checkpoint", "prediction", "preprocessing"]:
            _verify(receipt[key])
        saved = pd.read_parquet(_verify(receipt["prediction"]))
        np.testing.assert_array_equal(saved["mvt_id"], data["ids"])
        return saved["prediction"].to_numpy(float), receipt

    with threadpool_limits(limits=threads):
        group = "calls162" if family == "calls_ple" else "combo156"
        encdir, state, encoding = _encoding(data, out, group, shared, chunked)
        train, held = _load_tensors(encdir, state, data, family, chunked, tabm, device)
        tracks = train_rows = held_rows = track_record = None
        if family != "calls_ple":
            tracks, train_rows, held_rows, track_record = _tracks(data, track_tensors, device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = _make(family, state, train[2].shape[1], shared, physical, tabm, device)
        target = np.minimum(np.asarray(data["y"], float), CONFIG["label_clip"])
        ty = torch.from_numpy(((target - state["mu"]) / state["sd"]).astype(np.float32)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                               weight_decay=CONFIG["weight_decay"][family])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=CONFIG["learning_rate"], total_steps=SCHEDULE_EPOCHS * math.ceil(len(ty) / BATCH))
        generator = torch.Generator(device=device).manual_seed(seed)
        checkpoint_path, model_path = folder / "checkpoint.pt", folder / "model.pt"
        checkpoint_identity = dict(identity=identity, preprocessing=encoding["preprocessing"], tracks=track_record)
        curve, done = [], 0
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint["identity"] != checkpoint_identity:
                raise ValueError("Checkpoint inputs/encoding/track cache changed")
            model.load_state_dict(checkpoint["state_dict"])
            opt.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            torch.set_rng_state(checkpoint["cpu_rng"].cpu())
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
            generator.set_state(checkpoint["permutation_rng"].cpu())
            np.random.set_state(checkpoint["numpy_rng"])
            curve, done = checkpoint["curve"], checkpoint["epoch"]
            if not 0 <= done <= EPOCHS or [v["epoch"] for v in curve] != list(range(1, done + 1)):
                raise ValueError("Invalid checkpoint training progress")
            del checkpoint
        elif model_path.exists():
            raise ValueError("Model lacks its training checkpoint; refusing to refit")
        if model_path.exists() and done != EPOCHS:
            raise ValueError("Completed model has an incomplete epoch checkpoint")
        model.train()
        for epoch in range(done, EPOCHS):
            started = time.monotonic()
            perm = torch.randperm(len(ty), device=device, generator=generator)
            cpu = perm.cpu().numpy()
            total = 0.0
            for first in range(0, len(ty), BATCH):
                end = min(first + BATCH, len(ty))
                opt.zero_grad(set_to_none=True)
                for lo in range(first, end, MICROBATCH):
                    hi = min(lo + MICROBATCH, end)
                    ix = perm[lo:hi]
                    tr = None if tracks is None else train_rows[cpu[lo:hi]]
                    loss = _loss(_forward(model, train, ix, tracks, tr), ty[ix])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite neural loss; checkpoint retained without retry")
                    (loss * ((hi - lo) / (end - first))).backward()
                    total += float(loss.detach()) * (hi - lo)
                opt.step()
                scheduler.step()
            curve.append(dict(epoch=epoch + 1, train_mse_std=total / len(ty), seconds=time.monotonic() - started))
            _torch_save(checkpoint_path, dict(
                identity=checkpoint_identity, state_dict=model.state_dict(), optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(), cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device), permutation_rng=generator.get_state(),
                numpy_rng=np.random.get_state(), epoch=epoch + 1, curve=curve))
            print(f"{data['meta']['month']} {family} seed{seed} epoch{epoch+1}/{EPOCHS}: train={total / len(ty):.6f}", flush=True)
        if not model_path.exists():
            _torch_save(model_path, dict(identity=checkpoint_identity, state_dict=model.state_dict(), epochs=EPOCHS))
        stored = torch.load(model_path, map_location=device, weights_only=False)
        if stored["identity"] != checkpoint_identity or stored["epochs"] != EPOCHS:
            raise ValueError("Completed model identity changed")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, stored["state_dict"][key], atol=0, rtol=0)
        # Save weights before any held inference; compare all predictions after
        # an independent constructor/state reload, with the same chunk size.
        prediction = _prediction(model, held, state, tracks, held_rows)
        restored = _make(family, state, train[2].shape[1], shared, physical, tabm, device)
        restored.load_state_dict(stored["state_dict"])
        replay = _prediction(restored, held, state, tracks, held_rows)
        np.testing.assert_array_equal(replay, prediction)
        predpath = folder / "prediction.parquet"
        if predpath.exists():
            old = pd.read_parquet(predpath)
            np.testing.assert_array_equal(old["mvt_id"], data["ids"])
            np.testing.assert_array_equal(old["prediction"], prediction)
        else:
            temp = predpath.with_suffix(".parquet.tmp")
            pd.DataFrame(dict(mvt_id=data["ids"], prediction=prediction)).to_parquet(temp, index=False)
            temp.replace(predpath)
        receipt = dict(identity=identity, model=_artifact(model_path), checkpoint=_artifact(checkpoint_path),
                       prediction=_artifact(predpath), preprocessing=encoding["preprocessing"],
                       track_cache=track_record, curve=curve, epochs=EPOCHS,
                       train_rows=len(train[0]), held_rows=len(held[0]), held_labels_read=False,
                       all_held_reload_exact=True, updates_saved_before_held_inference=True)
        _json(complete, receipt)
        del model, restored, stored, train, held, ty, opt, scheduler, tracks
        gc.collect()
        torch.cuda.empty_cache()
        return prediction, receipt
