import io

import blobfile as bf
import torch as th


def setup_dist():
    """Single-process, single-device setup.

    This project originally supported multi-process distributed training (MPI +
    torch.distributed). For single-GPU/CPU usage, we keep the same API but make
    this a lightweight device initializer.
    """

    if th.cuda.is_available():
        # Always use the first visible GPU in single-card mode.
        th.cuda.set_device(0)


def dev():

    if th.cuda.is_available():
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """Load a checkpoint from local/remote storage.

    Previously this function broadcasted parameters via MPI.
    In single-process mode, we simply read the file once.
    """

    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """No-op in single-process mode."""

    return
