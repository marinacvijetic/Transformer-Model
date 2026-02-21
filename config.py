from pathlib import Path


def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 15,
        "lr": 1e-4,
        "seq_len": 256,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,  # None | "latest" | "05"
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]

    weights_files = list(Path(model_folder).glob(f"{model_basename}*.pt"))

    if not weights_files:
        return None

    weights_files.sort()
    return str(weights_files[-1])