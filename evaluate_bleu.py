import re
from pathlib import Path

import torch
import sacrebleu

from config import get_config
from train import get_ds, get_model, greedy_decode


def pick_checkpoint(weights_dir: str, model_basename: str) -> Path:
    """
    Picks the highest-epoch checkpoint from weights/ folder.
    """
    wdir = Path(weights_dir)
    if not wdir.exists():
        raise FileNotFoundError(f"weights folder not found: {wdir.resolve()}")

    pattern = re.compile(rf"^{re.escape(model_basename)}(\d+)\.pt$")
    candidates = []

    for p in wdir.glob(f"{model_basename}*.pt"):
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {wdir.resolve()} matching {model_basename}XX.pt"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # 3) Load best/latest checkpoint from weights/
    ckpt_path = pick_checkpoint(config["model_folder"], config["model_basename"])
    print("Loading checkpoint:", ckpt_path)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # 4) Run inference on validation set and compute BLEU
    predictions = []
    references = []

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # greedy decode 
            out_tokens = greedy_decode(
                model=model,
                source=encoder_input,
                source_mask=encoder_mask,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                max_len=config["seq_len"],
                device=device,
            )

            pred_text = tokenizer_tgt.decode(out_tokens.detach().cpu().numpy())
            tgt_text = batch["tgt_text"][0]  # because batch_size=1

            # strip special tokens if they appear in decoded text
            pred_text = pred_text.replace("[SOS]", "").replace("[EOS]", "").strip()

            predictions.append(pred_text)
            references.append(tgt_text)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print("\n=== BLEU (sacreBLEU) ===")
    print("BLEU:", bleu.score)

    Path("bleu_result.txt").write_text(
        f"Checkpoint: {ckpt_path}\nBLEU: {bleu.score}\n",
        encoding="utf-8",
    )
    print("Saved to bleu_result.txt")


if __name__ == "__main__":
    main()