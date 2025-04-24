import torch
from tqdm.notebook import tqdm


def evaluate_model(model, eval_dataloader, device):
    """Evaluation function"""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs["loss"]
            total_loss += loss.item()

    model.train()
    return total_loss / len(eval_dataloader)
