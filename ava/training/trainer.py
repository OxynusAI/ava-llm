
import torch

from tqdm.notebook import tqdm
from .metrics import evaluate_model

def train_model(model, train_dataloader, optimizer, scheduler, num_epochs, device, eval_dataloader=None):
    model.train()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')       
         
        progress_bar = tqdm(train_dataloader, desc='Training')
        total_loss = 0

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
            
        if eval_dataloader is not None:
            eval_loss = evaluate_model(model, eval_dataloader, device)
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_dataloader)}, Eval Loss: {eval_loss}')

        else:
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_dataloader)}')
            
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':                 total_loss / len(train_dataloader),
        }, f'ava_model_epoch_{epoch + 1}.pt')
        
        print(f"Checkpoint saved as ava_model_epoch_{epoch + 1}.pt")
    
    return model
