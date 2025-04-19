import torch
import traceback
import time
import os

from ava import AvaForCausalLM
from torch.optim import AdamW

def train_model(
    model: AvaForCausalLM, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    num_epochs: int, 
    device: torch.device, 
    optimizer: torch.optim.Optimizer = None, 
    checkpoint_dir: str = 'checkpoints',
    learning_rate: float = 5e-5
):
    print('✨ Starting training...')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_time = time.time()

    model.to(device)
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                max_id = torch.max(input_ids).item()

                if max_id >= model.config.vocab_size:
                    print(f'⚠ Warning: Batch {batch_idx} contains token ID {max_id} >= vocab size {model.config.vocab_size}')
                    continue
                
                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels
                )
                
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f'🍀 Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s')
            
            except Exception as e:
                print(f'❌ Error in batch {batch_idx}: {e}')
                traceback.print_exc()
                continue
        
        epoch_time = time.time() - epoch_start_time
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f'🍀 Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s | Average Loss: {avg_loss:.4f}')
        
        checkpoint_path = os.path.join(checkpoint_dir, f'ava_model_epoch_{epoch+1}.pt')
        torch.save({
            'epoch'               : epoch,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss'                : avg_loss,
            'config'              : model.config.to_dict(),
        }, checkpoint_path)

        print(f'💾 Checkpoint saved to {checkpoint_path}')
        
        if val_loader:
            model.eval()
            val_loss = 0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    try:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        if torch.max(input_ids).item() >= model.config.vocab_size:
                            continue
                        
                        outputs = model(
                            input_ids      = input_ids,
                            attention_mask = attention_mask,
                            labels         = labels
                        )
                        
                        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                        
                        val_loss += loss.item()
                        val_batches += 1

                    except Exception as e:
                        print(f'❌ Error in validation batch: {e}')
                        continue
            
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                print(f'🍀 Validation Loss: {avg_val_loss:.4f}')
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(checkpoint_dir, 'ava_model_best.pt')
                    torch.save({
                        'epoch'               : epoch,
                        'model_state_dict'    : model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss'                : best_val_loss,
                        'config'              : model.config.to_dict(),
                    }, best_model_path)

                    print(f'💾🧠 New best model saved with validation loss: {best_val_loss:.4f}')
            
            model.train()
    
    total_time = time.time() - start_time
    print(f'🚀 Training completed in {total_time:.2f} seconds')
    
    return model
