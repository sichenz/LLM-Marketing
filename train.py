import torch
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from torch.utils.data import DataLoader
from dataset import SocialMediaMarketingDataset
from model_utils import get_model, get_tokenizer
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "large"], help="Use small (CPU-friendly) or large (GPU) default models.")
    parser.add_argument("--model_name", type=str, default=None, help="Custom HuggingFace model ID. Overrides --model_size.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps. Effective batch = batch_size * grad_accum_steps.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Fraction of total steps used for LR warmup.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps for testing. -1 for full epochs.")
    parser.add_argument("--output_dir", type=str, default="output_model", help="Directory to save the fine-tuned model.")
    parser.add_argument("--use_gpu", action="store_true", help="Use NVIDIA GPU and Gemini Plugin if available.")
    args = parser.parse_args()

    # 1. Initialize distributed environment
    import os
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        import random
        os.environ["MASTER_PORT"] = str(random.randint(20000, 29999))

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo" if not (args.use_gpu and torch.cuda.is_available()) else "nccl")
    
    # 2. Get components
    if args.model_name:
        model_id = args.model_name
    else:
        model_id = "Qwen/Qwen1.5-0.5B" if args.model_size == "small" else "deepseek-ai/deepseek-coder-1.3b-base"
        
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = get_tokenizer(model_id)
    
    print(f"Loading model {model_id}...")
    model = get_model(model_id)
        
    # Apply LoRA (Parameter-Efficient Fine-Tuning)
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("Applying LoRA to the model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    dataset = SocialMediaMarketingDataset("data/data.csv", tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Choose Plugin and Device
    if args.use_gpu and torch.cuda.is_available():
        print("Using NVIDIA GPU with GeminiPlugin...")
        plugin = GeminiPlugin(precision='fp16', initial_scale=2**16)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Mac Apple Silicon GPU (MPS) for fine-tuning...")
        plugin = None
        model = model.to("mps")
    else:
        print("Using CPU for fine-tuning...")
        plugin = None
        model = model.to("cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    total_steps = len(dataloader) * args.epochs // args.grad_accum_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    print(f"Training plan: {total_steps} optimizer steps ({warmup_steps} warmup), effective batch size = {args.batch_size * args.grad_accum_steps}")
        
    # 4. Initialize Booster
    booster = Booster(plugin=plugin)
    
    # 5. Boost components
    model, optimizer, _, dataloader, _ = booster.boost(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader
    )
    
    # 6. Training Loop
    model.train()
    print("Starting training...")
    global_step = 0
    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(dataloader):
            if args.use_gpu and torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                batch = {k: v.to("mps") for k, v in batch.items()}
            else:
                batch = {k: v.cpu() for k, v in batch.items()}
                
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            
            if plugin is None:
                loss.backward()
            else:
                booster.backward(loss, optimizer)
            
            epoch_loss += loss.item() * args.grad_accum_steps
            num_batches += 1
            
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                if global_step % 5 == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item() * args.grad_accum_steps:.4f} | Avg: {avg_loss:.4f} | LR: {current_lr:.2e}")
                
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Reached max steps: {args.max_steps}")
                break
        
        # End of epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"--- Epoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f} ---")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"    New best loss! Saving checkpoint...")
            # Save best checkpoint
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            if plugin is not None:
                unwrapped_model = booster.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir)
            else:
                model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            # Remove auto-generated README.md from peft
            peft_readme = os.path.join(args.output_dir, "README.md")
            if os.path.exists(peft_readme):
                os.remove(peft_readme)
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
                
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to {args.output_dir}")
    
if __name__ == "__main__":
    main()
