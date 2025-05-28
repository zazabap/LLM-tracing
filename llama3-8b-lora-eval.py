# Load the fine-tuned model with the adapter attached
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training
import numpy as np
from transformers import DataCollatorWithPadding,DataCollatorForLanguageModeling, Trainer, TrainingArguments

model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # Example of a 8-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={'': torch.cuda.current_device()},
    load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load fine-tuned model
model_with_adapter = PeftModel.from_pretrained(model, "Meta-Llama-3.1-8B-Instruct-finetuned/checkpoint-80").to("cuda")
model_with_adapter.eval()

# Basic inference test
inputs = tokenizer("How does a brokerage firm work?", return_tensors="pt")

outputs = model_with_adapter.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=100)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

# Layer-specific evaluation
print("\n=== Layer-Specific Analysis ===")

def analyze_specific_layers(model, tokenizer, input_text, target_layers=[0, 8, 16, 24, 31]):
    """Analyze activation patterns in specific layers"""
    layer_activations = {}
    
    def create_hook(layer_name):
        def hook(module, input, output):
            # Store the hidden states
            layer_activations[layer_name] = {
                'hidden_states': output[0].detach().cpu(),
                'mean_activation': output[0].mean().item(),
                'max_activation': output[0].max().item(),
                'std_activation': output[0].std().item(),
                'sparsity': (output[0] == 0).float().mean().item()
            }
        return hook
    
    # Register hooks for target layers
    hooks = []
    for layer_idx in target_layers:
        # Access layers through the correct path for Llama models
        if hasattr(model, 'base_model'):
            # For PEFT models
            layers = model.base_model.model.model.layers
        else:
            # For base models
            layers = model.model.layers
            
        if layer_idx < len(layers):
            layer = layers[layer_idx]
            hook = layer.register_forward_hook(create_hook(f'layer_{layer_idx}'))
            hooks.append(hook)
    
    # Run inference
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return layer_activations

def compare_layer_outputs(model, tokenizer, texts, target_layers=[0, 8, 16, 24, 31]):
    """Compare layer outputs across different inputs"""
    all_results = []
    
    for i, text in enumerate(texts):
        print(f"\nAnalyzing text {i+1}: {text[:50]}...")
        results = analyze_specific_layers(model, tokenizer, text, target_layers)
        all_results.append(results)
        
        print("Layer Statistics:")
        for layer_name, stats in results.items():
            print(f"  {layer_name}: Mean={stats['mean_activation']:.3f}, "
                  f"Std={stats['std_activation']:.3f}, "
                  f"Sparsity={stats['sparsity']:.3f}")
    
    return all_results

def evaluate_layer_importance_by_masking(model, tokenizer, text, target_layers=[0, 8, 16, 24, 31]):
    """Evaluate layer importance by masking each layer"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Baseline generation
    with torch.no_grad():
        baseline_outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
    
    print(f"Baseline output: {baseline_text}")
    
    layer_importance = {}
    
    for layer_idx in target_layers:
        # Access layers through the correct path for Llama models
        if hasattr(model, 'base_model'):
            # For PEFT models
            layers = model.base_model.model.model.layers
        else:
            # For base models
            layers = model.model.layers
            
        if layer_idx >= len(layers):
            continue
            
        # Create a hook that zeros out the layer output
        def zero_hook(module, input, output):
            return (torch.zeros_like(output[0]),) + output[1:]
        
        layer = layers[layer_idx]
        hook = layer.register_forward_hook(zero_hook)
        
        try:
            with torch.no_grad():
                masked_outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            masked_text = tokenizer.decode(masked_outputs[0], skip_special_tokens=True)
            
            # Calculate similarity (simple length and token overlap)
            baseline_tokens = set(baseline_text.split())
            masked_tokens = set(masked_text.split())
            
            jaccard_sim = len(baseline_tokens & masked_tokens) / len(baseline_tokens | masked_tokens) if baseline_tokens | masked_tokens else 0
            length_ratio = len(masked_text) / len(baseline_text) if baseline_text else 0
            
            layer_importance[f'layer_{layer_idx}'] = {
                'jaccard_similarity': jaccard_sim,
                'length_ratio': length_ratio,
                'output_text': masked_text[:100] + "..." if len(masked_text) > 100 else masked_text
            }
            
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")
            layer_importance[f'layer_{layer_idx}'] = {'error': str(e)}
        finally:
            hook.remove()
    
    return layer_importance

# Run layer analysis
test_texts = [
    "How does a brokerage firm work?",
    "What are the key financial metrics for evaluating a company?",
    "Explain the concept of diversification in investment portfolios."
]

print("1. Comparing layer outputs across different inputs...")
layer_comparison = compare_layer_outputs(model_with_adapter, tokenizer, test_texts)

print("\n2. Evaluating layer importance by masking...")
importance_results = evaluate_layer_importance_by_masking(
    model_with_adapter, tokenizer, "How does a brokerage firm work?"
)

print("\nLayer Importance Analysis:")
for layer_name, metrics in importance_results.items():
    if 'error' not in metrics:
        print(f"{layer_name}:")
        print(f"  Jaccard Similarity: {metrics['jaccard_similarity']:.3f}")
        print(f"  Length Ratio: {metrics['length_ratio']:.3f}")
        print(f"  Sample Output: {metrics['output_text']}")
    else:
        print(f"{layer_name}: {metrics['error']}")

# Analyze LoRA-specific layers (adapter layers)
print("\n=== LoRA Adapter Analysis ===")
def analyze_lora_adapters(model):
    """Analyze the LoRA adapter parameters"""
    adapter_info = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_A = module.lora_A.default.weight.data
                lora_B = module.lora_B.default.weight.data
                
                adapter_info[name] = {
                    'lora_A_shape': lora_A.shape,
                    'lora_B_shape': lora_B.shape,
                    'lora_A_norm': lora_A.norm().item(),
                    'lora_B_norm': lora_B.norm().item(),
                    'rank': lora_A.shape[0]
                }
    
    return adapter_info

lora_analysis = analyze_lora_adapters(model_with_adapter)
print("LoRA Adapter Statistics:")
for adapter_name, stats in lora_analysis.items():
    print(f"  {adapter_name}:")
    print(f"    Rank: {stats['rank']}")
    print(f"    LoRA A norm: {stats['lora_A_norm']:.3f}")
    print(f"    LoRA B norm: {stats['lora_B_norm']:.3f}")

# Debug: Print model structure to understand layer access
def print_model_structure(model, max_depth=3):
    """Print the model structure to understand how to access layers"""
    def _print_structure(obj, name="", depth=0, max_depth=3):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        if hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            print(f"{indent}{name}: {class_name}")
            
            # For modules, print their named children
            if hasattr(obj, 'named_children'):
                for child_name, child in obj.named_children():
                    if child_name in ['layers', 'model', 'base_model'] or 'layer' in child_name.lower():
                        _print_structure(child, child_name, depth + 1, max_depth)
                    elif depth < 2:  # Only show top-level for first 2 depths
                        _print_structure(child, child_name, depth + 1, max_depth)
    
    print("=== Model Structure ===")
    _print_structure(model, "model", 0, max_depth)
    
    # Try to access layers and print count
    try:
        if hasattr(model, 'base_model'):
            layers = model.base_model.model.model.layers
            print(f"\nFound {len(layers)} layers via: model.base_model.model.model.layers")
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers  
            print(f"\nFound {len(layers)} layers via: model.model.layers")
        else:
            print("\nCould not find layers!")
    except Exception as e:
        print(f"\nError accessing layers: {e}")

print_model_structure(model_with_adapter)