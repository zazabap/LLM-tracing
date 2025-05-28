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

def analyze_specific_layers(model, tokenizer, input_text, target_layers=[0, 8, 16, 24, 31], generate_text=True):
    """Analyze activation patterns in specific layers and optionally generate text"""
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
    
    # Generate text if requested
    generated_text = None
    if generate_text:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        with torch.no_grad():
            _ = model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return layer_activations, generated_text

def compare_layer_outputs(model, tokenizer, texts, target_layers=[0, 8, 16, 24, 31]):
    """Compare layer outputs across different inputs"""
    all_results = []
    
    for i, text in enumerate(texts):
        print(f"\n{'='*60}")
        print(f"Analyzing text {i+1}: {text}")
        print('='*60)
        
        results, generated_text = analyze_specific_layers(model, tokenizer, text, target_layers)
        all_results.append((results, generated_text))
        
        print(f"\n📝 Generated Response:")
        print(f"Input: {text}")
        print(f"Output: {generated_text}")
        
        print(f"\n📊 Layer Statistics:")
        for layer_name, stats in results.items():
            print(f"  {layer_name}: Mean={stats['mean_activation']:.3f}, "
                  f"Std={stats['std_activation']:.3f}, "
                  f"Sparsity={stats['sparsity']:.3f}")
    
    return all_results

def evaluate_layer_importance_by_masking(model, tokenizer, text, target_layers=[0, 8, 16, 24, 31]):
    """Evaluate layer importance by masking each layer and showing generation changes"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Baseline generation
    with torch.no_grad():
        baseline_outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
    
    print(f"\n🔍 Layer Masking Analysis")
    print(f"📝 Baseline output: {baseline_text}")
    print(f"{'='*80}")
    
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
                masked_outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
            masked_text = tokenizer.decode(masked_outputs[0], skip_special_tokens=True)
            
            # Calculate similarity metrics
            baseline_tokens = set(baseline_text.split())
            masked_tokens = set(masked_text.split())
            
            jaccard_sim = len(baseline_tokens & masked_tokens) / len(baseline_tokens | masked_tokens) if baseline_tokens | masked_tokens else 0
            length_ratio = len(masked_text) / len(baseline_text) if baseline_text else 0
            
            # Calculate token-level differences
            baseline_words = baseline_text.split()
            masked_words = masked_text.split()
            
            layer_importance[f'layer_{layer_idx}'] = {
                'jaccard_similarity': jaccard_sim,
                'length_ratio': length_ratio,
                'output_text': masked_text,
                'baseline_length': len(baseline_words),
                'masked_length': len(masked_words)
            }
            
            print(f"\n🎯 Layer {layer_idx} masked:")
            print(f"   Similarity: {jaccard_sim:.3f} | Length ratio: {length_ratio:.3f}")
            print(f"   Output: {masked_text}")
            
        except Exception as e:
            print(f"❌ Error processing layer {layer_idx}: {e}")
            layer_importance[f'layer_{layer_idx}'] = {'error': str(e)}
        finally:
            hook.remove()
    
    return layer_importance, baseline_text

# Run layer analysis
test_texts = [
    "How does a brokerage firm work?",
    "What are the key financial metrics for evaluating a company?",
    "Explain the concept of diversification in investment portfolios."
]

print("1. Comparing layer outputs across different inputs...")
layer_comparison = compare_layer_outputs(model_with_adapter, tokenizer, test_texts)

print("\n2. Evaluating layer importance by masking...")
importance_results, baseline = evaluate_layer_importance_by_masking(
    model_with_adapter, tokenizer, "How does a brokerage firm work?"
)

print(f"\n📊 Layer Importance Summary:")
print(f"Baseline: {baseline}")
print(f"{'='*80}")
for layer_name, metrics in importance_results.items():
    if 'error' not in metrics:
        print(f"{layer_name}:")
        print(f"  📈 Similarity: {metrics['jaccard_similarity']:.3f}")
        print(f"  📏 Length ratio: {metrics['length_ratio']:.3f}")
        print(f"  📝 Output: {metrics['output_text'][:150]}...")
    else:
        print(f"{layer_name}: ❌ {metrics['error']}")

print("\n3. Analyzing generation quality across layers...")
# Test how different layers affect generation quality
def test_generation_quality(model, tokenizer, prompts, target_layers=[0, 8, 16, 24, 31]):
    """Test generation quality for different prompts"""
    for prompt in prompts:
        print(f"\n🎯 Testing prompt: '{prompt}'")
        print("-" * 60)
        
        _, generated = analyze_specific_layers(model, tokenizer, prompt, target_layers, generate_text=True)
        print(f"🤖 Generated: {generated}")

quality_test_prompts = [
    "What is the difference between stocks and bonds?",
    "Explain risk management in finance.",
    "How do interest rates affect the economy?"
]

test_generation_quality(model_with_adapter, tokenizer, quality_test_prompts)

print("\n4. Comparing Base Model vs Fine-tuned Model...")
def compare_base_vs_finetuned(base_model, finetuned_model, tokenizer, prompts):
    """Compare generations between base model and fine-tuned model"""
    for prompt in prompts:
        print(f"\n🔄 Comparing models for: '{prompt}'")
        print("=" * 70)
        
        # Base model generation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = next(base_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            base_outputs = base_model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # Fine-tuned model generation
        with torch.no_grad():
            ft_outputs = finetuned_model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        ft_text = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
        
        print(f"🏗️  Base Model: {base_text}")
        print(f"🎯 Fine-tuned: {ft_text}")
        
        # Calculate differences
        base_tokens = set(base_text.split())
        ft_tokens = set(ft_text.split())
        similarity = len(base_tokens & ft_tokens) / len(base_tokens | ft_tokens) if base_tokens | ft_tokens else 0
        
        print(f"📊 Similarity: {similarity:.3f}")

comparison_prompts = [
    "How does a brokerage firm work?",
    "What are the benefits of diversification?",
    "Explain compound interest."
]

compare_base_vs_finetuned(model, model_with_adapter, tokenizer, comparison_prompts)