# Layer-specific evaluation techniques for fine-tuned models
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
import numpy as np
import matplotlib.pyplot as plt

# Load model and tokenizer
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={'': torch.cuda.current_device()},
    load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load fine-tuned model
model_with_adapter = PeftModel.from_pretrained(model, "Meta-Llama-3.1-8B-Instruct-finetuned/checkpoint-80")
model_with_adapter.eval()

class LayerAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_outputs = {}
        self.hooks = []
        
    def register_hooks(self, layer_indices=None):
        """Register forward hooks to capture layer outputs"""
        if layer_indices is None:
            layer_indices = [0, 8, 16, 24, 31]  # Sample layers for Llama
            
        for layer_idx in layer_indices:
            if layer_idx < len(self.model.base_model.model.layers):
                layer = self.model.base_model.model.layers[layer_idx]
                
                def make_hook(idx):
                    def hook(module, input, output):
                        self.layer_outputs[f'layer_{idx}'] = output[0].detach().cpu()
                    return hook
                
                hook_handle = layer.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook_handle)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_layer_activations(self, texts, layer_indices=None):
        """Analyze activations for specific layers"""
        if layer_indices is None:
            layer_indices = [0, 8, 16, 24, 31]
            
        self.register_hooks(layer_indices)
        
        results = {'layer_stats': {}}
        
        for text in texts[:3]:  # Analyze first 3 examples
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Analyze each captured layer
            for layer_name, activations in self.layer_outputs.items():
                if layer_name not in results['layer_stats']:
                    results['layer_stats'][layer_name] = []
                
                stats = {
                    'mean': activations.mean().item(),
                    'std': activations.std().item(),
                    'max': activations.max().item(),
                    'min': activations.min().item(),
                    'sparsity': (activations == 0).float().mean().item()
                }
                results['layer_stats'][layer_name].append(stats)
        
        self.remove_hooks()
        return results
    
    def evaluate_layer_importance(self, texts, target_layers=None):
        """Evaluate importance of different layers by masking"""
        if target_layers is None:
            target_layers = [0, 8, 16, 24, 31]
        
        baseline_losses = []
        masked_losses = {layer: [] for layer in target_layers}
        
        for text in texts[:3]:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Baseline loss
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                baseline_loss = outputs.loss.item()
                baseline_losses.append(baseline_loss)
            
            # Test each layer by zeroing its output
            for layer_idx in target_layers:
                if layer_idx >= len(self.model.base_model.model.layers):
                    continue
                    
                def zero_layer_hook(module, input, output):
                    return torch.zeros_like(output[0]), *output[1:]
                
                layer = self.model.base_model.model.layers[layer_idx]
                hook = layer.register_forward_hook(zero_layer_hook)
                
                try:
                    with torch.no_grad():
                        outputs = self.model(**inputs, labels=inputs['input_ids'])
                        masked_loss = outputs.loss.item()
                        masked_losses[layer_idx].append(masked_loss)
                except:
                    masked_losses[layer_idx].append(float('inf'))
                
                hook.remove()
        
        # Calculate importance scores
        importance_scores = {}
        baseline_avg = np.mean(baseline_losses)
        
        for layer_idx in target_layers:
            if masked_losses[layer_idx]:
                masked_avg = np.mean([l for l in masked_losses[layer_idx] if l != float('inf')])
                importance_scores[f'layer_{layer_idx}'] = masked_avg - baseline_avg
        
        return {
            'baseline_loss': baseline_avg,
            'importance_scores': importance_scores
        }

# Example usage
if __name__ == "__main__":
    analyzer = LayerAnalyzer(model_with_adapter, tokenizer)
    
    test_texts = [
        "How does a brokerage firm work?",
        "What are the key financial metrics?",
        "Explain investment portfolio diversification."
    ]
    
    print("1. Analyzing layer activations...")
    activation_results = analyzer.analyze_layer_activations(test_texts)
    
    print("2. Evaluating layer importance...")
    importance_results = analyzer.evaluate_layer_importance(test_texts)
    
    print("\n=== Layer Analysis Results ===")
    print(f"Baseline Loss: {importance_results['baseline_loss']:.4f}")
    print("\nLayer Importance Scores (loss increase when masked):")
    for layer, score in importance_results['importance_scores'].items():
        print(f"  {layer}: {score:.4f}")
    
    print("\nLayer Activation Statistics:")
    for layer_name, stats_list in activation_results['layer_stats'].items():
        avg_stats = {
            'mean': np.mean([s['mean'] for s in stats_list]),
            'std': np.mean([s['std'] for s in stats_list]),
            'sparsity': np.mean([s['sparsity'] for s in stats_list])
        }
        print(f"  {layer_name}: Mean={avg_stats['mean']:.3f}, Std={avg_stats['std']:.3f}, Sparsity={avg_stats['sparsity']:.3f}")