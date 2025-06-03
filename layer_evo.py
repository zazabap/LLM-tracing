import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import pandas as pd
from scipy.stats import entropy
import os
from collections import defaultdict

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LayerEvolutionAnalyzer:
    def __init__(self, base_model_name, tokenizer, test_dataset):
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.checkpoint_data = defaultdict(dict)
        
    def load_model_checkpoint(self, checkpoint_path):
        """Load a specific checkpoint"""
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map={'': torch.cuda.current_device()},
            load_in_8bit=True
        )
        
        if os.path.exists(checkpoint_path):
            model = PeftModel.from_pretrained(base_model, checkpoint_path).to("cuda")
            model.eval()
            return model
        else:
            print(f"Checkpoint {checkpoint_path} not found!")
            return None

    def extract_layer_probabilities(self, model, sample_texts, target_layers=[0, 8, 16, 24, 31]):
        """Extract probability distributions from specific layers"""
        layer_activations = defaultdict(list)
        
        def create_hook(layer_name):
            def hook(module, input, output):
                # Get hidden states and convert to probabilities using softmax
                hidden_states = output[0].detach().cpu()
                # Apply softmax to get probability-like distributions
                probs = torch.softmax(hidden_states, dim=-1)
                layer_activations[layer_name].append({
                    'mean_prob': probs.mean().item(),
                    'std_prob': probs.std().item(),
                    'entropy': entropy(probs.flatten().numpy() + 1e-8),  # Add small epsilon for numerical stability
                    'max_prob': probs.max().item(),
                    'sparsity': (probs < 0.01).float().mean().item(),  # Probability sparsity threshold
                    'top_k_concentration': torch.topk(probs.flatten(), k=100)[0].sum().item()  # Concentration in top-k
                })
            return hook
        
        # Register hooks
        hooks = []
        for layer_idx in target_layers:
            if hasattr(model, 'base_model'):
                layers = model.base_model.model.model.layers
            else:
                layers = model.model.layers
                
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                hook = layer.register_forward_hook(create_hook(f'layer_{layer_idx}'))
                hooks.append(hook)
        
        # Process samples
        for text in sample_texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return layer_activations

    def analyze_checkpoint_evolution(self, checkpoint_dir, checkpoint_range=(10, 80, 10), num_samples=50):
        """Analyze how layers evolve across checkpoints"""
        
        # Get sample texts from test dataset
        sample_texts = []
        for i, sample in enumerate(self.test_dataset):
            if i >= num_samples:
                break
            # Reconstruct text from the sample
            if 'text' in sample:
                sample_texts.append(sample['text'])
            else:
                # If text is not available, decode from input_ids
                decoded = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                sample_texts.append(decoded)
        
        # Analyze each checkpoint
        for checkpoint_step in range(checkpoint_range[0], checkpoint_range[1] + 1, checkpoint_range[2]):
            checkpoint_path = f"{checkpoint_dir}/checkpoint-{checkpoint_step}"
            print(f"Analyzing checkpoint {checkpoint_step}...")
            
            model = self.load_model_checkpoint(checkpoint_path)
            if model is None:
                continue
                
            layer_data = self.extract_layer_probabilities(model, sample_texts)
            
            # Aggregate statistics across samples
            for layer_name, samples in layer_data.items():
                aggregated = {
                    'mean_prob': np.mean([s['mean_prob'] for s in samples]),
                    'std_prob': np.mean([s['std_prob'] for s in samples]),
                    'entropy': np.mean([s['entropy'] for s in samples]),
                    'max_prob': np.mean([s['max_prob'] for s in samples]),
                    'sparsity': np.mean([s['sparsity'] for s in samples]),
                    'top_k_concentration': np.mean([s['top_k_concentration'] for s in samples])
                }
                self.checkpoint_data[checkpoint_step][layer_name] = aggregated
            
            # Clean up
            del model
            torch.cuda.empty_cache()

    def visualize_layer_evolution(self, save_dir="layer_evolution_plots"):
        """Create comprehensive visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data for plotting
        checkpoints = sorted(self.checkpoint_data.keys())
        layer_names = list(next(iter(self.checkpoint_data.values())).keys())
        
        # 1. Probability Evolution Over Training
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        metrics = ['mean_prob', 'std_prob', 'entropy', 'max_prob', 'sparsity', 'top_k_concentration']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            for layer_name in layer_names:
                values = [self.checkpoint_data[ckpt][layer_name][metric] for ckpt in checkpoints]
                ax.plot(checkpoints, values, marker='o', label=layer_name, linewidth=2)
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/layer_evolution_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Heatmap of Layer Behavior
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create matrix: rows = layers, columns = checkpoints
            matrix = []
            for layer_name in layer_names:
                row = [self.checkpoint_data[ckpt][layer_name][metric] for ckpt in checkpoints]
                matrix.append(row)
            
            matrix = np.array(matrix)
            
            # Create heatmap
            sns.heatmap(matrix, 
                       xticklabels=[f"Step {ckpt}" for ckpt in checkpoints],
                       yticklabels=layer_names,
                       annot=True, 
                       fmt='.3f',
                       cmap='viridis',
                       ax=ax)
            
            ax.set_title(f'{metric.replace("_", " ").title()} Across Layers and Training Steps')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Layer')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/heatmap_{metric}.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Layer-wise Comparison
        fig, axes = plt.subplots(len(layer_names), 1, figsize=(12, 4 * len(layer_names)))
        if len(layer_names) == 1:
            axes = [axes]
            
        for idx, layer_name in enumerate(layer_names):
            ax = axes[idx]
            
            for metric in ['entropy', 'sparsity', 'top_k_concentration']:
                values = [self.checkpoint_data[ckpt][layer_name][metric] for ckpt in checkpoints]
                ax.plot(checkpoints, values, marker='o', label=metric, linewidth=2)
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Value')
            ax.set_title(f'{layer_name} Behavior Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/layer_wise_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Summary Statistics Table
        self.create_summary_table(save_dir)

    def create_summary_table(self, save_dir):
        """Create a summary table of key insights"""
        checkpoints = sorted(self.checkpoint_data.keys())
        layer_names = list(next(iter(self.checkpoint_data.values())).keys())
        
        summary_data = []
        
        for layer_name in layer_names:
            # Calculate trends
            entropy_trend = self.calculate_trend([self.checkpoint_data[ckpt][layer_name]['entropy'] for ckpt in checkpoints])
            sparsity_trend = self.calculate_trend([self.checkpoint_data[ckpt][layer_name]['sparsity'] for ckpt in checkpoints])
            
            final_entropy = self.checkpoint_data[checkpoints[-1]][layer_name]['entropy']
            final_sparsity = self.checkpoint_data[checkpoints[-1]][layer_name]['sparsity']
            
            summary_data.append({
                'Layer': layer_name,
                'Final Entropy': f"{final_entropy:.3f}",
                'Entropy Trend': entropy_trend,
                'Final Sparsity': f"{final_sparsity:.3f}",
                'Sparsity Trend': sparsity_trend
            })
        
        df = pd.DataFrame(summary_data)
        print("\n📊 Layer Evolution Summary:")
        print("=" * 60)
        print(df.to_string(index=False))
        
        # Save to file
        df.to_csv(f"{save_dir}/layer_evolution_summary.csv", index=False)

    def calculate_trend(self, values):
        """Calculate if values are increasing, decreasing, or stable"""
        if len(values) < 2:
            return "Stable"
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        diff = (second_half - first_half) / first_half if first_half != 0 else 0
        
        if diff > 0.05:
            return "↗ Increasing"
        elif diff < -0.05:
            return "↘ Decreasing"
        else:
            return "→ Stable"

# Usage script
def main():
    # Load your dataset preparation (same as in your training script)
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset exactly as in your training
    dataset = load_dataset("poornima9348/finance-alpaca-1k-test")
    train_test_split = dataset['test'].train_test_split(test_size=0.2, shuffle=True)
    test_dataset = train_test_split['test']
    
    # Remove columns and build prompts
    test_dataset = test_dataset.remove_columns(['input', 'text'])
    
    def prompt_builder(row):
        return {"text": row["instruction"] + row["output"]}
    
    test_dataset = test_dataset.map(prompt_builder)
    
    # Initialize analyzer
    analyzer = LayerEvolutionAnalyzer(model_name, tokenizer, test_dataset)
    
    # Analyze evolution from checkpoint-10 to checkpoint-80
    print("🔍 Analyzing layer evolution across training checkpoints...")
    analyzer.analyze_checkpoint_evolution(
        checkpoint_dir="Meta-Llama-3.1-8B-Instruct-finetuned",
        checkpoint_range=(10, 80, 10),  # start, end, step
        num_samples=30  # Use 30 test samples for analysis
    )
    
    # Create visualizations
    print("📊 Creating visualizations...")
    analyzer.visualize_layer_evolution()
    
    print("✅ Analysis complete! Check the 'layer_evolution_plots' directory for results.")

if __name__ == "__main__":
    main()