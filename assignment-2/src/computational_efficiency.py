"""
Computational Efficiency Analysis Module
Measures and compares training time, inference time, and resource usage
"""

import time
import psutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ComputationalEfficiencyAnalyzer:
    """Analyzer for computational efficiency metrics"""
    
    def __init__(self, output_dir='results'):
        """
        Initialize efficiency analyzer
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = {}
        
    def measure_training_time(self, model_name, training_function, *args, **kwargs):
        """
        Measure training time for a model
        
        Args:
            model_name: Name of the model
            training_function: Function that trains the model
            *args, **kwargs: Arguments for training function
            
        Returns:
            Dictionary with timing metrics
        """
        print(f"\n⏱️  Measuring training time for {model_name}...")
        
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 ** 3)  # GB
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        else:
            initial_gpu_memory = None
        
        # Measure training time
        start_time = time.time()
        result = training_function(*args, **kwargs)
        training_time = time.time() - start_time
        
        # Get final resource usage
        final_memory = process.memory_info().rss / (1024 ** 3)  # GB
        memory_used = final_memory - initial_memory
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            gpu_memory_used = final_gpu_memory - initial_gpu_memory
        else:
            gpu_memory_used = None
        
        metrics = {
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'training_time_hours': training_time / 3600,
            'cpu_memory_gb': memory_used,
            'gpu_memory_gb': gpu_memory_used,
            'device': 'GPU' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'CPU'
        }
        
        self.metrics[model_name] = metrics
        
        print(f"   ✓ Training time: {training_time/60:.2f} minutes")
        print(f"   ✓ CPU Memory used: {memory_used:.2f} GB")
        if gpu_memory_used is not None:
            print(f"   ✓ GPU Memory used: {gpu_memory_used:.2f} GB")
        
        return metrics, result
    
    def measure_inference_time(self, model_name, model, X_test, batch_sizes=[1, 16, 32, 64, 128]):
        """
        Measure inference time for different batch sizes
        
        Args:
            model_name: Name of the model
            model: Trained model
            X_test: Test data
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with inference metrics
        """
        print(f"\n⚡ Measuring inference time for {model_name}...")
        
        inference_metrics = {
            'batch_sizes': [],
            'inference_times': [],
            'samples_per_second': []
        }
        
        for batch_size in batch_sizes:
            # Take a subset of data for this batch size
            n_samples = min(batch_size * 10, len(X_test))
            X_subset = X_test[:n_samples]
            
            # Warm-up run
            try:
                if hasattr(model, 'predict'):
                    _ = model.predict(X_subset[:batch_size])
            except:
                continue
            
            # Measure inference time
            start_time = time.time()
            
            try:
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_subset)
                else:
                    continue
            except Exception as e:
                print(f"   Warning: Could not measure batch size {batch_size}: {str(e)}")
                continue
            
            inference_time = time.time() - start_time
            samples_per_second = len(X_subset) / inference_time
            
            inference_metrics['batch_sizes'].append(batch_size)
            inference_metrics['inference_times'].append(inference_time)
            inference_metrics['samples_per_second'].append(samples_per_second)
            
            print(f"   Batch size {batch_size}: {samples_per_second:.2f} samples/sec")
        
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
        
        self.metrics[model_name]['inference'] = inference_metrics
        
        return inference_metrics
    
    def measure_model_size(self, model_name, model, model_path=None):
        """
        Measure model size in memory and on disk
        
        Args:
            model_name: Name of the model
            model: Trained model
            model_path: Path to saved model file
            
        Returns:
            Dictionary with size metrics
        """
        print(f"\n💾 Measuring model size for {model_name}...")
        
        size_metrics = {}
        
        # Memory size (parameters)
        if TORCH_AVAILABLE and hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            num_params = sum(p.numel() for p in model.model.parameters())
            size_metrics['parameters'] = num_params
            size_metrics['memory_mb'] = (num_params * 4) / (1024 ** 2)  # Assuming float32
            print(f"   ✓ Parameters: {num_params:,}")
            print(f"   ✓ Memory size: {size_metrics['memory_mb']:.2f} MB")
        
        # Disk size
        if model_path and Path(model_path).exists():
            if Path(model_path).is_dir():
                total_size = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file())
            else:
                total_size = Path(model_path).stat().st_size
            
            size_metrics['disk_size_mb'] = total_size / (1024 ** 2)
            print(f"   ✓ Disk size: {size_metrics['disk_size_mb']:.2f} MB")
        
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
        
        self.metrics[model_name]['size'] = size_metrics
        
        return size_metrics
    
    def compare_all_models(self):
        """
        Create comparison of all models
        
        Returns:
            DataFrame with comparison
        """
        print(f"\n📊 Comparing computational efficiency across models...")
        
        comparison_data = {
            'Model': [],
            'Training Time (min)': [],
            'CPU Memory (GB)': [],
            'GPU Memory (GB)': [],
            'Parameters': [],
            'Model Size (MB)': [],
            'Inference Speed (samples/sec)': [],
            'Device': []
        }
        
        for model_name, metrics in self.metrics.items():
            comparison_data['Model'].append(model_name)
            comparison_data['Training Time (min)'].append(
                metrics.get('training_time_minutes', 'N/A')
            )
            comparison_data['CPU Memory (GB)'].append(
                f"{metrics.get('cpu_memory_gb', 0):.2f}" if metrics.get('cpu_memory_gb') else 'N/A'
            )
            
            gpu_mem = metrics.get('gpu_memory_gb')
            comparison_data['GPU Memory (GB)'].append(
                f"{gpu_mem:.2f}" if gpu_mem else 'N/A'
            )
            
            size_info = metrics.get('size', {})
            comparison_data['Parameters'].append(
                f"{size_info.get('parameters', 0):,}" if size_info.get('parameters') else 'N/A'
            )
            comparison_data['Model Size (MB)'].append(
                f"{size_info.get('memory_mb', 0):.2f}" if size_info.get('memory_mb') else 'N/A'
            )
            
            inference_info = metrics.get('inference', {})
            if inference_info.get('samples_per_second'):
                avg_speed = np.mean(inference_info['samples_per_second'])
                comparison_data['Inference Speed (samples/sec)'].append(f"{avg_speed:.2f}")
            else:
                comparison_data['Inference Speed (samples/sec)'].append('N/A')
            
            comparison_data['Device'].append(metrics.get('device', 'N/A'))
        
        df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_file = self.output_dir / 'computational_efficiency_comparison.csv'
        df.to_csv(csv_file, index=False)
        print(f"   ✓ Saved comparison to: {csv_file}")
        
        return df
    
    def plot_training_time_comparison(self):
        """Plot training time comparison across models"""
        print(f"\n📈 Creating training time comparison plot...")
        
        models = []
        times = []
        devices = []
        
        for model_name, metrics in self.metrics.items():
            if 'training_time_minutes' in metrics:
                models.append(model_name)
                times.append(metrics['training_time_minutes'])
                devices.append(metrics.get('device', 'CPU'))
        
        if not models:
            print("   ⚠️  No training time data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#2ecc71' if d == 'GPU' else '#3498db' for d in devices]
        bars = ax.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time Comparison Across Models', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.2f}m',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='GPU'),
            Patch(facecolor='#3498db', label='CPU')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plot_file = self.output_dir / 'training_time_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved plot to: {plot_file}")
    
    def plot_inference_speed_comparison(self):
        """Plot inference speed comparison across models"""
        print(f"\n📈 Creating inference speed comparison plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model_name, metrics in self.metrics.items():
            if 'inference' in metrics:
                inference_data = metrics['inference']
                if inference_data['batch_sizes']:
                    ax.plot(
                        inference_data['batch_sizes'],
                        inference_data['samples_per_second'],
                        marker='o',
                        linewidth=2,
                        label=model_name
                    )
        
        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Samples per Second', fontsize=12, fontweight='bold')
        ax.set_title('Inference Speed vs Batch Size', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'inference_speed_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved plot to: {plot_file}")
    
    def plot_resource_usage(self):
        """Plot memory usage comparison"""
        print(f"\n📈 Creating resource usage comparison plot...")
        
        models = []
        cpu_memory = []
        gpu_memory = []
        
        for model_name, metrics in self.metrics.items():
            if 'cpu_memory_gb' in metrics:
                models.append(model_name)
                cpu_memory.append(metrics.get('cpu_memory_gb', 0))
                gpu_memory.append(metrics.get('gpu_memory_gb') or 0)
        
        if not models:
            print("   ⚠️  No resource usage data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # CPU Memory
        bars1 = ax1.bar(models, cpu_memory, color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('CPU Memory (GB)', fontsize=12, fontweight='bold')
        ax1.set_title('CPU Memory Usage', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, mem in zip(bars1, cpu_memory):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem:.2f}GB',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # GPU Memory
        if any(gpu_memory):
            bars2 = ax2.bar(models, gpu_memory, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
            ax2.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
            ax2.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, mem in zip(bars2, gpu_memory):
                if mem > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{mem:.2f}GB',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No GPU Usage Data', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        plot_file = self.output_dir / 'resource_usage_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved plot to: {plot_file}")
    
    def save_results(self):
        """Save all efficiency metrics to JSON"""
        output_file = self.output_dir / 'computational_efficiency_metrics.json'
        
        # Convert to serializable format
        serializable_metrics = self._make_serializable(self.metrics)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\n💾 Computational efficiency metrics saved to: {output_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_summary_report(self):
        """Generate text summary of computational efficiency"""
        report = []
        report.append("=" * 80)
        report.append("COMPUTATIONAL EFFICIENCY ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        for model_name, metrics in self.metrics.items():
            report.append(f"\n{model_name}:")
            report.append("-" * 80)
            
            # Training metrics
            if 'training_time_minutes' in metrics:
                report.append(f"Training Time: {metrics['training_time_minutes']:.2f} minutes")
                report.append(f"Device: {metrics.get('device', 'N/A')}")
            
            if 'cpu_memory_gb' in metrics:
                report.append(f"CPU Memory: {metrics['cpu_memory_gb']:.2f} GB")
            
            if metrics.get('gpu_memory_gb'):
                report.append(f"GPU Memory: {metrics['gpu_memory_gb']:.2f} GB")
            
            # Model size
            if 'size' in metrics:
                size_info = metrics['size']
                if 'parameters' in size_info:
                    report.append(f"Parameters: {size_info['parameters']:,}")
                if 'memory_mb' in size_info:
                    report.append(f"Model Size: {size_info['memory_mb']:.2f} MB")
            
            # Inference metrics
            if 'inference' in metrics:
                inference_info = metrics['inference']
                if inference_info.get('samples_per_second'):
                    avg_speed = np.mean(inference_info['samples_per_second'])
                    report.append(f"Avg Inference Speed: {avg_speed:.2f} samples/sec")
            
            report.append("")
        
        # Efficiency rankings
        report.append("\nEFFICIENCY RANKINGS:")
        report.append("-" * 80)
        
        # Training time ranking
        training_times = [(name, m.get('training_time_minutes', float('inf'))) 
                         for name, m in self.metrics.items() 
                         if 'training_time_minutes' in m]
        if training_times:
            training_times.sort(key=lambda x: x[1])
            report.append("\nFastest Training:")
            for i, (name, time_val) in enumerate(training_times, 1):
                report.append(f"  {i}. {name}: {time_val:.2f} minutes")
        
        report.append("\n" + "=" * 80)
        
        summary_text = "\n".join(report)
        
        # Save to file
        summary_file = self.output_dir / 'computational_efficiency_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\n💾 Summary saved to: {summary_file}")
        
        return summary_text