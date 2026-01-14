"""
Utility Functions
Helper functions for logging, saving, etc.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(output_dir):
    """
    Setup logging configuration
    
    Args:
        output_dir: Directory for log files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    log_file = output_path / f'assignment2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 80)
    logging.info("Assignment 2: Transformer-Based ML Pipeline")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")

def save_results(comparison_df, bert_metrics, tabtrans_metrics, 
                baseline_results, output_dir):
    """
    Save all results to files
    
    Args:
        comparison_df: Model comparison dataframe
        bert_metrics: BERT metrics dictionary
        tabtrans_metrics: TabTransformer metrics dictionary
        baseline_results: Baseline results dictionary
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save comparison table
    comparison_df.to_csv(output_path / 'model_comparison.csv', index=False)
    
    # Save detailed metrics
    metrics_summary = {
        'timestamp': datetime.now().isoformat(),
        'bert': {
            'accuracy': float(bert_metrics['accuracy']),
            'precision': float(bert_metrics['precision']),
            'recall': float(bert_metrics['recall']),
            'f1': float(bert_metrics['f1'])
        },
        'tabtransformer': {
            'accuracy': float(tabtrans_metrics['accuracy']),
            'precision': float(tabtrans_metrics['precision']),
            'recall': float(tabtrans_metrics['recall']),
            'f1': float(tabtrans_metrics['f1'])
        },
        'baseline': {}
    }
    
    # Add baseline metrics
    for model_name, metrics in baseline_results.items():
        metrics_summary['baseline'][model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        }
    
    # Save as JSON
    with open(output_path / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Save confusion matrices
    import numpy as np
    
    confusion_matrices = {
        'bert': bert_metrics['confusion_matrix'].tolist(),
        'tabtransformer': tabtrans_metrics['confusion_matrix'].tolist()
    }
    
    for model_name, metrics in baseline_results.items():
        confusion_matrices[model_name.lower().replace(' ', '_')] = \
            metrics['confusion_matrix'].tolist()
    
    with open(output_path / 'confusion_matrices.json', 'w') as f:
        json.dump(confusion_matrices, f, indent=2)
    
    print(f"      ✓ Saved model_comparison.csv")
    print(f"      ✓ Saved metrics_summary.json")
    print(f"      ✓ Saved confusion_matrices.json")

def print_ascii_banner():
    """Print ASCII art banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║          ASSIGNMENT 2: TRANSFORMER-BASED ML PIPELINE                     ║
    ║          Phishing Website Detection using BERT & TabTransformer          ║
    ║                                                                           ║
    ║          Author: Gurmandeep Deol                                         ║
    ║          Course: SRT521 - Advanced Data Analysis for Security            ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def format_time(seconds):
    """
    Format time in human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def check_gpu_availability():
    """
    Check GPU availability and print info
    
    Returns:
        Boolean indicating GPU availability
    """
    import torch
    
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print(f"⚠️  GPU not available, using CPU")
        print(f"   Training will be slower")
        return False

def create_directory_structure(base_dir):
    """
    Create directory structure for outputs
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Dictionary of directory paths
    """
    base_path = Path(base_dir)
    
    dirs = {
        'base': base_path,
        'models': base_path / 'models',
        'plots': base_path / 'plots',
        'logs': base_path / 'logs',
        'data': base_path / 'data'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True, parents=True)
    
    return dirs