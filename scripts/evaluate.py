#!/usr/bin/env python3
"""
Evaluation script for Event-Enriched Image Captioning
====================================================

This script evaluates generated captions against ground truth using
various metrics like CIDEr, BLEU, and ROUGE.

Usage:
    python scripts/evaluate.py --predictions outputs/submissions/submission_final.csv --ground-truth data/processed/train/gt_train.csv
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.evaluation import evaluate_captions, compare_models
from src.utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Event-Enriched Image Captions")
    parser.add_argument(
        "--predictions", 
        type=str, 
        required=True,
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--ground-truth", 
        type=str,
        help="Path to ground truth CSV file (optional)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Path to save evaluation results (optional)"
    )
    parser.add_argument(
        "--compare", 
        type=str, 
        nargs="+",
        help="Paths to multiple result files for comparison"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="DEBUG" if args.verbose else "INFO")
    
    try:
        if args.compare:
            # Compare multiple models
            logger.info("Comparing multiple models...")
            comparison_df = compare_models(
                results_files=args.compare,
                output_file=args.output
            )
            
            print("\n=== Model Comparison ===")
            print(comparison_df.to_string(index=False))
            
        else:
            # Evaluate single model
            logger.info(f"Evaluating predictions from: {args.predictions}")
            
            if not os.path.exists(args.predictions):
                raise FileNotFoundError(f"Predictions file not found: {args.predictions}")
            
            results = evaluate_captions(
                predictions_file=args.predictions,
                ground_truth_file=args.ground_truth,
                output_file=args.output
            )
            
            # Print results
            print("\n=== Evaluation Results ===")
            print(f"Total predictions: {results['total_predictions']}")
            print(f"Empty predictions: {results['empty_predictions']}")
            print(f"Average caption length: {results['average_length']:.2f}")
            
            if results['metrics']:
                print("\n=== Metrics ===")
                for metric, score in results['metrics'].items():
                    print(f"{metric}: {score:.4f}")
            else:
                print("\nNo ground truth provided - metrics not calculated")
            
            if args.output:
                print(f"\nResults saved to: {args.output}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()