"""
Evaluation utilities for Event-Enriched Image Captioning
=======================================================

This module provides evaluation metrics for caption quality assessment.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from .logging_utils import get_logger

logger = get_logger(__name__)


def calculate_cider(predictions: List[str], references: List[List[str]]) -> float:
    """
    Calculate CIDEr score for caption evaluation.
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions (each item is a list of references)
        
    Returns:
        CIDEr score
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # Format data for CIDEr evaluation
        gts = {}
        res = {}
        
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            gts[i] = [{'caption': ref} for ref in refs]
            res[i] = [{'caption': pred}]
        
        # Calculate CIDEr
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        
        return float(score)
        
    except ImportError:
        logger.warning("pycocoevalcap not available. Install it for CIDEr evaluation.")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating CIDEr: {str(e)}")
        return 0.0


def calculate_bleu(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Calculate BLEU scores for caption evaluation.
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions (each item is a list of references)
        
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        
        # Format data for BLEU evaluation
        gts = {}
        res = {}
        
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            gts[i] = [{'caption': ref} for ref in refs]
            res[i] = [{'caption': pred}]
        
        # Calculate BLEU
        bleu_scorer = Bleu(4)
        scores, _ = bleu_scorer.compute_score(gts, res)
        
        return {
            'BLEU-1': float(scores[0]),
            'BLEU-2': float(scores[1]),
            'BLEU-3': float(scores[2]),
            'BLEU-4': float(scores[3])
        }
        
    except ImportError:
        logger.warning("pycocoevalcap not available. Install it for BLEU evaluation.")
        return {'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0}
    except Exception as e:
        logger.error(f"Error calculating BLEU: {str(e)}")
        return {'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0}


def calculate_rouge(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Calculate ROUGE scores for caption evaluation.
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions (each item is a list of references)
        
    Returns:
        Dictionary with ROUGE-L score
    """
    try:
        from pycocoevalcap.rouge.rouge import Rouge
        
        # Format data for ROUGE evaluation
        gts = {}
        res = {}
        
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            gts[i] = [{'caption': ref} for ref in refs]
            res[i] = [{'caption': pred}]
        
        # Calculate ROUGE
        rouge_scorer = Rouge()
        score, _ = rouge_scorer.compute_score(gts, res)
        
        return {'ROUGE-L': float(score)}
        
    except ImportError:
        logger.warning("pycocoevalcap not available. Install it for ROUGE evaluation.")
        return {'ROUGE-L': 0.0}
    except Exception as e:
        logger.error(f"Error calculating ROUGE: {str(e)}")
        return {'ROUGE-L': 0.0}


def evaluate_captions(
    predictions_file: str,
    ground_truth_file: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of generated captions.
    
    Args:
        predictions_file: Path to CSV file with predictions
        ground_truth_file: Path to ground truth file (optional)
        output_file: Path to save evaluation results (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating captions from {predictions_file}")
    
    # Load predictions
    df_pred = pd.read_csv(predictions_file)
    
    if 'generated_caption' not in df_pred.columns:
        raise ValueError("Predictions file must contain 'generated_caption' column")
    
    results = {
        'total_predictions': len(df_pred),
        'empty_predictions': len(df_pred[df_pred['generated_caption'].isna() | 
                                        (df_pred['generated_caption'] == '')]),
        'average_length': df_pred['generated_caption'].str.len().mean(),
        'metrics': {}
    }
    
    # If ground truth is available, calculate metrics
    if ground_truth_file and os.path.exists(ground_truth_file):
        logger.info("Ground truth available, calculating metrics...")
        
        # Load ground truth
        df_gt = pd.read_csv(ground_truth_file)
        
        # Merge predictions with ground truth
        df_merged = df_pred.merge(df_gt, on='query_id', how='inner')
        
        if len(df_merged) == 0:
            logger.warning("No matching query_ids found between predictions and ground truth")
            return results
        
        # Prepare data for evaluation
        predictions = df_merged['generated_caption'].fillna('').tolist()
        
        # Assuming ground truth has multiple reference columns or a single reference
        if 'reference_caption' in df_merged.columns:
            references = [[ref] for ref in df_merged['reference_caption'].fillna('').tolist()]
        else:
            # Look for multiple reference columns
            ref_columns = [col for col in df_merged.columns if col.startswith('reference')]
            if ref_columns:
                references = []
                for _, row in df_merged.iterrows():
                    refs = [str(row[col]) for col in ref_columns if pd.notna(row[col]) and str(row[col]).strip()]
                    references.append(refs if refs else [''])
            else:
                logger.warning("No reference captions found in ground truth file")
                return results
        
        # Calculate metrics
        logger.info("Calculating CIDEr score...")
        cider_score = calculate_cider(predictions, references)
        results['metrics']['CIDEr'] = cider_score
        
        logger.info("Calculating BLEU scores...")
        bleu_scores = calculate_bleu(predictions, references)
        results['metrics'].update(bleu_scores)
        
        logger.info("Calculating ROUGE score...")
        rouge_scores = calculate_rouge(predictions, references)
        results['metrics'].update(rouge_scores)
        
        logger.info(f"Evaluation complete. CIDEr: {cider_score:.4f}")
    
    else:
        logger.info("No ground truth provided, skipping metric calculation")
    
    # Save results if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {output_file}")
    
    return results


def compare_models(results_files: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Compare evaluation results from multiple models.
    
    Args:
        results_files: List of paths to evaluation result JSON files
        output_file: Path to save comparison table (optional)
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for file_path in results_files:
        if not os.path.exists(file_path):
            logger.warning(f"Results file not found: {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        model_name = os.path.basename(file_path).replace('.json', '')
        
        row = {'Model': model_name}
        row.update(results.get('metrics', {}))
        row['Total_Predictions'] = results.get('total_predictions', 0)
        row['Empty_Predictions'] = results.get('empty_predictions', 0)
        row['Average_Length'] = results.get('average_length', 0)
        
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    if output_file:
        df_comparison.to_csv(output_file, index=False)
        logger.info(f"Model comparison saved to {output_file}")
    
    return df_comparison