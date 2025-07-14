"""
Visual context extraction for Event-Enriched Image Captioning
============================================================

This module provides utilities for extracting visual context from images
using Qwen-VL models.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
from PIL import Image
from tqdm import tqdm

from ..models.qwen_models import QwenVLModel
from ..utils.logging_utils import get_logger, LoggerMixin

logger = get_logger(__name__)


class VisualExtractor(LoggerMixin):
    """
    Visual context extractor using Qwen-VL model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visual extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model = QwenVLModel(config)
        self.model.load_model()
        
        # Default prompt for visual analysis
        self.default_prompt = self.config.get('prompts', {}).get('visual_analysis', """
You are a visual analyst. The following image is taken from a CNN news article. Please provide a detailed and comprehensive description.

Your description should cover:
1. **Objective Description:** What do you see in the image? Describe the people, objects, setting, and any text visible.
2. **Contextual Inference:** Based on the visual cues and the fact that this is from CNN, what could the news story be about? What is the likely location or event?
3. **Overall Mood and Atmosphere:** What is the emotional tone of the image (e.g., tense, somber, celebratory, urgent)?
4. **Potential Headline:** Suggest a possible news headline for this image.

**Important:** Provide your complete analysis as a single, comprehensive paragraph that incorporates all four elements above. Do not use bullet points or separate sections - integrate everything into one cohesive paragraph.
        """)
    
    def extract_visual_context(
        self, 
        image_path: str, 
        prompt: Optional[str] = None
    ) -> str:
        """
        Extract visual context from an image.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt (optional)
            
        Returns:
            Visual context description
        """
        if not os.path.exists(image_path):
            self.logger.warning(f"Image not found at {image_path}")
            return "Image not found"
        
        prompt_to_use = prompt or self.default_prompt
        
        try:
            description = self.model.generate_caption(image_path, prompt_to_use)
            return description
            
        except Exception as e:
            self.logger.error(f"Error extracting visual context from {image_path}: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_batch(
        self,
        image_paths: List[str],
        output_file: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        save_every: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images for visual context extraction.
        
        Args:
            image_paths: List of image file paths
            output_file: Path to save results (optional)
            checkpoint_file: Path to checkpoint file (optional)
            save_every: Save checkpoint every N items
            
        Returns:
            List of results with image paths and descriptions
        """
        results = []
        processed_count = 0
        
        # Load checkpoint if exists
        checkpoint_data = {}
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            self.logger.info(f"Loaded checkpoint with {len(checkpoint_data)} processed items")
        
        for image_path in tqdm(image_paths, desc="Extracting visual context"):
            # Skip if already processed
            if image_path in checkpoint_data:
                results.append({
                    'image_path': image_path,
                    'description': checkpoint_data[image_path]['description'],
                    'timestamp': checkpoint_data[image_path]['timestamp'],
                    'status': checkpoint_data[image_path]['status']
                })
                continue
            
            try:
                description = self.extract_visual_context(image_path)
                
                result = {
                    'image_path': image_path,
                    'description': description,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                results.append(result)
                
                # Update checkpoint
                checkpoint_data[image_path] = result
                
                self.logger.info(f"✅ Processed {image_path}: {description[:100]}...")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                result = {
                    'image_path': image_path,
                    'description': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }
                
                results.append(result)
                checkpoint_data[image_path] = result
                
                self.logger.error(f"❌ Error processing {image_path}: {str(e)}")
            
            processed_count += 1
            
            # Save checkpoint periodically
            if checkpoint_file and processed_count % save_every == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                self.logger.info(f"💾 Checkpoint saved ({processed_count} items processed)")
        
        # Final save
        if checkpoint_file:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        
        return results
    
    def process_submission_file(
        self,
        submission_path: str,
        image_dir: str,
        output_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        batch_size: int = 10,
        save_every: int = 5
    ) -> pd.DataFrame:
        """
        Process submission CSV file with visual context extraction.
        
        Args:
            submission_path: Path to submission CSV file
            image_dir: Directory containing query images
            output_path: Path for output CSV
            checkpoint_path: Path for checkpoint file
            batch_size: Number of items to process before saving
            save_every: Save checkpoint every N items
            
        Returns:
            DataFrame with visual descriptions
        """
        self.logger.info(f"Processing submission file: {submission_path}")
        
        # Read submission file
        df = pd.read_csv(submission_path)
        
        # Add description column if it doesn't exist
        if 'description' not in df.columns:
            df['description'] = ''
        
        # Load checkpoint
        processed_ids = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                processed_ids = json.load(f)
            self.logger.info(f"Found {len(processed_ids)} previously processed items")
        
        # Determine output path
        if output_path is None:
            output_path = submission_path.replace('.csv', '_with_descriptions.csv')
        
        # If output file exists, load it to preserve previous results
        if os.path.exists(output_path):
            df_existing = pd.read_csv(output_path)
            # Update df with existing descriptions
            for idx, row in df_existing.iterrows():
                if row['query_id'] in processed_ids:
                    df.loc[df['query_id'] == row['query_id'], 'description'] = row.get('description', '')
        
        # Process each query_id
        total_items = len(df)
        items_to_process = df[~df['query_id'].isin(processed_ids.keys())]
        
        self.logger.info(f"Processing {len(items_to_process)} remaining query images...")
        
        processed_count = 0
        
        for idx, row in tqdm(items_to_process.iterrows(), total=len(items_to_process), desc="Generating descriptions"):
            query_id = row['query_id']
            query_image_path = os.path.join(image_dir, f"{query_id}.jpg")
            
            try:
                # Generate visual description
                description = self.extract_visual_context(query_image_path)
                
                # Update dataframe
                df.loc[df['query_id'] == query_id, 'description'] = description
                
                # Update checkpoint
                processed_ids[query_id] = {
                    'description': description,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                self.logger.info(f"✅ {query_id}: {description[:100]}...")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.logger.error(f"❌ Error processing {query_id}: {str(e)}")
                
                # Update with error
                df.loc[df['query_id'] == query_id, 'description'] = error_msg
                
                # Update checkpoint with error
                processed_ids[query_id] = {
                    'description': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }
            
            processed_count += 1
            
            # Save checkpoint periodically
            if checkpoint_path and processed_count % save_every == 0:
                with open(checkpoint_path, 'w') as f:
                    json.dump(processed_ids, f, indent=2)
                self.logger.info(f"💾 Checkpoint saved ({len(processed_ids)} items processed)")
            
            # Save results periodically
            if processed_count % batch_size == 0:
                df.to_csv(output_path, index=False)
                self.logger.info(f"💾 Results saved to: {output_path}")
        
        # Final save
        if checkpoint_path:
            with open(checkpoint_path, 'w') as f:
                json.dump(processed_ids, f, indent=2)
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"✅ Processing complete! Results saved to: {output_path}")
        self.logger.info(f"✅ Total items processed: {len(processed_ids)}")
        
        # Clean up checkpoint if all items processed successfully
        if len(processed_ids) == total_items and checkpoint_path:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                self.logger.info("✅ Checkpoint file removed (all items processed)")
        
        return df


def main():
    """Main function for standalone execution."""
    import yaml
    
    # Load configuration
    config_path = "config/model_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Initialize extractor
    extractor = VisualExtractor(config)
    
    # Process submission file
    submission_path = "data/submission/submission.csv"
    private_img_path = "data/processed/private/query"
    
    if os.path.exists(submission_path):
        results_df = extractor.process_submission_file(
            submission_path=submission_path,
            image_dir=private_img_path,
            output_path="data/submission/submission_with_descriptions.csv",
            checkpoint_path="data/submission/checkpoint.json",
            batch_size=1,
            save_every=1
        )
        
        # Display summary
        print("\n--- Processing Summary ---")
        print(f"Total queries: {len(results_df)}")
        print(f"Successfully processed: {len(results_df[~results_df['description'].str.startswith('Error:')])}")
        print(f"Errors: {len(results_df[results_df['description'].str.startswith('Error:')])}")
        
        # Display sample results
        print("\n--- Sample Results ---")
        print(results_df[['query_id', 'description']].head())
    else:
        print(f"Submission file not found: {submission_path}")


if __name__ == "__main__":
    main()