"""
Caption generation for Event-Enriched Image Captioning
=====================================================

This module provides utilities for generating context-aware captions
by combining visual descriptions with retrieved contextual information.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any
from tqdm import tqdm

from ..models.qwen_models import QwenTextModel
from ..utils.logging_utils import get_logger, LoggerMixin

logger = get_logger(__name__)


class CaptionGenerator(LoggerMixin):
    """
    Context-aware caption generator using Qwen text model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize caption generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model = QwenTextModel(config)
        self.model.load_model()
        
        # Default prompt for caption generation
        self.default_prompt = self.config.get('prompts', {}).get('caption_generation', """
# GOAL
Your primary objective is to generate a single, compelling paragraph that serves as a caption for an image. This caption must skillfully synthesize the provided `[IMAGE DESCRIPTION]` with the context from the `[RETRIEVED CONTEXT]`. The final caption should be of a quality that would achieve a high CIDEr score when compared against human-generated captions.

# INPUTS
1. `[IMAGE DESCRIPTION]`: A description of the visual elements in the photograph.
2. `[RETRIEVED CONTEXT]`: Pre-extracted relevant text content related to the image from news articles.

# INSTRUCTIONS
Follow this methodology precisely:

1. **Analyze and Synthesize, Do Not Summarize:** Your task is not to summarize the image or the context independently. You must **weave them together**. The image is your anchor; the retrieved context is your source of truth and contextual information.

2. **Start with the Visual Anchor:** Begin the caption by describing the core scene or action from the `[IMAGE DESCRIPTION]`. Mention the key subjects, the setting, and the overall mood conveyed by the visual.

3. **Enrich with Context from Retrieved Text:** Use the `[RETRIEVED CONTEXT]` to immediately identify the **who, what, where, when, and why** of the image.

4. **Connect the Visuals to the Narrative:** This is the most crucial step. Explicitly link what is seen in the image to the story from the retrieved context.

5. **Explain the Significance and Symbolism:** Conclude by explaining the broader importance of the moment captured based on the retrieved context.

# CONSTRAINTS
- The output must be a **single, well-structured paragraph**. Output ONLY the caption paragraph.
- Do NOT add any prefixes or suffixes
- Just write the caption directly
- Do **NOT** invent any information that is not present in the inputs.
- Maintain a professional, engaging, and journalistic tone.
- Focus on a smooth narrative flow, not a bulleted list of facts.
        """)
        
        self.system_message = "You are an expert photo caption writer for a major international news organization like Reuters, Associated Press, or CNN. Your captions are rich, informative, and provide deep context."
    
    def generate_caption(
        self,
        image_description: str,
        retrieved_context: str,
        prompt: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate a context-aware caption.
        
        Args:
            image_description: Description of the visual elements
            retrieved_context: Pre-extracted relevant context
            prompt: Custom prompt template (optional)
            system_message: Custom system message (optional)
            
        Returns:
            Generated caption
        """
        prompt_template = prompt or self.default_prompt
        system_msg = system_message or self.system_message
        
        # Format the prompt with inputs
        formatted_prompt = f"""
{prompt_template}

---
### **INPUT TEMPLATE**

**[IMAGE DESCRIPTION]**
{image_description}

**[RETRIEVED CONTEXT]**
{retrieved_context}

---"""
        
        try:
            caption = self.model.generate_text(
                prompt=formatted_prompt,
                system_message=system_msg
            )
            return caption
            
        except Exception as e:
            self.logger.error(f"Error generating caption: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_submission(
        self,
        submission_csv_path: str,
        output_path: Optional[str] = None,
        batch_size: int = 1
    ) -> pd.DataFrame:
        """
        Process all queries in submission.csv and generate captions.
        
        Args:
            submission_csv_path: Path to submission CSV with retrieved_text column
            output_path: Path to save the updated submission file (if None, overwrites input)
            batch_size: Number of queries to process before saving (for checkpointing)
            
        Returns:
            DataFrame with generated captions
        """
        # Load submission file
        self.logger.info(f"Loading submission file: {submission_csv_path}")
        df = pd.read_csv(submission_csv_path)
        self.logger.info(f"Found {len(df)} queries to process.")
        
        # Verify required columns exist
        required_columns = ['query_id', 'generated_caption', 'retrieved_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check if description column exists
        if 'description' not in df.columns:
            self.logger.warning("'description' column not found. Will use query_id as fallback.")
        
        # If output_path is not specified, use the input path
        if output_path is None:
            output_path = submission_csv_path
        
        # Create a checkpoint file path
        checkpoint_path = output_path.replace('.csv', '_checkpoint.csv')
        
        # Check if there's a checkpoint file to resume from
        if os.path.exists(checkpoint_path):
            self.logger.info("Found checkpoint file. Resuming from previous run...")
            df_checkpoint = pd.read_csv(checkpoint_path)
            # Find where we left off
            processed_mask = df_checkpoint['generated_caption'] != 'Caption to be generated'
            last_processed_idx = processed_mask.sum() - 1
            if last_processed_idx >= 0:
                df = df_checkpoint
                self.logger.info(f"Resuming from index {last_processed_idx + 1}")
        
        # Process each query
        total_queries = len(df)
        for idx, row in df.iterrows():
            # Skip if already processed
            if row['generated_caption'] != 'Caption to be generated':
                continue
                
            query_id = row['query_id']
            
            # Print progress
            self.logger.info(f"Processing query {idx + 1}/{total_queries} (ID: {query_id})")
            
            try:
                # Get image description from the CSV description column (if available)
                if 'description' in df.columns and pd.notna(row['description']) and row['description'].strip() != '':
                    image_description = row['description']
                else:
                    self.logger.warning(f"No description found for query {query_id}, using fallback")
                    image_description = f"An image from query {query_id}"
                
                # Get pre-extracted context from retrieved_text column
                retrieved_context = row['retrieved_text']
                
                if pd.isna(retrieved_context) or retrieved_context.strip() == '':
                    self.logger.warning(f"No retrieved context found for query {query_id}")
                    retrieved_context = "No context available."
                
                # Generate caption using Qwen text model with image description and retrieved context
                caption = self.generate_caption(
                    image_description,
                    retrieved_context
                )
                
                # Update the dataframe
                df.at[idx, 'generated_caption'] = caption
                
                self.logger.info(f"✅ Generated caption for query {query_id}")
                
                # Save checkpoint every batch_size queries
                if (idx + 1) % batch_size == 0:
                    df.to_csv(checkpoint_path, index=False)
                    self.logger.info(f"💾 Checkpoint saved at query {idx + 1}")

            except Exception as e:
                self.logger.error(f"❌ Error processing query {query_id}: {str(e)}")
                continue
        
        # Save final results
        df.to_csv(output_path, index=False)
        self.logger.info(f"✅ Final results saved to: {output_path}")

        # Remove checkpoint file if everything completed successfully
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            self.logger.info("🧹 Checkpoint file removed.")

        return df
    
    def generate_single_caption(
        self,
        query_id: str,
        image_description: str,
        retrieved_context: str
    ) -> Dict[str, Any]:
        """
        Generate a single caption with metadata.
        
        Args:
            query_id: Query identifier
            image_description: Visual description of the image
            retrieved_context: Retrieved contextual information
            
        Returns:
            Dictionary with caption and metadata
        """
        try:
            caption = self.generate_caption(image_description, retrieved_context)
            
            return {
                'query_id': query_id,
                'generated_caption': caption,
                'status': 'success',
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error generating caption for {query_id}: {str(e)}")
            return {
                'query_id': query_id,
                'generated_caption': f"Error: {str(e)}",
                'status': 'error',
                'error': str(e)
            }
    
    def batch_generate_captions(
        self,
        data: list,
        output_file: Optional[str] = None
    ) -> list:
        """
        Generate captions for a batch of data.
        
        Args:
            data: List of dictionaries with query_id, image_description, retrieved_context
            output_file: Path to save results (optional)
            
        Returns:
            List of results with generated captions
        """
        results = []
        
        for item in tqdm(data, desc="Generating captions"):
            result = self.generate_single_caption(
                item['query_id'],
                item['image_description'],
                item['retrieved_context']
            )
            results.append(result)
        
        if output_file:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False)
            self.logger.info(f"Results saved to {output_file}")
        
        return results


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
    
    # Initialize caption generator
    caption_generator = CaptionGenerator(config)
    
    # Process submission file
    submission_path = "data/submission/submission_with_descriptions_retrieval.csv"
    
    if os.path.exists(submission_path):
        results_df = caption_generator.process_submission(
            submission_csv_path=submission_path,
            output_path="outputs/submissions/submission_final.csv",
            batch_size=1
        )
        
        print(f"🎉 Caption generation completed successfully!")
        print(f"📊 Processed {len(results_df)} queries")
    else:
        print(f"Submission file not found: {submission_path}")


if __name__ == "__main__":
    main()