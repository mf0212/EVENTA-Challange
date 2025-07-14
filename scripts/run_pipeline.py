#!/usr/bin/env python3
"""
Main pipeline script for Event-Enriched Image Captioning
=======================================================

This script orchestrates the complete pipeline:
1. Data download and preparation
2. Visual context extraction
3. Retrieval of relevant articles
4. Context-aware caption generation
5. Evaluation and results

Usage:
    python scripts/run_pipeline.py --config config/model_config.yaml
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.downloader import DataDownloader
from src.captioning.visual_extractor import VisualExtractor
from src.retrieval.retriever import Retriever
from src.captioning.caption_generator import CaptionGenerator
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.evaluation import evaluate_captions

logger = get_logger(__name__)


class CaptioningPipeline:
    """
    Main pipeline for event-enriched image captioning.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        
        # Setup logging
        setup_logging(config_path=config_path)
        
        # Initialize components
        self.data_downloader = None
        self.visual_extractor = None
        self.retriever = None
        self.caption_generator = None
        
        logger.info("Pipeline initialized")
    
    def load_config(self) -> dict:
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_data(self, force_download: bool = False) -> bool:
        """
        Setup and download data if needed.
        
        Args:
            force_download: Force re-download even if data exists
            
        Returns:
            True if successful
        """
        logger.info("Setting up data...")
        
        # Check if data already exists
        data_exists = (
            os.path.exists("data/processed/database/database.json") and
            os.path.exists("data/processed/private/query.csv")
        )
        
        if data_exists and not force_download:
            logger.info("Data already exists, skipping download")
            return True
        
        # Initialize data downloader
        self.data_downloader = DataDownloader("config/data_config.yaml")
        
        # Download all data
        success = self.data_downloader.download_all()
        
        if success:
            logger.info("✅ Data setup completed successfully")
        else:
            logger.error("❌ Data setup failed")
        
        return success
    
    def extract_visual_context(
        self,
        submission_path: str,
        force_reprocess: bool = False
    ) -> str:
        """
        Extract visual context from images.
        
        Args:
            submission_path: Path to submission CSV
            force_reprocess: Force re-processing even if results exist
            
        Returns:
            Path to file with visual descriptions
        """
        logger.info("Extracting visual context...")
        
        output_path = submission_path.replace('.csv', '_with_descriptions.csv')
        
        # Check if already processed
        if os.path.exists(output_path) and not force_reprocess:
            logger.info("Visual descriptions already exist, skipping extraction")
            return output_path
        
        # Initialize visual extractor
        self.visual_extractor = VisualExtractor(self.config)
        
        # Process submission file
        self.visual_extractor.process_submission_file(
            submission_path=submission_path,
            image_dir="data/processed/private/query",
            output_path=output_path,
            checkpoint_path="outputs/visual_extraction_checkpoint.json",
            batch_size=1,
            save_every=1
        )
        
        logger.info(f"✅ Visual context extraction completed: {output_path}")
        return output_path
    
    def retrieve_context(
        self,
        submission_path: str,
        force_reprocess: bool = False
    ) -> str:
        """
        Retrieve relevant context from article database.
        
        Args:
            submission_path: Path to submission CSV with visual descriptions
            force_reprocess: Force re-processing even if results exist
            
        Returns:
            Path to file with retrieved context
        """
        logger.info("Retrieving contextual information...")
        
        output_path = submission_path.replace('.csv', '_with_retrieval.csv')
        
        # Check if already processed
        if os.path.exists(output_path) and not force_reprocess:
            logger.info("Retrieved context already exists, skipping retrieval")
            return output_path
        
        # Initialize retriever
        self.retriever = Retriever(self.config)
        
        # TODO: Implement actual retrieval process
        # For now, create a placeholder file
        import pandas as pd
        df = pd.read_csv(submission_path)
        
        # Add placeholder retrieved_text column
        if 'retrieved_text' not in df.columns:
            df['retrieved_text'] = "Placeholder retrieved context. Replace with actual retrieval pipeline."
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Context retrieval completed: {output_path}")
        logger.warning("⚠️  Using placeholder retrieval. Replace with your actual retrieval pipeline.")
        
        return output_path
    
    def generate_captions(
        self,
        submission_path: str,
        force_reprocess: bool = False
    ) -> str:
        """
        Generate context-aware captions.
        
        Args:
            submission_path: Path to submission CSV with descriptions and retrieved context
            force_reprocess: Force re-processing even if results exist
            
        Returns:
            Path to final submission file
        """
        logger.info("Generating context-aware captions...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/submissions/submission_final_{timestamp}.csv"
        
        # Initialize caption generator
        self.caption_generator = CaptionGenerator(self.config)
        
        # Process submission file
        self.caption_generator.process_submission(
            submission_csv_path=submission_path,
            output_path=output_path,
            batch_size=1
        )
        
        logger.info(f"✅ Caption generation completed: {output_path}")
        return output_path
    
    def evaluate_results(
        self,
        predictions_file: str,
        ground_truth_file: str = None
    ) -> dict:
        """
        Evaluate generated captions.
        
        Args:
            predictions_file: Path to predictions file
            ground_truth_file: Path to ground truth file (optional)
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/evaluation_results_{timestamp}.json"
        
        results = evaluate_captions(
            predictions_file=predictions_file,
            ground_truth_file=ground_truth_file,
            output_file=output_file
        )
        
        logger.info("✅ Evaluation completed")
        return results
    
    def run_full_pipeline(
        self,
        submission_path: str = "data/submission/submission.csv",
        force_reprocess: bool = False,
        skip_data_setup: bool = False
    ) -> str:
        """
        Run the complete pipeline.
        
        Args:
            submission_path: Path to initial submission CSV
            force_reprocess: Force re-processing of all steps
            skip_data_setup: Skip data download step
            
        Returns:
            Path to final submission file
        """
        logger.info("🚀 Starting full pipeline...")
        
        try:
            # Step 1: Setup data
            if not skip_data_setup:
                if not self.setup_data(force_download=force_reprocess):
                    raise RuntimeError("Data setup failed")
            
            # Create initial submission file if it doesn't exist
            if not os.path.exists(submission_path):
                self.create_initial_submission(submission_path)
            
            # Step 2: Extract visual context
            submission_with_descriptions = self.extract_visual_context(
                submission_path, force_reprocess
            )
            
            # Step 3: Retrieve contextual information
            submission_with_retrieval = self.retrieve_context(
                submission_with_descriptions, force_reprocess
            )
            
            # Step 4: Generate captions
            final_submission = self.generate_captions(
                submission_with_retrieval, force_reprocess
            )
            
            # Step 5: Evaluate results (if ground truth available)
            ground_truth_path = "data/processed/train/gt_train.csv"
            if os.path.exists(ground_truth_path):
                self.evaluate_results(final_submission, ground_truth_path)
            
            logger.info(f"🎉 Pipeline completed successfully!")
            logger.info(f"📄 Final submission: {final_submission}")
            
            return final_submission
            
        except Exception as e:
            logger.error(f"💥 Pipeline failed: {str(e)}")
            raise
    
    def create_initial_submission(self, submission_path: str) -> None:
        """
        Create initial submission file from private query data.
        
        Args:
            submission_path: Path where to create submission file
        """
        logger.info("Creating initial submission file...")
        
        import pandas as pd
        
        # Read private query data
        private_query_path = "data/processed/private/query.csv"
        if os.path.exists(private_query_path):
            df_queries = pd.read_csv(private_query_path)
            
            # Create submission format
            df_submission = pd.DataFrame({
                'query_id': df_queries['query_id'],
                'generated_caption': 'Caption to be generated'
            })
            
            # Create directory if needed
            os.makedirs(os.path.dirname(submission_path), exist_ok=True)
            
            # Save submission file
            df_submission.to_csv(submission_path, index=False)
            logger.info(f"✅ Initial submission file created: {submission_path}")
        else:
            raise FileNotFoundError(f"Private query file not found: {private_query_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Event-Enriched Image Captioning Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--submission", 
        type=str, 
        default="data/submission/submission.csv",
        help="Path to submission CSV file"
    )
    parser.add_argument(
        "--force-reprocess", 
        action="store_true",
        help="Force re-processing of all steps"
    )
    parser.add_argument(
        "--skip-data-setup", 
        action="store_true",
        help="Skip data download step"
    )
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["data", "visual", "retrieval", "caption", "evaluate", "full"],
        default="full",
        help="Run specific pipeline step"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CaptioningPipeline(args.config)
    
    try:
        if args.step == "full":
            # Run complete pipeline
            final_submission = pipeline.run_full_pipeline(
                submission_path=args.submission,
                force_reprocess=args.force_reprocess,
                skip_data_setup=args.skip_data_setup
            )
            print(f"\n🎉 Pipeline completed!")
            print(f"📄 Final submission: {final_submission}")
            
        elif args.step == "data":
            # Setup data only
            pipeline.setup_data(force_download=args.force_reprocess)
            
        elif args.step == "visual":
            # Visual extraction only
            result = pipeline.extract_visual_context(
                args.submission, args.force_reprocess
            )
            print(f"Visual extraction completed: {result}")
            
        elif args.step == "retrieval":
            # Retrieval only
            result = pipeline.retrieve_context(
                args.submission, args.force_reprocess
            )
            print(f"Context retrieval completed: {result}")
            
        elif args.step == "caption":
            # Caption generation only
            result = pipeline.generate_captions(
                args.submission, args.force_reprocess
            )
            print(f"Caption generation completed: {result}")
            
        elif args.step == "evaluate":
            # Evaluation only
            pipeline.evaluate_results(args.submission)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()