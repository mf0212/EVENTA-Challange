#!/usr/bin/env python3
"""
Example usage of Event-Enriched Image Captioning system
======================================================

This script demonstrates how to use the various components
of the captioning system.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.downloader import DataDownloader
from src.captioning.visual_extractor import VisualExtractor
from src.retrieval.retriever import Retriever
from src.captioning.caption_generator import CaptionGenerator
from src.models.qwen_models import QwenVLModel, QwenTextModel
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.evaluation import evaluate_captions

# Setup logging
setup_logging(log_level="INFO", console=True)
logger = get_logger(__name__)


def example_data_download():
    """Example: Download and setup data."""
    print("=== Example: Data Download ===")
    
    # Initialize data downloader
    downloader = DataDownloader("config/data_config.yaml")
    
    # Get database statistics
    stats = downloader.get_database_stats() if hasattr(downloader, 'get_database_stats') else {}
    print(f"Database stats: {stats}")
    
    # Download all data (uncomment to actually download)
    # success = downloader.download_all()
    # print(f"Download successful: {success}")
    
    print("✅ Data download example completed\n")


def example_visual_extraction():
    """Example: Extract visual context from images."""
    print("=== Example: Visual Context Extraction ===")
    
    # Load configuration
    import yaml
    config = {}
    if os.path.exists("config/model_config.yaml"):
        with open("config/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    
    # Initialize visual extractor
    extractor = VisualExtractor(config)
    
    # Example: Extract context from a single image
    image_path = "data/processed/private/query/example.jpg"
    if os.path.exists(image_path):
        description = extractor.extract_visual_context(image_path)
        print(f"Image: {image_path}")
        print(f"Description: {description[:200]}...")
    else:
        print(f"Example image not found: {image_path}")
        print("This would extract visual context from the image using Qwen-VL")
    
    print("✅ Visual extraction example completed\n")


def example_retrieval():
    """Example: Retrieve relevant context."""
    print("=== Example: Context Retrieval ===")
    
    # Initialize retriever
    retriever = Retriever()
    
    # Get database statistics
    stats = retriever.get_database_stats()
    print(f"Database statistics: {stats}")
    
    # Example: Retrieve context for an image
    image_path = "data/processed/private/query/example.jpg"
    context = retriever.retrieve_context(image_path)
    print(f"Retrieved context: {context[:200]}...")
    
    # Example: Search articles
    articles = retriever.search_articles("climate change", top_k=3)
    print(f"Found {len(articles)} articles")
    
    print("✅ Retrieval example completed\n")


def example_caption_generation():
    """Example: Generate context-aware captions."""
    print("=== Example: Caption Generation ===")
    
    # Load configuration
    import yaml
    config = {}
    if os.path.exists("config/model_config.yaml"):
        with open("config/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    
    # Initialize caption generator
    caption_generator = CaptionGenerator(config)
    
    # Example inputs
    image_description = """
    This striking photograph captures a moment of intense action during a climate protest. 
    The image shows a diverse group of young activists holding colorful signs with messages 
    like "ACT NOW" and "SAVE OUR PLANET" while marching through what appears to be a city street. 
    The atmosphere is energetic and determined, with participants displaying passionate expressions 
    as they advocate for urgent climate action.
    """
    
    retrieved_context = """
    Thousands of climate activists gathered in Glasgow during the COP26 summit to demand 
    immediate action on climate change. The protest, organized by youth climate groups, 
    drew participants from around the world who marched through the city center calling 
    for world leaders to commit to more ambitious carbon reduction targets. The demonstration 
    was part of a global day of action coinciding with the UN climate conference.
    """
    
    # Generate caption
    caption = caption_generator.generate_caption(
        image_description=image_description,
        retrieved_context=retrieved_context
    )
    
    print("Generated Caption:")
    print(f"{caption}")
    
    print("✅ Caption generation example completed\n")


def example_model_usage():
    """Example: Direct model usage."""
    print("=== Example: Direct Model Usage ===")
    
    # Load configuration
    import yaml
    config = {}
    if os.path.exists("config/model_config.yaml"):
        with open("config/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    
    # Example: Qwen-VL model
    print("Qwen-VL Model Example:")
    try:
        qwen_vl = QwenVLModel(config)
        # qwen_vl.load_model()  # Uncomment to actually load
        print("✅ Qwen-VL model initialized (not loaded)")
    except Exception as e:
        print(f"❌ Qwen-VL model error: {str(e)}")
    
    # Example: Qwen text model
    print("Qwen Text Model Example:")
    try:
        qwen_text = QwenTextModel(config)
        # qwen_text.load_model()  # Uncomment to actually load
        print("✅ Qwen text model initialized (not loaded)")
    except Exception as e:
        print(f"❌ Qwen text model error: {str(e)}")
    
    print("✅ Model usage example completed\n")


def example_evaluation():
    """Example: Evaluate generated captions."""
    print("=== Example: Caption Evaluation ===")
    
    # Example predictions and references
    predictions = [
        "A group of climate activists march through the streets holding protest signs during COP26.",
        "Students gather in a university courtyard for a climate change awareness event.",
        "Environmental protesters demonstrate outside government buildings demanding policy changes."
    ]
    
    references = [
        ["Climate activists march with signs during COP26 summit protests in Glasgow.",
         "Protesters demand climate action during international climate conference."],
        ["University students organize climate awareness rally on campus.",
         "Young people gather to discuss environmental issues at educational event."],
        ["Environmental groups protest outside parliament for climate legislation.",
         "Activists demonstrate for stronger environmental policies and regulations."]
    ]
    
    # Calculate metrics (would need actual evaluation libraries)
    print("Example evaluation metrics:")
    print("CIDEr Score: 0.85 (example)")
    print("BLEU-4 Score: 0.42 (example)")
    print("ROUGE-L Score: 0.58 (example)")
    
    print("✅ Evaluation example completed\n")


def example_full_workflow():
    """Example: Complete workflow."""
    print("=== Example: Complete Workflow ===")
    
    # This demonstrates the complete pipeline workflow
    workflow_steps = [
        "1. Download and setup data",
        "2. Extract visual context from query images", 
        "3. Retrieve relevant articles from database",
        "4. Generate context-aware captions",
        "5. Evaluate results against ground truth",
        "6. Save final submission file"
    ]
    
    print("Complete workflow steps:")
    for step in workflow_steps:
        print(f"  {step}")
    
    print("\nTo run the complete workflow:")
    print("  python scripts/run_pipeline.py --config config/model_config.yaml")
    
    print("✅ Workflow example completed\n")


def main():
    """Run all examples."""
    print("🚀 Event-Enriched Image Captioning - Usage Examples\n")
    
    examples = [
        example_data_download,
        example_visual_extraction,
        example_retrieval,
        example_caption_generation,
        example_model_usage,
        example_evaluation,
        example_full_workflow
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            logger.error(f"Example failed: {example_func.__name__}: {str(e)}")
    
    print("🎉 All examples completed!")
    print("\nNext steps:")
    print("1. Configure your models in config/model_config.yaml")
    print("2. Download data: bash scripts/download_data.sh")
    print("3. Run pipeline: python scripts/run_pipeline.py")
    print("4. Evaluate results: python scripts/evaluate.py --predictions <file>")


if __name__ == "__main__":
    main()