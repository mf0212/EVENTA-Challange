import torch
import pandas as pd
import os
import json
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def load_model():
    """Load the Qwen-VL model and processor"""
    print("\nLoading MiMo model for caption generation...")
    QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Load model
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load processor
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    qwen_processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_ID, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    
    print("✅ Qwen-VL model loaded.")
    return qwen_model, qwen_processor

def generate_caption_qwen(image_path, model, processor):
    """
    Generates a context-aware caption using the Qwen-VL model.
    
    Args:
        image_path: Path to the image file
        model: The pre-trained Qwen-VL model
        processor: The Qwen-VL processor
    
    Returns:
        str: The generated caption
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return "Image not found"
    
    # Prompt instruction
    prompt_instruction = """
    You are a visual analyst. The following image is taken from a CNN news article. Please provide a detailed and comprehensive description.

    Your description should cover:
    1.  **Objective Description:** What do you see in the image? Describe the people, objects, setting, and any text visible.
    2.  **Contextual Inference:** Based on the visual cues and the fact that this is from CNN, what could the news story be about? What is the likely location or event?
    3.  **Overall Mood and Atmosphere:** What is the emotional tone of the image (e.g., tense, somber, celebratory, urgent)?
    4.  **Potential Headline:** Suggest a possible news headline for this image.

    **Important:** Provide your complete analysis as a single, comprehensive paragraph that incorporates all four elements above. Do not use bullet points or separate sections - integrate everything into one cohesive paragraph.
    """
    
    # Qwen-VL messages format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_instruction},
            ],
        }
    ]
    
    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=None, 
        padding=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=2048, 
            temperature=0.7, 
            do_sample=True
        )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()

def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_path, processed_ids):
    """Save checkpoint"""
    with open(checkpoint_path, 'w') as f:
        json.dump(processed_ids, f, indent=2)

def process_submission_file_with_checkpoint(
    submission_path, 
    private_img_path, 
    model, 
    processor, 
    output_path=None,
    checkpoint_path="checkpoint.json",
    batch_size=10,
    save_every=5
):
    """
    Process submission CSV file with checkpointing and batch saving
    
    Args:
        submission_path: Path to submission CSV file
        private_img_path: Path to directory containing query images
        model: The pre-trained Qwen-VL model
        processor: The Qwen-VL processor
        output_path: Path for output CSV
        checkpoint_path: Path for checkpoint file
        batch_size: Number of items to process before saving
        save_every: Save checkpoint every N items
    """
    # Read submission file
    print(f"\nReading submission file from: {submission_path}")
    df = pd.read_csv(submission_path)
    
    # Add description column if it doesn't exist
    if 'description' not in df.columns:
        df['description'] = ''
    
    # Load checkpoint
    processed_ids = load_checkpoint(checkpoint_path)
    print(f"Found {len(processed_ids)} previously processed items")
    
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
    
    print(f"\nProcessing {len(items_to_process)} remaining query images...")
    
    processed_count = 0
    
    for idx, row in tqdm(items_to_process.iterrows(), total=len(items_to_process), desc="Generating descriptions"):
        query_id = row['query_id']
        query_image_path = os.path.join(private_img_path, f"{query_id}.jpg")
        
        try:
            # Generate caption
            caption = generate_caption_qwen(query_image_path, model, processor)
            
            # Update dataframe
            df.loc[df['query_id'] == query_id, 'description'] = caption
            
            # Update checkpoint
            processed_ids[query_id] = {
                'caption': caption,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            print(f"\n✅ {query_id}: {caption[:100]}...")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n❌ Error processing {query_id}: {str(e)}")
            
            # Update with error
            df.loc[df['query_id'] == query_id, 'description'] = error_msg
            
            # Update checkpoint with error
            processed_ids[query_id] = {
                'caption': error_msg,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
        
        processed_count += 1
        
        # Save checkpoint periodically
        if processed_count % save_every == 0:
            save_checkpoint(checkpoint_path, processed_ids)
            print(f"\n💾 Checkpoint saved ({len(processed_ids)} items processed)")
        
        # Save results periodically
        if processed_count % batch_size == 0:
            df.to_csv(output_path, index=False)
            print(f"\n💾 Results saved to: {output_path}")
    
    # Final save
    save_checkpoint(checkpoint_path, processed_ids)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Processing complete! Results saved to: {output_path}")
    print(f"✅ Total items processed: {len(processed_ids)}")
    
    # Clean up checkpoint if all items processed successfully
    if len(processed_ids) == total_items:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("✅ Checkpoint file removed (all items processed)")
    
    return df

def main():
    """Main function to run the script"""
    # Define paths
    base_dir = 'data/processed'
    private_path = os.path.join(base_dir, 'private')
    private_img_path = os.path.join(private_path, 'query')
    submission_path = "data/submission/submission.csv"
    
    # Load model
    model, processor = load_model()
    
    # Process submission file with checkpointing
    results_df = process_submission_file_with_checkpoint(
        submission_path=submission_path,
        private_img_path=private_img_path,
        model=model,
        processor=processor,
        output_path="data/submission/submission_with_descriptions.csv",
        checkpoint_path="data/submission/checkpoint.json",
        batch_size=1,  # Save results every 10 items
        save_every=1    # Save checkpoint every 5 items
    )
    
    # Display summary
    print("\n--- Processing Summary ---")
    print(f"Total queries: {len(results_df)}")
    print(f"Successfully processed: {len(results_df[~results_df['description'].str.startswith('Error:')])}")
    print(f"Errors: {len(results_df[results_df['description'].str.startswith('Error:')])}")
    
    # Display sample results
    print("\n--- Sample Results ---")
    print(results_df[['query_id', 'description']].head())

if __name__ == "__main__":
    main()