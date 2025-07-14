#!/usr/bin/env python3
"""
Refined Caption Generation Script using Qwen-VL
===============================================
This script processes image queries and generates captions using Qwen-VL
with pre-extracted context from the retrieved_text column.

Requirements:
- torch
- transformers
- Pillow
- pandas

Usage:
    python refined_caption_generation_script.py
"""

import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_paths():
    """Setup all data paths"""
    # Base directory where the download script saves processed data
    base_dir = 'data/processed'

    # Query paths (public query set)
    query_path = os.path.join(base_dir, 'query')
    query_img_path = os.path.join(query_path, 'pub_images')

    # Private set paths
    private_path = os.path.join(base_dir, 'private')
    private_img_path = os.path.join(private_path, 'query')

    # Submission path - using your file with retrieved_text column
    submission_path = "/home/sv-lkhai/mf/data/submission/submission_with_descriptions_retrieval_sent013.csv"
    
    paths = {
        'base_dir': base_dir,
        'query_img_path': query_img_path,
        'private_img_path': private_img_path,
        'submission_path': submission_path
    }
    
    return paths

def load_qwen_model():
    """Load Qwen 2.5 7B model"""
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Load Qwen 2.5 7B Model
    print("\nLoading Qwen 2.5 7B model for caption generation...")
    QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    print("✅ Qwen 2.5 7B model loaded.")
    
    return qwen_model, qwen_tokenizer, DEVICE

def generate_caption_qwen(image_description, retrieved_context, model, tokenizer):
    """
    Generates a context-aware caption using the Qwen 2.5 7B model based on image description
    and pre-retrieved context.

    Args:
        image_description (str): Text description of the image
        retrieved_context (str): Pre-extracted relevant context from articles
        model: The pre-trained Qwen 2.5 7B model.
        tokenizer: The Qwen 2.5 7B tokenizer.

    Returns:
        str: The generated caption.
    """
    
    # Modified prompt to work with image description and pre-extracted context
    prompt_instruction = f"""
# GOAL
Your primary objective is to generate a single, compelling paragraph that serves as a caption for an image. This caption must skillfully synthesize the provided `[IMAGE DESCRIPTION]` with the context from the `[RETRIEVED CONTEXT]`. The final caption should be of a quality that would achieve a high CIDEr score when compared against human-generated captions.

# INPUTS
1.  `[IMAGE DESCRIPTION]`: A description of the visual elements in the photograph.
2.  `[RETRIEVED CONTEXT]`: Pre-extracted relevant text content related to the image from news articles.

# INSTRUCTIONS
Follow this methodology precisely:

1.  **Analyze and Synthesize, Do Not Summarize:** Your task is not to summarize the image or the context independently. You must **weave them together**. The image is your anchor; the retrieved context is your source of truth and contextual information.

2.  **Start with the Visual Anchor:** Begin the caption by describing the core scene or action from the `[IMAGE DESCRIPTION]`. Mention the key subjects, the setting, and the overall mood conveyed by the visual.
    *   *Example:* "This striking photograph captures a moment of intense action..." or "In this poignant image, two leaders stand side-by-side..."

3.  **Enrich with Context from Retrieved Text:** Use the `[RETRIEVED CONTEXT]` to immediately identify the **who, what, where, when, and why** of the image.
    *   Name the specific people, places, and the event mentioned in the context.
    *   Explain the circumstances leading to this moment based on the retrieved information.
    *   *Example:* "...featuring tennis champion Serena Williams during a pivotal moment in her career..."

4.  **Connect the Visuals to the Narrative:** This is the most crucial step. Explicitly link what is seen in the image to the story from the retrieved context. Explain the meaning behind a facial expression, a posture, or a specific object in the photo.
    *   *Example:* "Her determined expression **reflects the resilience** mentioned in the article about her journey to becoming one of tennis's greatest players."
    *   *Example:* "The setting **underscores the significance** of this moment in the ongoing story described in the context."

5.  **Explain the Significance and Symbolism:** Conclude by explaining the broader importance of the moment captured based on the retrieved context. What does this image symbolize? What was the impact or outcome of this event?
    *   *Example:* "This moment **represents a turning point** in the narrative described in the context."
    *   *Example:* "The image serves as a **powerful visual testament** to the themes explored in the related coverage."

# CONSTRAINTS
-   The output must be a **single, well-structured paragraph**. Output ONLY the caption paragraph.
-   Do NOT add any prefixes or suffixes
-   Just write the caption directly
-   Do **NOT** invent any information that is not present in the `[IMAGE DESCRIPTION]` or `[RETRIEVED CONTEXT]`.
-   Maintain a professional, engaging, and journalistic tone.
-   Focus on a smooth narrative flow, not a bulleted list of facts.

---
### **INPUT TEMPLATE**

**[IMAGE DESCRIPTION]**
{image_description}

**[RETRIEVED CONTEXT]**
{retrieved_context}

---"""

    # Create messages in the format expected by Qwen 2.5
    messages = [
        {"role": "system", "content": "You are an expert photo caption writer for a major international news organization like Reuters, Associated Press, or CNN. Your captions are rich, informative, and provide deep context."},
        {"role": "user", "content": prompt_instruction}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the caption
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the generated part (remove the input prompt)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response.strip()

def process_submission(submission_csv_path, query_img_path, qwen_model, qwen_processor,
                      output_path=None, batch_size=1):
    """
    Process all queries in submission.csv and generate captions using pre-extracted context.
    
    Args:
        submission_csv_path: Path to submission CSV with retrieved_text column
        query_img_path: Path to query images directory
        qwen_model: Loaded Qwen-VL model
        qwen_processor: Qwen-VL processor
        output_path: Path to save the updated submission file (if None, overwrites input)
        batch_size: Number of queries to process before saving (for checkpointing)
    Returns:
        DataFrame with generated captions
    """
    # Load submission file
    print(f"\nLoading submission file: {submission_csv_path}")
    df = pd.read_csv(submission_csv_path)
    print(f"Found {len(df)} queries to process.")
    
    # Verify required columns exist
    required_columns = ['query_id', 'generated_caption', 'retrieved_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check if description column exists
    if 'description' not in df.columns:
        print("Warning: 'description' column not found. Will use query_id as fallback.")
    
    # If output_path is not specified, use the input path
    if output_path is None:
        output_path = submission_csv_path
    
    # Create a checkpoint file path
    checkpoint_path = output_path.replace('.csv', '_checkpoint.csv')
    
    # Check if there's a checkpoint file to resume from
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint file. Resuming from previous run...")
        df_checkpoint = pd.read_csv(checkpoint_path)
        # Find where we left off
        processed_mask = df_checkpoint['generated_caption'] != 'Caption to be generated'
        last_processed_idx = processed_mask.sum() - 1
        if last_processed_idx >= 0:
            df = df_checkpoint
            print(f"Resuming from index {last_processed_idx + 1}")
    
    # Process each query
    total_queries = len(df)
    for idx, row in df.iterrows():
        # Skip if already processed
        if row['generated_caption'] != 'Caption to be generated':
            continue
            
        query_id = row['query_id']
        
        # Print progress
        print(f"Processing query {idx + 1}/{total_queries} (ID: {query_id})")
        
        try:
            # Get image description from the CSV description column (if available)
            if 'description' in df.columns and pd.notna(row['description']) and row['description'].strip() != '':
                image_description = row['description']
            else:
                print(f"Warning: No description found for query {query_id}, using fallback")
                image_description = f"An image from query {query_id}"
            
            # Get pre-extracted context from retrieved_text column
            retrieved_context = row['retrieved_text']
            
            if pd.isna(retrieved_context) or retrieved_context.strip() == '':
                print(f"Warning: No retrieved context found for query {query_id}")
                retrieved_context = "No context available."
            
            # Generate caption using Qwen-VL with image description and retrieved context
            caption = generate_caption_qwen(
                image_description,
                retrieved_context,
                qwen_model, 
                qwen_processor
            )
            
            # Update the dataframe
            df.at[idx, 'generated_caption'] = caption
            
            print(f"✅ Generated caption for query {query_id}")
            
            # Save checkpoint every batch_size queries
            if (idx + 1) % batch_size == 0:
                df.to_csv(checkpoint_path, index=False)
                print(f"\n💾 Checkpoint saved at query {idx + 1}")

        except Exception as e:
            print(f"\n❌ Error processing query {query_id}: {str(e)}")
            continue
    
    # Save final results
    df.to_csv(output_path, index=False)
    print(f"\n✅ Final results saved to: {output_path}")

    # Remove checkpoint file if everything completed successfully
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("🧹 Checkpoint file removed.")

    return df

def main():
    """Main function to run the caption generation process"""
    try:
        # Setup paths
        paths = setup_paths()
        print("📁 Paths configured")
        
        qwen_model, qwen_processor, device = load_qwen_model()
        
        # Process submission using pre-extracted context
        print("\n🚀 Starting caption generation process with pre-extracted context...")
        results_df = process_submission(
            submission_csv_path=paths['submission_path'],
            query_img_path=paths['private_img_path'],
            qwen_model=qwen_model,
            qwen_processor=qwen_processor,
            output_path="retrieve_text_llm/submission_context_llm_description.csv",
            batch_size=1
        )
        
        print(f"\n🎉 Caption generation completed successfully!")
        print(f"📊 Processed {len(results_df)} queries")
        
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()