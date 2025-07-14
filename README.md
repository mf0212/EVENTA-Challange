# Hierarchical Multi-Modal Retrieval for Knowledge-Grounded News Image Captioning

[![ACM MM 2025](https://img.shields.io/badge/ACM%20MM-2025-blue)](https://2025.acmmm.org/)
[![EVENTA Challenge](https://img.shields.io/badge/EVENTA-Grand%20Challenge-green)](https://eventa.sensemaker.ai/)

A hierarchical multi-modal retrieval-augmented framework for the Event-Enriched Image Captioning Challenge (Track 1). Our system combines sophisticated article retrieval with structured caption generation to produce comprehensive, contextually-aware image descriptions that go beyond simple visual observations.

## 🏆 Challenge Results

- **Final Ranking**: 5th place in EVENTA Grand Challenge (Team: noname_)
- **Overall Score**: 0.2824
- **Retrieval Performance**: mAP 0.708, R@1 0.663, R@10 0.801
- **Captioning Performance**: CIDEr 0.081, CLIP Score 0.783

## 🎯 Problem Statement

Traditional image captioning methods struggle to generate comprehensive, context-rich descriptions, especially for news images where critical details like names, dates, locations, and event significance cannot be inferred from visual cues alone. This challenge requires generating captions that provide:

- **Object Identification**: Names and attributes of people, places, and objects
- **Event Context**: Timing, location, and circumstances of events
- **Factual Details**: Information not directly observable from the image
- **Underlying Significance**: Why the event matters and its broader implications

## 🚀 Our Approach

### Two-Stage Pipeline Architecture

```
Input Image → Hierarchical Multi-Modal Retrieval → Three-Stage Captioning → Enhanced Caption
```

### 1. Hierarchical Multi-Modal Retrieval Module

Our retrieval system treats news articles as structured entities rather than monolithic text, incorporating:

#### **Article Structure-Aware Features**
- **Spatial-Semantic Text Analysis**: Differential weighting of headlines, lead paragraphs, body sections, and image captions
- **Visual Placement Integration**: Considers image positioning within articles as semantic cues

#### **Multi-Faceted Similarity Computation**
- **Content-Visual Alignment**: Similarity between query image and structured article text
- **Visual-Visual Coherence**: Cross-image similarity between query and article-embedded images  
- **Discourse Positioning Score**: Novel metric capturing image placement correlation with event centrality

#### **Contextual Relevance Refinement**
- **Temporal Clustering**: Boosting scores for articles from related time periods
- **Citation Network Analysis**: Leveraging inter-article reference patterns

### 2. Three-Stage Captioning Module

#### **Stage 1: Structured Visual Context Extraction**
Uses Vision-Language Models (VLM) to generate comprehensive image analysis across four dimensions:
- Objective description of scene elements
- Contextual inference about events and settings
- Mood and atmosphere assessment
- Potential headline generation

#### **Stage 2: Relevant Context Extraction**
- Encodes visual context using M3-embedding for language-focused retrieval
- Segments retrieved articles into individual sentences
- Ranks sentences by semantic similarity to visual context
- Extracts top-3 sentences with surrounding context for coherence

#### **Stage 3: Knowledge-Grounded Caption Synthesis**
- Combines structured visual analysis with factual article content
- Uses LLM configured as "expert news photo caption writer"
- Anchors narrative in visual evidence while injecting specific factual details
- Ensures captions are both visually faithful and contextually rich


## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup
```bash
# Clone repository
git clone <repository-url>
cd hierarchical-multimodal-captioning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download OpenEvents V1 dataset
bash scripts/download_dataset.sh
```

## 🚀 Quick Start

### 1. Download and Prepare Data
```bash
# Download OpenEvents V1 dataset
bash scripts/download_dataset.sh

# Preprocess articles for structure analysis
python src/data/preprocessor.py --config config/data_config.yaml
```

### 2. Train Components (Optional)
```bash
# Train retrieval module
python scripts/train_retrieval.py --config config/retrieval_config.yaml

# Train captioning module  
python scripts/train_captioning.py --config config/model_config.yaml
```

### 3. Run Full Pipeline
```bash
# Generate captions for test set
python scripts/run_inference.py \
    --config config/model_config.yaml \
    --input_dir data/openevents_v1/test \
    --output_dir outputs/submissions

# Evaluate results
python scripts/evaluate_results.py \
    --predictions outputs/submissions/captions.json \
    --ground_truth data/openevents_v1/test/annotations.json
```

### 4. Custom Image Captioning
```python
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from src.captioning.caption_synthesizer import CaptionSynthesizer

# Initialize components
retriever = HierarchicalRetriever()
captioner = CaptionSynthesizer()

# Process image
image_path = "path/to/news_image.jpg"
retrieved_articles = retriever.retrieve(image_path, top_k=1)
caption = captioner.generate_caption(image_path, retrieved_articles)

print(f"Generated Caption: {caption}")
```

## 📊 Performance Analysis

### Retrieval Module Ablation
| Method | mAP | R@1 | R@10 |
|--------|-----|-----|------|
| CLIP Baseline | 0.11 | 0.08 | 0.10 |
| Structure-Aware (s_content) | 0.15 | 0.12 | 0.16 |
| + Visual Coherence (s_visual) | 0.95 | 0.942 | 0.985 |
| + Discourse Positioning (Full) | **0.97** | **0.956** | **0.991** |

### Captioning Module Comparison
| Method | CIDEr | CLIP Score |
|--------|-------|------------|
| Image-to-Text (No Retrieval) | 0.039 | 0.890 |
| One-Stage Captioning | - | - |
| Three-Stage Pipeline (Ours) | **0.123** | **0.883** |

## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)
```yaml
# Vision-Language Models
vlm_model: "Qwen/Qwen2-VL-7B-Instruct"
llm_model: "meta-llama/Llama-3.1-8B-Instruct"

# Embedding Models
clip_model: "openai/clip-vit-base-patch32"
text_embedding_model: "BAAI/bge-m3"

# Generation Parameters
max_caption_length: 150
temperature: 0.7
top_p: 0.9
```

### Retrieval Configuration (`config/retrieval_config.yaml`)
```yaml
# Hierarchical Weights
alpha_headline: 0.4
alpha_lead: 0.3
alpha_body: 0.2
alpha_captions: 0.1

# Similarity Combination
beta_content: 0.3
beta_visual: 0.5
beta_position: 0.2

# Retrieval Parameters
top_k: 10
similarity_threshold: 0.1
```

*This repository contains the official implementation for our ACM MM 2025 paper "Hierarchical Multi-Modal Retrieval for Knowledge-Grounded News Image Captioning" submitted to the EVENTA Grand Challenge.*