import numpy as np
from sentence_transformers import SentenceTransformer
try:
    import torch
    import clip
except ImportError:
    clip = None  # Mock if not available
from .retriever import Retriever
from typing import Optional, Dict, Any

class HierarchicalMultimodalRetriever(Retriever):
    """
    Hierarchical Multi-Modal Retriever as described in the paper.
    Uses M3E for text embedding and CLIP for image embedding (mocked if not available).
    Implements structure-aware, multi-faceted retrieval.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.text_encoder = SentenceTransformer('moka-ai/m3e-base')
        if clip is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        else:
            self.clip_model = None
            self.clip_preprocess = None
            self.device = "cpu"
        # Default weights (could be learned)
        self.alpha = {'headline': 0.3, 'lead': 0.2, 'body': 0.3, 'caption': 0.2}
        self.beta = {'content': 0.5, 'visual': 0.3, 'position': 0.2}

    def encode_text(self, text: str) -> np.ndarray:
        return self.text_encoder.encode([text])[0]

    def encode_image(self, image_path: str) -> np.ndarray:
        if self.clip_model is not None:
            from PIL import Image
            image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
            return image_features.cpu().numpy()[0]
        else:
            # Mock: return random vector
            return np.random.rand(512)

    def get_article_features(self, article: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # Decompose article into components
        headline = article.get('title', '')
        lead = article.get('lead', '')
        body = article.get('body', '')
        captions = ' '.join(article.get('captions', []))
        # Weighted sum of embeddings
        features = {
            'headline': self.encode_text(headline),
            'lead': self.encode_text(lead),
            'body': self.encode_text(body),
            'caption': self.encode_text(captions)
        }
        weighted = sum(self.alpha[k] * features[k] for k in features)
        return {'weighted_text': weighted, 'features': features}

    def get_article_image_features(self, article: Dict[str, Any]) -> list:
        image_paths = article.get('images', [])
        return [self.encode_image(img_path) for img_path in image_paths]

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Cosine similarity
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def retrieve_context(self, query_image_path: str, query_text: Optional[str] = None, top_k: int = 5) -> str:
        """
        Retrieve relevant context for a query image using hierarchical multi-modal retrieval.
        """
        self.logger.info(f"Hierarchical retrieval for {query_image_path}")
        if not self.database:
            return "No database available for retrieval."
        # Step 1: Encode query image
        v_q = self.encode_image(query_image_path)
        # Step 2: For each article, compute features and similarities
        scored_articles = []
        for article in self.database:
            feats = self.get_article_features(article)
            v_a_imgs = self.get_article_image_features(article)
            # Content-visual alignment
            s_content = self.compute_similarity(v_q, feats['weighted_text'])
            # Visual-visual coherence
            s_visual = max((self.compute_similarity(v_q, v_img) for v_img in v_a_imgs), default=0.0)
            # Discourse positioning (mock: higher if image is first in list)
            s_position = 0.0
            if v_a_imgs:
                s_position = 1.0 if article.get('images', [])[0] else 0.5
            # Aggregate score
            score = (self.beta['content'] * s_content +
                     self.beta['visual'] * s_visual +
                     self.beta['position'] * s_position)
            scored_articles.append((score, article))
        # Step 3: Rank and select top-k
        scored_articles.sort(reverse=True, key=lambda x: x[0])
        top_articles = [a for _, a in scored_articles[:top_k]]
        # Step 4: Extract relevant context (mock: concatenate title and first 2 sentences)
        contexts = []
        for art in top_articles:
            title = art.get('title', '')
            content = art.get('content', '')
            sentences = content.split('. ')
            context = f"{title}: {' '.join(sentences[:2])}"
            contexts.append(context)
        return '\n'.join(contexts) 