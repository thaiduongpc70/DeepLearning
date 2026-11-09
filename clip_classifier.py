"""Phân loại zero-shot bằng CLIP trên các nhóm trong fruit_groups.py."""

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from fruit_groups import GROUP_PROMPTS

DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"

TEMPLATES = [
    "a photo of {}.",
    "a photo of a {}, a type of fruit or vegetable.",
    "a close-up photo of {}.",
    "a photo of fresh {} on a white background.",
    "a photo of {} in a market.",
    "a product photo of {}.",
]


class ClipClassifier:
    def __init__(self, model_name=DEFAULT_CLIP_MODEL, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device).eval()
        self.text_features = self._encode_prompts()

    @torch.inference_mode()
    def _encode_prompts(self):
        features = []
        for prompt in GROUP_PROMPTS:
            texts = [template.format(prompt) for template in TEMPLATES]
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            emb = self.model.get_text_features(**inputs)
            emb = getattr(emb, "pooler_output", emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(dim=0)
            features.append(mean / mean.norm())
        return torch.stack(features)

    @torch.inference_mode()
    def predict_group_probabilities(self, image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(self.dtype)
        emb = self.model.get_image_features(pixel_values=pixel_values)
        emb = getattr(emb, "pooler_output", emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        logits = 100.0 * emb @ self.text_features.T
        return torch.softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
