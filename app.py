import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b0, mobilenet_v2, resnet50

from fruit_groups import GROUP_DISPLAY, group_probabilities


DEFAULT_MODEL_DIR = Path("saved_models_torch")
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


MODEL_CONFIG = {
    "fast_cnn": {
        "display_name": "Fast GPU CNN",
        "checkpoint": "fast_fruit_cnn_64.pt",
        "metadata": "fast_fruit_cnn_64_metadata.json",
    },
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "checkpoint": "mobilenetv2_64.pt",
        "metadata": "mobilenetv2_64_metadata.json",
    },
    "efficientnetb0": {
        "display_name": "EfficientNetB0",
        "checkpoint": "efficientnetb0_64.pt",
        "metadata": "efficientnetb0_64_metadata.json",
    },
    "resnet50": {
        "display_name": "ResNet50",
        "checkpoint": "resnet50_64.pt",
        "metadata": "resnet50_64_metadata.json",
    },
    # Model huấn luyện lại trên ảnh web (notebook v2 / train_web.py), lớp = nhóm trong fruit_groups.py
    "fast_cnn_web": {
        "arch": "fast_cnn",
        "display_name": "Fast GPU CNN (web)",
        "checkpoint": "fast_cnn_web.pt",
        "metadata": "fast_cnn_web_metadata.json",
    },
    "efficientnetb0_web": {
        "arch": "efficientnetb0",
        "display_name": "EfficientNetB0 (web)",
        "checkpoint": "efficientnetb0_web.pt",
        "metadata": "efficientnetb0_web_metadata.json",
    },
    "resnet50_web": {
        "arch": "resnet50",
        "display_name": "ResNet50 (web)",
        "checkpoint": "resnet50_web.pt",
        "metadata": "resnet50_web_metadata.json",
    },
    "mobilenetv2_web": {
        "arch": "mobilenetv2",
        "display_name": "MobileNetV2 (web)",
        "checkpoint": "mobilenetv2_web.pt",
        "metadata": "mobilenetv2_web_metadata.json",
    },
}


CSS = """
.app-shell {
    max-width: 1180px;
    margin: 0 auto;
}
.hero {
    border-bottom: 1px solid #d7ded8;
    padding: 10px 0 14px;
}
.hero h1 {
    font-size: 30px;
    line-height: 1.15;
    margin-bottom: 6px;
}
.hero p {
    color: #475569;
    margin: 0;
}
.result-box {
    border-left: 4px solid #16a34a;
    padding-left: 14px;
}
.result-box h2 {
    font-size: 22px;
    margin: 0 0 4px;
}
.result-box p {
    color: #475569;
    margin: 0;
}
.meta-strip {
    color: #334155;
    font-size: 14px;
}
"""

APP_THEME = gr.themes.Soft(primary_hue="green", secondary_hue="orange")


@dataclass
class ModelBundle:
    model_key: str
    model_name: str
    model: nn.Module
    class_names: np.ndarray
    image_size: int
    test_accuracy: float
    best_val_accuracy: float | None
    checkpoint_path: Path
    metadata_path: Path
    device: torch.device
    grouped: bool = False


class FastFruitCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 192),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class ImageNetNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return (images - self.mean) / self.std


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Gradio UI for the fruit image classifier.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIG), default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--no-clip", action="store_true", help="Chỉ dùng ResNet50, không tải CLIP.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def build_model(model_key, num_classes):
    model_key = MODEL_CONFIG.get(model_key, {}).get("arch", model_key)
    if model_key == "fast_cnn":
        return FastFruitCNN(num_classes)
    if model_key == "mobilenetv2":
        base = mobilenet_v2(weights=None)
        base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, num_classes)
        return nn.Sequential(ImageNetNormalize(), base)
    if model_key == "efficientnetb0":
        base = efficientnet_b0(weights=None)
        base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, num_classes)
        return nn.Sequential(ImageNetNormalize(), base)
    if model_key == "resnet50":
        base = resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        return nn.Sequential(ImageNetNormalize(), base)
    raise ValueError(f"Unsupported model key: {model_key}")


def load_metadata(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_or_none(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if np.isfinite(score) else None


def collect_model_candidates(model_dir):
    model_dir = Path(model_dir)
    candidates = []
    for model_key, config in MODEL_CONFIG.items():
        metadata_path = model_dir / config["metadata"]
        checkpoint_path = model_dir / config["checkpoint"]
        if not metadata_path.exists() or not checkpoint_path.exists():
            continue
        metadata = load_metadata(metadata_path)
        test_accuracy = score_or_none(metadata.get("test_accuracy"))
        if test_accuracy is None:
            continue
        candidates.append(
            {
                "model_key": model_key,
                "model_name": metadata.get("model_name", config["display_name"]),
                "test_accuracy": test_accuracy,
                "best_val_accuracy": score_or_none(metadata.get("best_val_accuracy")),
                "image_size": int(metadata.get("image_size", 64)),
                "grouped": bool(metadata.get("grouped", False)),
                "class_names": np.asarray(metadata.get("classes", []), dtype=str),
                "checkpoint_path": checkpoint_path,
                "metadata_path": metadata_path,
            }
        )
    return candidates


def load_best_model(model_dir=DEFAULT_MODEL_DIR, model_key=None):
    candidates = collect_model_candidates(model_dir)
    if model_key is not None:
        candidates = [item for item in candidates if item["model_key"] == model_key]
    if not candidates:
        raise FileNotFoundError(
            f"No usable PyTorch checkpoints found in {Path(model_dir).resolve()}. "
            "Run the training cells first or check --model-dir."
        )

    # Ưu tiên model huấn luyện trên ảnh web (grouped), rồi mới xét độ chính xác test
    selected = max(candidates, key=lambda item: (item["grouped"], item["test_accuracy"]))
    class_names = selected["class_names"]
    if class_names.size == 0:
        class_names_path = Path(model_dir) / "class_names.npy"
        class_names = np.load(class_names_path, allow_pickle=True).astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(selected["model_key"], len(class_names))
    checkpoint = torch.load(selected["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device, memory_format=torch.channels_last)
    model.eval()

    return ModelBundle(
        model_key=selected["model_key"],
        model_name=selected["model_name"],
        model=model,
        class_names=class_names,
        image_size=selected["image_size"],
        test_accuracy=selected["test_accuracy"],
        best_val_accuracy=selected["best_val_accuracy"],
        checkpoint_path=selected["checkpoint_path"],
        metadata_path=selected["metadata_path"],
        device=device,
        grouped=selected["grouped"],
    )


def pretty_label(label):
    return str(label).replace("_", " ").strip().title()


def predict_probabilities(image, bundle):
    if image is None:
        return None
    image = image.convert("RGB").resize((bundle.image_size, bundle.image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).div(255.0)
    tensor = tensor.to(bundle.device, memory_format=torch.channels_last)
    with torch.inference_mode():
        logits = bundle.model(tensor)
        return torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()


ENGINE_CLIP = "CLIP (model có sẵn, zero-shot)"
ENGINE_RESNET = "Model tự huấn luyện"
ENGINE_COMBINED = "Kết hợp CLIP + model tự huấn luyện"
CLIP_WEIGHT = 0.7


def own_model_group_probabilities(image, bundle):
    probs = predict_probabilities(image, bundle)
    # Model huấn luyện trên ảnh web đã có lớp = nhóm; model cũ (64 lớp) thì cộng xác suất theo nhóm
    return probs if bundle.grouped else group_probabilities(probs, bundle.class_names)


def predict_group_probabilities_for_engine(image, bundle, clip, engine):
    resnet_probs = None
    if engine in (ENGINE_RESNET, ENGINE_COMBINED) or clip is None:
        resnet_probs = own_model_group_probabilities(image, bundle)
    if clip is None or engine == ENGINE_RESNET:
        return resnet_probs
    clip_probs = clip.predict_group_probabilities(image)
    if engine == ENGINE_CLIP:
        return clip_probs
    return CLIP_WEIGHT * clip_probs + (1 - CLIP_WEIGHT) * resnet_probs


def format_prediction_outputs(
    image, bundle, top_k=DEFAULT_TOP_K, threshold=DEFAULT_CONFIDENCE_THRESHOLD, clip=None, engine=ENGINE_CLIP
):
    if image is None:
        empty_rows = [["", "", ""] for _ in range(top_k)]
        return {}, "<div class='result-box'><h2>Chưa có ảnh</h2><p></p></div>", empty_rows

    probabilities = predict_group_probabilities_for_engine(image, bundle, clip, engine)
    top_k = max(1, min(int(top_k), len(GROUP_DISPLAY)))
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    label_scores = {GROUP_DISPLAY[index]: float(probabilities[index]) for index in top_indices}
    best_index = int(top_indices[0])
    best_label = GROUP_DISPLAY[best_index]
    best_score = float(probabilities[best_index])
    rows = [
        [rank, GROUP_DISPLAY[index], f"{float(probabilities[index]):.2%}"]
        for rank, index in enumerate(top_indices, start=1)
    ]
    if best_score < threshold:
        summary = (
            "<div class='result-box'>"
            "<h2>Không nhận diện được</h2>"
            f"<p>Độ tin cậy cao nhất chỉ <strong>{best_score:.2%}</strong> "
            f"(dưới ngưỡng {threshold:.0%}, gần nhất: {best_label}). "
            "Ảnh có thể nằm ngoài dữ liệu huấn luyện.</p>"
            "</div>"
        )
    else:
        summary = (
            "<div class='result-box'>"
            f"<h2>{best_label}</h2>"
            f"<p>Độ tin cậy: <strong>{best_score:.2%}</strong></p>"
            "</div>"
        )
    return label_scores, summary, rows


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def prepare_example_images(class_names, limit=8, output_dir=Path(".gradio") / "demo_examples"):
    x_path = Path("X_test.npy")
    y_path = Path("y_test.npy")
    if not x_path.exists() or not y_path.exists():
        return []

    images = np.load(x_path, mmap_mode="r")
    labels = np.load(y_path, mmap_mode="r")
    preferred_terms = ["apple", "banana", "mango", "grape", "orange", "tomato", "carrot", "cucumber"]
    selected_indices = []
    selected_labels = set()

    for term in preferred_terms:
        matches = [i for i, name in enumerate(class_names) if term in str(name).lower()]
        for label_index in matches:
            positions = np.where(labels == label_index)[0]
            if len(positions) and label_index not in selected_labels:
                selected_indices.append(int(positions[0]))
                selected_labels.add(label_index)
                break
        if len(selected_indices) >= limit:
            break

    if len(selected_indices) < limit:
        for label_index in np.unique(labels):
            if int(label_index) in selected_labels:
                continue
            positions = np.where(labels == label_index)[0]
            if len(positions):
                selected_indices.append(int(positions[0]))
                selected_labels.add(int(label_index))
            if len(selected_indices) >= limit:
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in selected_indices[:limit]:
        label = class_names[int(labels[index])]
        path = output_dir / f"{index:05d}_{sanitize_filename(label)}.jpg"
        if not path.exists():
            Image.fromarray(np.asarray(images[index], dtype=np.uint8)).save(path, quality=95)
        paths.append(str(path))
    return paths


def build_demo(bundle, top_k=DEFAULT_TOP_K, threshold=DEFAULT_CONFIDENCE_THRESHOLD, clip=None):
    accuracy_text = f"{bundle.test_accuracy:.2%}"
    val_text = "N/A" if bundle.best_val_accuracy is None else f"{bundle.best_val_accuracy:.2%}"
    device_text = "GPU CUDA" if bundle.device.type == "cuda" else "CPU"
    examples = [] if bundle.grouped else prepare_example_images(bundle.class_names)

    with gr.Blocks(
        title="Phân loại trái cây qua hình ảnh",
    ) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                f"""
<div class="hero">
  <h1>Phân loại trái cây qua hình ảnh</h1>
  <p class="meta-strip">Model: <strong>{bundle.model_name}</strong> · Test accuracy: <strong>{accuracy_text}</strong> · Validation tốt nhất: <strong>{val_text}</strong> · Thiết bị: <strong>{device_text}</strong></p>
</div>
                """
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=320):
                    image_input = gr.Image(type="pil", label="Ảnh đầu vào", height=430)
                    engines = [ENGINE_CLIP, ENGINE_COMBINED, ENGINE_RESNET] if clip else [ENGINE_RESNET]
                    engine_input = gr.Radio(engines, value=engines[0], label="Bộ nhận diện")
                    with gr.Row():
                        predict_button = gr.Button("Dự đoán", variant="primary")
                        clear_button = gr.Button("Xóa ảnh")
                with gr.Column(scale=4, min_width=300):
                    top_summary = gr.HTML(
                        "<div class='result-box'><h2>Chưa có ảnh</h2><p></p></div>",
                        label="Kết quả chính",
                    )
                    label_output = gr.Label(num_top_classes=top_k, label=f"Top {top_k}")
                    table_output = gr.Dataframe(
                        headers=["Hạng", "Lớp", "Xác suất"],
                        datatype=["number", "str", "str"],
                        row_count=(top_k, "fixed"),
                        interactive=False,
                        label="Chi tiết dự đoán",
                    )
            if examples:
                gr.Examples(examples=examples, inputs=image_input, label="Ảnh mẫu từ tập test")

        def classify(image, engine):
            return format_prediction_outputs(
                image, bundle, top_k=top_k, threshold=threshold, clip=clip, engine=engine
            )

        def clear():
            rows = [["", "", ""] for _ in range(top_k)]
            return None, {}, "<div class='result-box'><h2>Chưa có ảnh</h2><p></p></div>", rows

        outputs = [label_output, top_summary, table_output]
        predict_button.click(classify, inputs=[image_input, engine_input], outputs=outputs)
        image_input.change(classify, inputs=[image_input, engine_input], outputs=outputs)
        engine_input.change(classify, inputs=[image_input, engine_input], outputs=outputs)
        clear_button.click(clear, outputs=[image_input, label_output, top_summary, table_output])

    return demo


def launch_demo(demo, **kwargs):
    return demo.launch(theme=APP_THEME, css=CSS, **kwargs)


def main():
    args = parse_args()
    bundle = load_best_model(args.model_dir, args.model_key)
    print(f"Loaded model: {bundle.model_name}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print(f"Classes: {len(bundle.class_names)} | image size: {bundle.image_size} | device: {bundle.device}")
    clip = None
    if not args.no_clip:
        from clip_classifier import ClipClassifier

        clip = ClipClassifier()
        print("Loaded CLIP zero-shot classifier")
    demo = build_demo(bundle, top_k=args.top_k, threshold=args.threshold, clip=clip)
    launch_demo(demo, server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
