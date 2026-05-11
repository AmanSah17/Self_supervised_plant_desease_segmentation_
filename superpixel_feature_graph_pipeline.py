from __future__ import annotations

import os
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import networkx as nx
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import entropy
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, find_boundaries, slic, watershed
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModel

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops


@dataclass
class PipelineConfig:
    dataset_base: str = "Lettuce_disease_datasets_split"
    output_base: str = "feature_extraction_output"
    splits: Tuple[str, ...] = ("train", "validation", "test")

    superpixel_algorithm: str = "slic"
    num_segments: int = 1600
    adaptive_segments: bool = True
    segments_per_megapixel: int = 2600

    compactness: float = 6.0
    slic_sigma: float = 0.8
    enforce_connectivity: bool = True

    felz_scale: float = 120.0
    felz_sigma: float = 0.6
    felz_min_size: int = 12

    watershed_compactness: float = 0.002
    watershed_min_size: int = 10

    context_dilation: int = 3
    crop_padding: int = 4

    lbp_n_points: int = 8
    lbp_radius: int = 1
    lbp_method: str = "nri_uniform"
    glcm_levels: int = 32
    glcm_distances: Tuple[int, ...] = (1, 2)
    glcm_angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

    n_clusters: int = 3
    feature_sigma: float = 2.5
    feature_distance_threshold: float = 3.0
    k_feature_neighbors: int = 2
    unary_weight: float = 1.0
    pairwise_weight: float = 1.6
    crf_iterations: int = 4

    use_deep_features: bool = False
    deep_feature_models: Tuple[str, ...] = ("resnet18", "efficientnet_b0", "vit_b_16", "dinov2", "clip")
    deep_batch_size: int = 24
    deep_crop_size: int = 224
    deep_context_dilation: int = 5
    deep_pretrained: bool = True
    deep_local_files_only: bool = False
    skip_unavailable_deep_models: bool = True
    device: Optional[str] = None

    samples_per_class: Optional[int] = None
    save_features: bool = True
    save_masks: bool = True
    save_graphs: bool = True
    overwrite: bool = False


class TorchvisionEmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values)


class TransformersEmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, pooler_attr: Optional[str] = None):
        super().__init__()
        self.backbone = backbone
        self.pooler_attr = pooler_attr

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        if self.pooler_attr and hasattr(outputs, self.pooler_attr):
            pooled = getattr(outputs, self.pooler_attr)
            if pooled is not None:
                return pooled
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        raise ValueError("Unsupported transformer output for embedding extraction")


class DeepFeatureExtractor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(
            config.device
            if config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.models: List[dict] = []
        if config.use_deep_features:
            self._load_requested_models()

    def _load_requested_models(self) -> None:
        for model_name in self.config.deep_feature_models:
            try:
                self.models.append(self._load_model(model_name))
            except Exception as exc:
                if self.config.skip_unavailable_deep_models:
                    warnings.warn(f"Skipping deep feature model '{model_name}': {exc}")
                else:
                    raise

    def _load_model(self, model_name: str) -> dict:
        normalized = model_name.lower()
        if normalized == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if self.config.deep_pretrained else None
            backbone = models.resnet18(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            preprocess = weights.transforms() if weights is not None else None
            return {
                "name": "resnet18",
                "kind": "torchvision",
                "model": TorchvisionEmbeddingModel(backbone.eval().to(self.device)),
                "preprocess": preprocess,
                "feature_dim": feature_dim,
            }

        if normalized == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if self.config.deep_pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            preprocess = weights.transforms() if weights is not None else None
            return {
                "name": "efficientnet_b0",
                "kind": "torchvision",
                "model": TorchvisionEmbeddingModel(backbone.eval().to(self.device)),
                "preprocess": preprocess,
                "feature_dim": feature_dim,
            }

        if normalized == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if self.config.deep_pretrained else None
            backbone = models.vit_b_16(weights=weights)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            preprocess = weights.transforms() if weights is not None else None
            return {
                "name": "vit_b_16",
                "kind": "torchvision",
                "model": TorchvisionEmbeddingModel(backbone.eval().to(self.device)),
                "preprocess": preprocess,
                "feature_dim": feature_dim,
            }

        if normalized == "dinov2":
            repo_id = "facebook/dinov2-base"
            processor = AutoImageProcessor.from_pretrained(
                repo_id,
                local_files_only=self.config.deep_local_files_only,
            )
            backbone = AutoModel.from_pretrained(
                repo_id,
                local_files_only=self.config.deep_local_files_only,
            )
            feature_dim = int(getattr(backbone.config, "hidden_size", 768))
            return {
                "name": "dinov2",
                "kind": "transformers",
                "model": TransformersEmbeddingModel(backbone.eval().to(self.device)),
                "processor": processor,
                "feature_dim": feature_dim,
            }

        if normalized == "clip":
            repo_id = "openai/clip-vit-base-patch32"
            processor = CLIPImageProcessor.from_pretrained(
                repo_id,
                local_files_only=self.config.deep_local_files_only,
            )
            backbone = CLIPVisionModel.from_pretrained(
                repo_id,
                local_files_only=self.config.deep_local_files_only,
            )
            feature_dim = int(getattr(backbone.config, "hidden_size", 768))
            return {
                "name": "clip",
                "kind": "transformers",
                "model": TransformersEmbeddingModel(backbone.eval().to(self.device)),
                "processor": processor,
                "feature_dim": feature_dim,
            }

        raise ValueError(f"Unsupported deep feature model: {model_name}")

    @property
    def enabled(self) -> bool:
        return len(self.models) > 0

    def get_feature_names(self) -> List[str]:
        names: List[str] = []
        for entry in self.models:
            names.extend(
                [f"{entry['name']}_{index:04d}" for index in range(entry["feature_dim"])]
            )
        return names

    def _build_segment_crops(self, img_rgb: np.ndarray, segments: np.ndarray) -> List[Image.Image]:
        crops: List[Image.Image] = []
        segment_slices = ndi.find_objects(segments + 1)
        pad = self.config.deep_context_dilation + self.config.crop_padding
        height, width = segments.shape

        for seg_id, seg_slice in enumerate(segment_slices):
            if seg_slice is None:
                crops.append(Image.fromarray(np.zeros((self.config.deep_crop_size, self.config.deep_crop_size, 3), dtype=np.uint8)))
                continue

            y0, y1 = seg_slice[0].start, seg_slice[0].stop
            x0, x1 = seg_slice[1].start, seg_slice[1].stop
            ey0 = max(0, y0 - pad)
            ey1 = min(height, y1 + pad)
            ex0 = max(0, x0 - pad)
            ex1 = min(width, x1 + pad)

            crop = img_rgb[ey0:ey1, ex0:ex1].copy()
            crop_mask = segments[ey0:ey1, ex0:ex1] == seg_id
            ring_mask = ndi.binary_dilation(crop_mask, iterations=self.config.deep_context_dilation)
            keep_mask = ring_mask[..., None]
            background = np.full_like(crop, int(crop[crop_mask].mean()) if np.any(crop_mask) else 0)
            crop = np.where(keep_mask, crop, background)
            crops.append(Image.fromarray(crop))
        return crops

    def _extract_torchvision_embeddings(self, entry: dict, crops: Sequence[Image.Image]) -> np.ndarray:
        preprocess = entry["preprocess"]
        tensors: List[torch.Tensor] = []
        for crop in crops:
            if preprocess is not None:
                tensors.append(preprocess(crop))
            else:
                crop_resized = crop.resize((self.config.deep_crop_size, self.config.deep_crop_size))
                crop_array = np.asarray(crop_resized).astype(np.float32) / 255.0
                tensor = torch.from_numpy(crop_array.transpose(2, 0, 1))
                tensors.append(tensor)

        embeddings: List[np.ndarray] = []
        model = entry["model"]
        with torch.inference_mode():
            for start in range(0, len(tensors), self.config.deep_batch_size):
                batch = torch.stack(tensors[start:start + self.config.deep_batch_size]).to(self.device)
                output = model(batch)
                embeddings.append(output.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embeddings)

    def _extract_transformer_embeddings(self, entry: dict, crops: Sequence[Image.Image]) -> np.ndarray:
        processor = entry["processor"]
        model = entry["model"]
        embeddings: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.config.deep_batch_size):
                batch_crops = list(crops[start:start + self.config.deep_batch_size])
                inputs = processor(images=batch_crops, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = model(inputs["pixel_values"])
                embeddings.append(output.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embeddings)

    def extract(self, img_rgb: np.ndarray, segments: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.zeros((int(segments.max() + 1), 0), dtype=np.float32)

        crops = self._build_segment_crops(img_rgb, segments)
        all_embeddings: List[np.ndarray] = []
        for entry in self.models:
            if entry["kind"] == "torchvision":
                all_embeddings.append(self._extract_torchvision_embeddings(entry, crops))
            else:
                all_embeddings.append(self._extract_transformer_embeddings(entry, crops))
        return np.concatenate(all_embeddings, axis=1).astype(np.float32)


def lbp_bins(config: PipelineConfig) -> int:
    if config.lbp_method == "nri_uniform":
        return config.lbp_n_points * (config.lbp_n_points - 1) + 3
    if config.lbp_method == "uniform":
        return config.lbp_n_points + 2
    if config.lbp_method == "var":
        return 16
    return 2 ** config.lbp_n_points


def get_feature_names(
    config: PipelineConfig,
    deep_feature_extractor: Optional[DeepFeatureExtractor] = None,
) -> List[str]:
    names = [
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "mean_h",
        "mean_s",
        "mean_v",
        "std_h",
        "std_s",
        "std_v",
        "mean_l",
        "mean_a",
        "mean_b_lab",
        "std_l",
        "std_a",
        "std_b_lab",
    ]

    names.extend([f"lbp_{idx}" for idx in range(lbp_bins(config))])
    names.extend(
        [
            "glcm_contrast",
            "glcm_dissimilarity",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
            "glcm_asm",
            "entropy",
            "grad_mean",
            "grad_std",
            "grad_p25",
            "grad_p75",
            "ring_mean_l",
            "ring_mean_a",
            "ring_mean_b_lab",
            "delta_l",
            "delta_a",
            "delta_b_lab",
            "area_ratio",
            "bbox_fill_ratio",
            "boundary_grad_mean",
            "boundary_grad_std",
            "centroid_y",
            "centroid_x",
        ]
    )
    if deep_feature_extractor is not None and deep_feature_extractor.enabled:
        names.extend(deep_feature_extractor.get_feature_names())
    return names


def ensure_output_dirs(config: PipelineConfig) -> None:
    base = Path(config.output_base)
    for split in config.splits:
        for artifact in ("features", "masks", "graphs"):
            (base / artifact / split).mkdir(parents=True, exist_ok=True)


def collect_image_paths(split_dir: Path, samples_per_class: Optional[int]) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        image_path_map: Dict[str, Path] = {}
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for path in class_dir.glob(pattern):
                image_path_map[str(path.resolve()).lower()] = path
        image_paths = sorted(image_path_map.values())
        if samples_per_class is not None:
            image_paths = image_paths[:samples_per_class]
        items.extend((class_dir.name, path) for path in image_paths)
    return items


def load_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def relabel_segments(segments: np.ndarray) -> np.ndarray:
    labels, relabeled = np.unique(segments, return_inverse=True)
    return relabeled.reshape(segments.shape).astype(np.int32)


def compute_edge_map(img_rgb: np.ndarray) -> np.ndarray:
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = img_lab[..., 0] / 255.0
    a_channel = (img_lab[..., 1] - 128.0) / 127.0
    b_channel = (img_lab[..., 2] - 128.0) / 127.0

    edge_map = 0.6 * sobel(l_channel) + 0.2 * sobel(a_channel) + 0.2 * sobel(b_channel)
    edge_map = edge_map.astype(np.float32)
    return (edge_map - edge_map.min()) / (np.ptp(edge_map) + 1e-6)


def resolve_target_segments(image_shape: Tuple[int, int], config: PipelineConfig) -> int:
    height, width = image_shape
    target = config.num_segments
    if config.adaptive_segments:
        megapixels = (height * width) / 1_000_000.0
        adaptive_target = int(round(config.segments_per_megapixel * megapixels))
        target = max(target, adaptive_target)
    return max(target, 64)


def build_watershed_markers(image_shape: Tuple[int, int], target_segments: int) -> np.ndarray:
    height, width = image_shape
    aspect = width / max(height, 1)
    rows = max(1, int(np.sqrt(target_segments / max(aspect, 1e-6))))
    cols = max(1, int(np.ceil(target_segments / rows)))

    ys = np.linspace(0, height - 1, rows, dtype=np.int32)
    xs = np.linspace(0, width - 1, cols, dtype=np.int32)
    markers = np.zeros((height, width), dtype=np.int32)

    marker_id = 1
    for y in ys:
        for x in xs:
            markers[y, x] = marker_id
            marker_id += 1
    return markers


def merge_tiny_segments(segments: np.ndarray, edge_map: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return segments

    segments = segments.copy()
    changed = True
    while changed:
        changed = False
        ids, counts = np.unique(segments, return_counts=True)
        small_ids = ids[counts < min_size]
        if len(small_ids) == 0:
            break

        adjacency, _ = find_adjacent_superpixels(segments)
        for seg_id in small_ids:
            mask = segments == seg_id
            if not np.any(mask):
                continue
            neighbors = list(adjacency.get(seg_id, ()))
            if not neighbors:
                continue

            best_neighbor = neighbors[0]
            best_score = float("inf")
            for neighbor in neighbors:
                neighbor_mask = segments == neighbor
                if not np.any(neighbor_mask):
                    continue
                mean_gap = abs(edge_map[mask].mean() - edge_map[neighbor_mask].mean())
                if mean_gap < best_score:
                    best_neighbor = neighbor
                    best_score = mean_gap
            segments[mask] = best_neighbor
            changed = True

        if changed:
            segments = relabel_segments(segments)
    return segments


def generate_superpixels(img_rgb: np.ndarray, config: PipelineConfig) -> np.ndarray:
    target_segments = resolve_target_segments(img_rgb.shape[:2], config)
    edge_map = compute_edge_map(img_rgb)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    slic_input = np.dstack([img_lab, edge_map])

    algorithm = config.superpixel_algorithm.lower()
    if algorithm == "slic":
        segments = slic(
            slic_input,
            n_segments=target_segments,
            compactness=config.compactness,
            sigma=config.slic_sigma,
            start_label=0,
            convert2lab=False,
            enforce_connectivity=config.enforce_connectivity,
            channel_axis=-1,
        )
    elif algorithm == "felzenszwalb":
        segments = felzenszwalb(
            img_lab,
            scale=config.felz_scale,
            sigma=config.felz_sigma,
            min_size=config.felz_min_size,
            channel_axis=-1,
        )
    elif algorithm == "watershed":
        markers = build_watershed_markers(img_rgb.shape[:2], target_segments)
        segments = watershed(
            edge_map,
            markers=markers,
            compactness=config.watershed_compactness,
            watershed_line=False,
        )
        segments = merge_tiny_segments(segments, edge_map, config.watershed_min_size)
    else:
        raise ValueError(f"Unsupported superpixel algorithm: {config.superpixel_algorithm}")

    return relabel_segments(segments)


def safe_stats(pixels: np.ndarray, n_channels: int) -> np.ndarray:
    if pixels.size == 0:
        return np.zeros(n_channels * 2, dtype=np.float32)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


def quantize_gray(image_gray: np.ndarray, levels: int) -> np.ndarray:
    return np.clip((image_gray.astype(np.float32) / 255.0) * (levels - 1), 0, levels - 1).astype(np.uint8)


def extract_texture_features(
    quantized_patch: np.ndarray,
    patch_gray: np.ndarray,
    patch_mask: np.ndarray,
    patch_grad: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    region_gray = patch_gray[patch_mask]
    region_grad = patch_grad[patch_mask]

    lbp = local_binary_pattern(
        patch_gray,
        config.lbp_n_points,
        config.lbp_radius,
        method=config.lbp_method,
    )
    lbp_values = lbp[patch_mask]
    lbp_hist, _ = np.histogram(lbp_values, bins=lbp_bins(config), range=(0, lbp_bins(config)))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= lbp_hist.sum() + 1e-6

    try:
        glcm = graycomatrix(
            quantized_patch,
            distances=list(config.glcm_distances),
            angles=list(config.glcm_angles),
            levels=config.glcm_levels,
            symmetric=True,
            normed=True,
        )
        glcm_props = np.array(
            [
                graycoprops(glcm, "contrast").mean(),
                graycoprops(glcm, "dissimilarity").mean(),
                graycoprops(glcm, "homogeneity").mean(),
                graycoprops(glcm, "energy").mean(),
                graycoprops(glcm, "correlation").mean(),
                graycoprops(glcm, "ASM").mean(),
            ],
            dtype=np.float32,
        )
    except Exception:
        glcm_props = np.zeros(6, dtype=np.float32)

    if region_gray.size:
        gray_hist, _ = np.histogram(region_gray, bins=256, range=(0, 256))
        gray_hist = gray_hist.astype(np.float32)
        gray_hist /= gray_hist.sum() + 1e-6
        gray_entropy = np.array([entropy(gray_hist)], dtype=np.float32)
        grad_stats = np.array(
            [
                float(np.mean(region_grad)),
                float(np.std(region_grad)),
                float(np.percentile(region_grad, 25)),
                float(np.percentile(region_grad, 75)),
            ],
            dtype=np.float32,
        )
    else:
        gray_entropy = np.zeros(1, dtype=np.float32)
        grad_stats = np.zeros(4, dtype=np.float32)

    return np.concatenate([lbp_hist, glcm_props, gray_entropy, grad_stats]).astype(np.float32)


def find_adjacent_superpixels(segments: np.ndarray) -> Tuple[Dict[int, set], np.ndarray]:
    left_right = np.column_stack((segments[:, :-1].ravel(), segments[:, 1:].ravel()))
    up_down = np.column_stack((segments[:-1, :].ravel(), segments[1:, :].ravel()))
    pairs = np.vstack((left_right, up_down))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.sort(pairs, axis=1)
    unique_pairs = np.unique(pairs, axis=0)

    adjacency: Dict[int, set] = {int(seg_id): set() for seg_id in np.unique(segments)}
    for seg_a, seg_b in unique_pairs:
        adjacency[int(seg_a)].add(int(seg_b))
        adjacency[int(seg_b)].add(int(seg_a))
    return adjacency, unique_pairs


def extract_superpixel_features(
    img_rgb: np.ndarray,
    segments: np.ndarray,
    config: PipelineConfig,
    deep_feature_extractor: Optional[DeepFeatureExtractor] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_map = compute_edge_map(img_rgb)

    feature_rows: List[np.ndarray] = []
    metadata_rows: List[dict] = []

    segment_slices = ndi.find_objects(segments + 1)
    image_area = float(segments.shape[0] * segments.shape[1])

    for seg_id, seg_slice in enumerate(segment_slices):
        if seg_slice is None:
            continue

        y0, y1 = seg_slice[0].start, seg_slice[0].stop
        x0, x1 = seg_slice[1].start, seg_slice[1].stop
        patch_mask = segments[y0:y1, x0:x1] == seg_id
        if not np.any(patch_mask):
            continue

        patch_rgb = img_rgb[y0:y1, x0:x1]
        patch_hsv = img_hsv[y0:y1, x0:x1]
        patch_lab = img_lab[y0:y1, x0:x1]
        patch_gray = img_gray[y0:y1, x0:x1]
        patch_grad = grad_map[y0:y1, x0:x1]

        rgb_pixels = patch_rgb[patch_mask]
        hsv_pixels = patch_hsv[patch_mask]
        lab_pixels = patch_lab[patch_mask]
        gray_pixels = patch_gray[patch_mask]
        grad_pixels = patch_grad[patch_mask]

        fill_value = int(np.median(gray_pixels)) if gray_pixels.size else 0
        filled_gray = np.where(patch_mask, patch_gray, fill_value).astype(np.uint8)
        quantized_patch = quantize_gray(filled_gray, config.glcm_levels)

        pad = config.context_dilation + 1
        ey0 = max(0, y0 - pad)
        ey1 = min(segments.shape[0], y1 + pad)
        ex0 = max(0, x0 - pad)
        ex1 = min(segments.shape[1], x1 + pad)

        expanded_mask = segments[ey0:ey1, ex0:ex1] == seg_id
        ring_mask = ndi.binary_dilation(expanded_mask, iterations=config.context_dilation)
        ring_mask = np.logical_and(ring_mask, ~expanded_mask)
        ring_lab_pixels = img_lab[ey0:ey1, ex0:ex1][ring_mask]
        if ring_lab_pixels.size == 0:
            ring_lab_pixels = lab_pixels

        boundary_mask = find_boundaries(expanded_mask, mode="inner")
        boundary_grads = grad_map[ey0:ey1, ex0:ex1][boundary_mask]
        if boundary_grads.size == 0:
            boundary_grads = grad_pixels

        color_features = np.concatenate(
            [
                safe_stats(rgb_pixels, 3),
                safe_stats(hsv_pixels, 3),
                safe_stats(lab_pixels, 3),
            ]
        )
        texture_features = extract_texture_features(
            quantized_patch=quantized_patch,
            patch_gray=patch_gray,
            patch_mask=patch_mask,
            patch_grad=patch_grad,
            config=config,
        )

        ring_mean = ring_lab_pixels.mean(axis=0) if ring_lab_pixels.size else np.zeros(3, dtype=np.float32)
        core_mean = lab_pixels.mean(axis=0) if lab_pixels.size else np.zeros(3, dtype=np.float32)
        delta_mean = ring_mean - core_mean

        ys, xs = np.where(patch_mask)
        centroid_y = float((ys.mean() + y0) / max(segments.shape[0], 1))
        centroid_x = float((xs.mean() + x0) / max(segments.shape[1], 1))
        area = int(patch_mask.sum())
        bbox_area = int(patch_mask.shape[0] * patch_mask.shape[1])

        geometry_features = np.array(
            [
                float(area / image_area),
                float(area / max(bbox_area, 1)),
                float(np.mean(boundary_grads)) if boundary_grads.size else 0.0,
                float(np.std(boundary_grads)) if boundary_grads.size else 0.0,
                centroid_y,
                centroid_x,
            ],
            dtype=np.float32,
        )

        context_features = np.concatenate(
            [ring_mean.astype(np.float32), delta_mean.astype(np.float32), geometry_features]
        )
        features = np.concatenate([color_features, texture_features, context_features]).astype(np.float32)

        feature_rows.append(features)
        metadata_rows.append(
            {
                "segment_id": seg_id,
                "area": area,
                "bbox_height": patch_mask.shape[0],
                "bbox_width": patch_mask.shape[1],
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "boundary_grad_mean": geometry_features[2],
                "boundary_grad_std": geometry_features[3],
            }
        )

    feature_matrix = (
        np.vstack(feature_rows)
        if feature_rows
        else np.zeros((0, len(get_feature_names(config, deep_feature_extractor))), dtype=np.float32)
    )

    if deep_feature_extractor is not None and deep_feature_extractor.enabled and len(feature_matrix):
        deep_features = deep_feature_extractor.extract(img_rgb, segments)
        if len(deep_features) == len(feature_matrix):
            feature_matrix = np.concatenate([feature_matrix, deep_features], axis=1).astype(np.float32)
        else:
            warnings.warn(
                "Deep feature count did not match classical feature count; skipping deep feature concatenation."
            )
    metadata = pd.DataFrame(metadata_rows)
    return feature_matrix, metadata


def build_adjacency_graph(
    segments: np.ndarray,
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> nx.Graph:
    adjacency, adjacent_pairs = find_adjacent_superpixels(segments)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features) if len(features) else features

    graph = nx.Graph()
    for seg_id, feature_vector in enumerate(features):
        node_meta = metadata.iloc[seg_id].to_dict()
        graph.add_node(seg_id, features=feature_vector, **node_meta)

    for seg_a, seg_b in adjacent_pairs:
        feat_distance = float(np.linalg.norm(scaled_features[seg_a] - scaled_features[seg_b]))
        weight = float(np.exp(-(feat_distance ** 2) / (2 * (config.feature_sigma ** 2))))
        graph.add_edge(int(seg_a), int(seg_b), weight=weight, distance=feat_distance, relation="adjacent")

    if len(features) > 1 and config.k_feature_neighbors > 0:
        n_neighbors = min(config.k_feature_neighbors + 1, len(features))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(scaled_features)
        distances, indices = nn.kneighbors(scaled_features)

        for seg_id in range(len(features)):
            for distance, neighbor_id in zip(distances[seg_id][1:], indices[seg_id][1:]):
                if distance > config.feature_distance_threshold:
                    continue
                if neighbor_id in adjacency.get(seg_id, set()):
                    continue
                if graph.has_edge(seg_id, int(neighbor_id)):
                    continue
                weight = float(np.exp(-(distance ** 2) / (2 * (config.feature_sigma ** 2))))
                graph.add_edge(
                    seg_id,
                    int(neighbor_id),
                    weight=weight,
                    distance=float(distance),
                    relation="feature_knn",
                )
    return graph


def perform_graph_clustering(graph: nx.Graph, features: np.ndarray, n_clusters: int) -> np.ndarray:
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return np.zeros(0, dtype=np.uint8)
    if node_count == 1:
        return np.zeros(1, dtype=np.uint8)

    n_clusters = max(1, min(n_clusters, node_count))
    if n_clusters == 1:
        return np.zeros(node_count, dtype=np.uint8)

    adjacency_matrix = nx.to_numpy_array(graph, weight="weight", dtype=np.float32)
    adjacency_matrix += np.eye(node_count, dtype=np.float32) * 1e-6

    try:
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
            n_init=10,
        )
        labels = clustering.fit_predict(adjacency_matrix)
    except Exception:
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(features)

    return labels.astype(np.uint8)


def refine_labels_with_graph_crf(
    graph: nx.Graph,
    features: np.ndarray,
    labels: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    if len(labels) == 0 or graph.number_of_nodes() == 0:
        return labels

    n_clusters = int(labels.max()) + 1
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features) if len(features) else features
    centroids = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if np.any(cluster_mask):
            centroids.append(scaled_features[cluster_mask].mean(axis=0))
        else:
            centroids.append(np.zeros(scaled_features.shape[1], dtype=np.float32))
    centroids = np.vstack(centroids)

    distances = np.linalg.norm(scaled_features[:, None, :] - centroids[None, :, :], axis=2)
    unary = np.exp(-distances)
    unary /= unary.sum(axis=1, keepdims=True) + 1e-6

    probs = unary.copy()
    for _ in range(config.crf_iterations):
        updated = config.unary_weight * unary
        for node in graph.nodes:
            neighbor_message = np.zeros(n_clusters, dtype=np.float32)
            for neighbor in graph.neighbors(node):
                edge_weight = graph[node][neighbor].get("weight", 1.0)
                neighbor_message += float(edge_weight) * probs[neighbor]
            updated[node] += config.pairwise_weight * neighbor_message
        probs = updated / (updated.sum(axis=1, keepdims=True) + 1e-6)

    return np.argmax(probs, axis=1).astype(np.uint8)


def cluster_labels_to_mask(segments: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(segments, dtype=np.uint8)
    for seg_id, label_value in enumerate(labels):
        mask[segments == seg_id] = label_value
    return mask


def save_artifacts(
    image_path: Path,
    class_name: str,
    split: str,
    config: PipelineConfig,
    segments: np.ndarray,
    features: np.ndarray,
    metadata: pd.DataFrame,
    graph: nx.Graph,
    cluster_mask: np.ndarray,
    feature_names: Sequence[str],
) -> None:
    stem = image_path.stem
    base_dir = Path(config.output_base)

    feature_dir = base_dir / "features" / split / class_name
    mask_dir = base_dir / "masks" / split / class_name
    graph_dir = base_dir / "graphs" / split / class_name
    feature_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    if config.save_features:
        np.savez_compressed(
            feature_dir / f"{stem}.npz",
            features=features,
            segments=segments,
            feature_names=np.array(feature_names, dtype=object),
            metadata_columns=np.array(metadata.columns, dtype=object),
            metadata=metadata.to_numpy(),
        )

    if config.save_masks:
        cv2.imwrite(str(mask_dir / f"{stem}.png"), cluster_mask)

    if config.save_graphs:
        with open(graph_dir / f"{stem}.pkl", "wb") as handle:
            pickle.dump(graph, handle)


def get_artifact_paths(
    image_path: Path,
    class_name: str,
    split: str,
    config: PipelineConfig,
) -> Dict[str, Path]:
    stem = image_path.stem
    base_dir = Path(config.output_base)
    return {
        "feature": base_dir / "features" / split / class_name / f"{stem}.npz",
        "mask": base_dir / "masks" / split / class_name / f"{stem}.png",
        "graph": base_dir / "graphs" / split / class_name / f"{stem}.pkl",
    }


def process_image(
    image_path: Path,
    class_name: str,
    split: str,
    config: PipelineConfig,
    feature_names: Sequence[str],
    deep_feature_extractor: Optional[DeepFeatureExtractor] = None,
) -> dict:
    started = time.time()
    artifact_paths = get_artifact_paths(image_path, class_name, split, config)
    if (
        not config.overwrite
        and artifact_paths["feature"].exists()
        and artifact_paths["mask"].exists()
        and artifact_paths["graph"].exists()
    ):
        return {
            "split": split,
            "class_name": class_name,
            "image_path": str(image_path),
            "status": "skipped_existing",
            "elapsed_seconds": 0.0,
        }

    img_rgb = load_rgb_image(image_path)
    segments = generate_superpixels(img_rgb, config)
    features, metadata = extract_superpixel_features(img_rgb, segments, config, deep_feature_extractor)
    graph = build_adjacency_graph(segments, features, metadata, config)
    labels = perform_graph_clustering(graph, features, config.n_clusters)
    labels = refine_labels_with_graph_crf(graph, features, labels, config)
    cluster_mask = cluster_labels_to_mask(segments, labels)

    save_artifacts(
        image_path=image_path,
        class_name=class_name,
        split=split,
        config=config,
        segments=segments,
        features=features,
        metadata=metadata,
        graph=graph,
        cluster_mask=cluster_mask,
        feature_names=feature_names,
    )

    return {
        "split": split,
        "class_name": class_name,
        "image_path": str(image_path),
        "height": img_rgb.shape[0],
        "width": img_rgb.shape[1],
        "segments": int(segments.max() + 1),
        "feature_dim": int(features.shape[1]),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def process_dataset(config: PipelineConfig) -> pd.DataFrame:
    ensure_output_dirs(config)
    deep_feature_extractor = DeepFeatureExtractor(config)
    feature_names = get_feature_names(config, deep_feature_extractor)
    dataset_base = Path(config.dataset_base)

    all_rows: List[dict] = []
    for split in config.splits:
        split_dir = dataset_base / split
        if not split_dir.exists():
            continue

        items = collect_image_paths(split_dir, config.samples_per_class)
        progress = tqdm(items, desc=f"{split:>10}", unit="image")
        for class_name, image_path in progress:
            try:
                row = process_image(
                    image_path,
                    class_name,
                    split,
                    config,
                    feature_names,
                    deep_feature_extractor,
                )
                all_rows.append(row)
                if row.get("status") == "skipped_existing":
                    progress.set_postfix({"status": "resume-skip"})
                else:
                    progress.set_postfix({"segments": row["segments"], "edges": row["graph_edges"]})
            except Exception as exc:
                all_rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "image_path": str(image_path),
                        "error": str(exc),
                    }
                )

    summary = pd.DataFrame(all_rows)
    summary_path = Path(config.output_base) / "processing_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def print_recommendations(config: PipelineConfig) -> None:
    print("Pipeline configuration")
    print(f"  Algorithm: {config.superpixel_algorithm}")
    print(f"  Base segments: {config.num_segments}")
    print(f"  Adaptive segments: {config.adaptive_segments}")
    print(f"  Segments per megapixel: {config.segments_per_megapixel}")
    print(f"  Context dilation: {config.context_dilation}")
    print(f"  Feature distance threshold: {config.feature_distance_threshold}")
    print(f"  Deep features enabled: {config.use_deep_features}")
    if config.use_deep_features:
        print(f"  Deep models: {', '.join(config.deep_feature_models)}")
    print("")
    print("Tuning notes")
    print("  - SLIC: increase num_segments and lower compactness to follow disease edges more tightly.")
    print("  - Felzenszwalb: lower felz_scale and min_size for finer leaf lesion boundaries.")
    print("  - Watershed: increase marker density with num_segments for more edge-aware regions.")
    print("  - Context dilation adds neighboring pixels around each superpixel for disease-aware features.")
    print("  - Deep crops use masked region context, so neighboring lesion boundaries contribute to embeddings.")
    print("  - Graph refinement runs a CRF-style label smoothing step over the superpixel graph.")


def main() -> None:
    config = PipelineConfig()
    print_recommendations(config)
    summary = process_dataset(config)
    print("")
    print("Completed dataset processing")
    print(summary.head())
    print(f"Summary saved to: {Path(config.output_base) / 'processing_summary.csv'}")


if __name__ == "__main__":
    main()
