from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import lanms
import numpy as np
import torch
from loguru import logger
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def _round_by_factor(number: int, factor: int) -> int:
    """Return the closest integer to `number` that is divisible by `factor`."""
    return int(round(number / factor) * factor)


def _ceil_by_factor(number: float, factor: int) -> int:
    """Return the smallest integer >= `number` that is divisible by `factor`."""
    return int(math.ceil(number / factor) * factor)


def _floor_by_factor(number: float, factor: int) -> int:
    """Return the largest integer <= `number` that is divisible by `factor`."""
    return int(math.floor(number / factor) * factor)


# Target pixel range for smart resize: higher floor = clearer small images (e.g. ~200 DPI equivalent).
SMART_RESIZE_MIN_PIXELS: int = 1_000_000   # ~1 MP floor: upscale small docs for clearer text
SMART_RESIZE_MAX_PIXELS: int = 3_200_000  # cap to control memory/speed while keeping detail


def _smart_resize_dims(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 640_000,
    max_pixels: int = 2_822_400,
) -> Tuple[int, int]:
    """
    Compute target (height, width) so that:

    1. Both dimensions are divisible by `factor` (EAST requires multiples of 32).
    2. Total pixels are within [min_pixels, max_pixels].
    3. Aspect ratio is preserved as much as possible.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: height={height}, width={width}")

    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 200:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than 200, got {aspect_ratio}"
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, _floor_by_factor(h_bar / beta, factor))
            w_bar = max(factor, _floor_by_factor(w_bar / beta, factor))

    return int(h_bar), int(w_bar)


def resize_img(img: np.ndarray, max_length: int = 1600) -> Tuple[np.ndarray, float, float]:
    """
    Resize image using a smart, DPI-friendly strategy:

    - Output height/width are divisible by 32 for EAST.
    - Total pixels are kept within [SMART_RESIZE_MIN_PIXELS, SMART_RESIZE_MAX_PIXELS],
      upscaling small images for clearer text and downscaling very large ones.
    - Uses Lanczos when upscaling (sharper result), INTER_AREA when downscaling.
    """
    h, w, _ = img.shape
    resize_h, resize_w = _smart_resize_dims(
        h, w, factor=32,
        min_pixels=SMART_RESIZE_MIN_PIXELS,
        max_pixels=SMART_RESIZE_MAX_PIXELS,
    )

    # Sharper upscaling for low-res inputs; area-based downscaling for large images.
    is_upscale = (resize_w * resize_h) > (w * h)
    interp = cv2.INTER_LANCZOS4 if is_upscale else cv2.INTER_AREA
    img_resized = cv2.resize(img, (resize_w, resize_h), interpolation=interp)

    ratio_h = resize_h / h
    ratio_w = resize_w / w
    return img_resized, ratio_h, ratio_w


def _get_rotate_mat(theta: float) -> np.ndarray:
    """Positive theta means rotate clockwise."""
    return np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )


def _is_valid_poly(res: np.ndarray, score_shape: Tuple[int, int], scale: int) -> bool:
    """Check if polygon lies mostly inside the image."""
    cnt = 0
    for i in range(res.shape[1]):
        if (
            res[0, i] < 0
            or res[0, i] >= score_shape[1] * scale
            or res[1, i] < 0
            or res[1, i] >= score_shape[0] * scale
        ):
            cnt += 1
    return cnt <= 1


def _restore_polys(
    valid_pos: np.ndarray,
    valid_geo: np.ndarray,
    score_shape: Tuple[int, int],
    scale: int = 4,
) -> Tuple[np.ndarray, List[int]]:
    """Restore rotated quadrilaterals from feature maps."""
    polys: List[List[float]] = []
    index: List[int] = []
    valid_pos = valid_pos * scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = _get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if _is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append(
                [
                    res[0, 0],
                    res[1, 0],
                    res[0, 1],
                    res[1, 1],
                    res[0, 2],
                    res[1, 2],
                    res[0, 3],
                    res[1, 3],
                ]
            )
    return np.array(polys), index


def get_boxes(
    score: np.ndarray,
    geo: np.ndarray,
    score_thresh: float = 0.9,
    nms_thresh: float = 0.2,
) -> np.ndarray | None:
    """Get text boxes from score and geometry maps."""
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, [row, col]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = _restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype("float32"), nms_thresh)
    return boxes


def adjust_ratio(boxes: np.ndarray | None, ratio_w: float, ratio_h: float) -> np.ndarray | None:
    """Scale boxes back to original image size."""
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def get_bounding_rect_coords(
    boxes: np.ndarray | None,
) -> Tuple[List[List[Dict[str, float]]], List[List[int]]]:
    """Return (points, [x_min, x_max, y_min, y_max]) for each detected box."""
    if boxes is None or boxes.size == 0:
        return [], []

    points: List[List[Dict[str, float]]] = []
    bboxes: List[List[int]] = []

    for i in range(len(boxes)):
        x1, y1, x2, y2, x3, y3, x4, y4, _ = boxes[i, :]
        x_min = min(x1, x2, x3, x4)
        x_max = max(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)

        point = [
            {"x": float(x_min), "y": float(y_min)},
            {"x": float(x_min), "y": float(y_max)},
            {"x": float(x_max), "y": float(y_min)},
            {"x": float(x_max), "y": float(y_max)},
        ]

        points.append(point)
        bboxes.append([int(x_min), int(x_max), int(y_min), int(y_max)])

    return points, bboxes


_VGG_CFG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]


def _make_layers(cfg: List[int | str], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, int(v), kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = int(v)
    return nn.Sequential(*layers)


class _Extractor(nn.Module):
    """VGG-style feature extractor used inside EAST."""

    def __init__(self) -> None:
        super().__init__()
        vgg = _make_layers(_VGG_CFG, batch_norm=True)
        self.features = vgg

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)
        return out[1:]


class _Merge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        y = F.interpolate(x[3], scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        y = self.relu7(self.bn7(self.conv7(y)))
        return y


class _OutputHead(nn.Module):
    def __init__(self, scope: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = scope

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo = torch.cat((loc, angle), 1)
        return score, geo


class EAST(nn.Module):
    """EAST text detector backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.extractor = _Extractor()
        self.merge = _Merge()
        self.output = _OutputHead()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.output(self.merge(self.extractor(x)))


class EASTSegmentation:
    """Wrap EAST model and provide `detect_words` API."""

    def __init__(self, model_path: str, device: torch.device, num_gpus: int = 0) -> None:
        self.device = device
        base_model = EAST().to(self.device)
        # Load weights into the base (non-DataParallel) model first
        state = torch.load(model_path, map_location=self.device)
        base_model.load_state_dict(state)
        # Then optionally wrap with DataParallel for multi-GPU inference
        if num_gpus > 1 and self.device.type == "cuda":
            base_model = nn.DataParallel(base_model)
        self.model = base_model.eval()

    def detect_words(
        self,
        img_bgr: np.ndarray,
        save_preprocessed_path: Optional[Path] = None,
    ) -> Tuple[List[List[Dict[str, float]]], List[List[int]]]:
        h_in, w_in = img_bgr.shape[:2]
        logger.debug("detection start | input_shape=({}, {})", h_in, w_in)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized, ratio_h, ratio_w = resize_img(img_rgb)
        h_rsz, w_rsz = img_resized.shape[:2]
        logger.info(
            "detection resize | input=({}, {}), resized=({}, {}), ratio_h={:.4f}, ratio_w={:.4f}",
            h_in, w_in, h_rsz, w_rsz, ratio_h, ratio_w,
        )

        if save_preprocessed_path is not None:
            try:
                bgr_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_preprocessed_path), bgr_resized)
                logger.info("detection preprocessed saved | path={}", save_preprocessed_path)
            except Exception as e:  # pragma: no cover - best-effort save
                logger.warning("detection preprocessed save failed | path={} error={}", save_preprocessed_path, e)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        inp = transform(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score, geo = self.model(inp)

        score_np = score.squeeze(0).cpu().numpy()
        geo_np = geo.squeeze(0).cpu().numpy()

        boxes = get_boxes(score_np, geo_np)
        n_boxes_raw = len(boxes) if boxes is not None else 0
        logger.debug("detection get_boxes | raw_boxes={}", n_boxes_raw)

        boxes = adjust_ratio(boxes, ratio_w, ratio_h)
        points, bboxes = get_bounding_rect_coords(boxes)
        logger.info(
            "detection done | points={}, rois={}",
            len(points), len(bboxes),
        )
        return points, bboxes

