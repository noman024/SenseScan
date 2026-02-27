from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import quantization
from torch.nn.intrinsic import qat


class VGGFeatureExtractor(nn.Module):
    """Feature extractor backbone architecture."""

    def __init__(self, input_channel: int = 1, output_channel: int = 512) -> None:
        super().__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                self.output_channel[1],
                self.output_channel[2],
                3,
                1,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.output_channel[2]),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, (2, 1)),
            nn.Conv2d(
                self.output_channel[2],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[3]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ConvNet(x)


class VGGQuantizedFeatureExtractor(nn.Module):
    """Quantized feature extractor backbone architecture (for QAT checkpoints)."""

    def __init__(
        self, input_channel: int = 1, output_channel: int = 512, qatconfig: str = "fbgemm"
    ) -> None:
        super().__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]  # [64, 128, 256, 512]

        self.qconfig = quantization.get_default_qat_qconfig(qatconfig)

        self.ConvNet = nn.Sequential(
            qat.ConvReLU2d(
                input_channel,
                self.output_channel[0],
                3,
                1,
                1,
                qconfig=self.qconfig,
            ),
            nn.MaxPool2d(2, 2),
            qat.ConvReLU2d(
                self.output_channel[0],
                self.output_channel[1],
                3,
                1,
                1,
                qconfig=self.qconfig,
            ),
            nn.MaxPool2d(2, 2),
            qat.ConvBnReLU2d(
                self.output_channel[1],
                self.output_channel[2],
                3,
                1,
                1,
                qconfig=self.qconfig,
            ),
            qat.ConvReLU2d(
                self.output_channel[2],
                self.output_channel[2],
                3,
                1,
                1,
                qconfig=self.qconfig,
            ),
            nn.MaxPool2d(2, (2, 1)),
            qat.ConvBnReLU2d(
                self.output_channel[2],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False,
                qconfig=self.qconfig,
            ),
            qat.ConvReLU2d(
                self.output_channel[3],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False,
                qconfig=self.qconfig,
            ),
            nn.MaxPool2d(2, (2, 1)),
            qat.ConvBnReLU2d(
                self.output_channel[3],
                self.output_channel[3],
                2,
                1,
                0,
                qconfig=self.qconfig,
            ),
        )

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.ConvNet(x)
        x = self.dequant(x)
        return x


class BidirectionalLSTM(nn.Module):
    """Sequence predictor."""

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, bidirectional=True, num_layers=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # (W, B, C)
        x, _ = self.lstm(x)
        return x


class CRNN(nn.Module):
    """CRNN architecture compatible with QAT checkpoints."""

    def __init__(self, nclass: int, hidden_size: int = 256, qatconfig: str | None = None):
        super().__init__()

        if qatconfig is None:
            self.backbone = VGGFeatureExtractor()
        else:
            self.backbone = VGGQuantizedFeatureExtractor(qatconfig=qatconfig)
            self.backbone = quantization.prepare_qat(self.backbone)

        self.rnn = BidirectionalLSTM(512, hidden_size)
        self.embedding = nn.Linear(hidden_size * 2, nclass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.backbone(x)
        x = self.rnn(x)
        t, b, h = x.size()
        x = x.view(t * b, h)
        x = self.embedding(x)
        x = x.view(t, b, -1)

        x = nn.functional.log_softmax(x, dim=2)
        _, x = x.max(2)
        x = x.transpose(1, 0)

        return x


def get_crnn(n_class: int, qatconfig: str | None = None) -> CRNN:
    """Return CRNN model; pass qatconfig (e.g. 'fbgemm') for QAT checkpoints."""
    return CRNN(n_class, qatconfig=qatconfig)


# Handwritten grapheme dictionary for HW model (inlined from original project)
inv_grapheme_dict_hw = {
    1: "<pad>",
    2: "",
    3: "ভ",
    4: "ৌ",
    5: "ত",
    6: ",",
    7: "র",
    8: "া",
    9: "স",
    10: "য়",
    11: "ন",
    12: "ি",
    13: "প্ট",
    14: "র্",
    15: "ধ",
    16: "\u200d",
    17: "্য",
    18: "প",
    19: "্র",
    20: "ণ",
    21: "ম",
    22: "য",
    23: "ক",
    24: "ষ",
    25: "ল্ল",
    26: "ী",
    27: "শ্ম",
    28: "ে",
    29: "শ্ব",
    30: "এ",
    31: "ঁ",
    32: "ই",
    33: "ো",
    34: "ট",
    35: "ড",
    36: "।",
    37: "ু",
    38: "ব",
    39: "জ",
    40: "হ",
    41: "ল",
    42: "ল্ট",
    43: "ণ্ট",
    44: "ষ্ট",
    45: "ষ্প",
    46: "ম্প",
    47: "ং",
    48: "গ",
    49: "প্ল",
    50: "ল্প",
    51: "ধ্ব",
    52: "ন্দ",
    53: "দ",
    54: "৭",
    55: "০",
    56: "-",
    57: "১",
    58: "৪",
    59: "ঋ",
    60: "ফ",
    61: "ষ্ণ",
    62: "ন্ব",
    63: "ষ্ক",
    64: "স্ট",
    65: "স্প",
    66: "ন্ধ",
    67: "ড়",
    68: "৩",
    69: "৫",
    70: "৯",
    71: "ল্ফ",
    72: "শ",
    73: "জ্ব",
    74: "৮",
    75: "ফ্ল",
    76: "দ্ম",
    77: "অ",
    78: "ঞ্জ",
    79: "ম্ন",
    80: "চ্চ",
    81: "ব্দ",
    82: "ৃ",
    83: "গ্ব",
    84: "প্ত",
    85: "আ",
    86: "ক্ষ",
    87: "ূ",
    88: "ন্ন",
    89: "চ",
    90: "চ্ছ",
    91: "স্থ",
    92: "ব্ব",
    93: "খ",
    94: "ঞ্চ",
    95: "ত্ন",
    96: "থ",
    97: "ঙ্গ",
    98: "ত্ম",
    99: "ঙ্ক্ষ",
    100: "স্ত",
    101: "ত্ত",
    102: "ও",
    103: "স্ক",
    104: "ক্ত",
    105: "ন্ম",
    106: "ঘ",
    107: "জ্জ",
    108: "গ্ন",
    109: "ক্ট",
    110: "ষ্ঠ",
    111: "ড্ড",
    112: "শ্চ",
    113: "স্ম",
    114: "দ্ধ",
    115: "ঙ্খ",
    116: "ঝ",
    117: "ঠ",
    118: "ন্ত্ব",
    119: "ক্ক",
    120: "প্ন",
    121: "৬",
    122: "২",
    123: "ন্ড",
    124: "উ",
    125: "ন্থ",
    126: "ত্থ",
    127: "জ্ঞ",
    128: "ছ",
    129: "ষ্ফ",
    130: "ন্দ্ব",
    131: "হ্ন",
    132: "ঙ",
    133: "শ্ন",
    134: "ৈ",
    135: "দ্ব",
    136: "ম্ভ",
    137: "শ্ল",
    138: "ৎ",
    139: "ন্স",
    140: "ক্ষ্ম",
    141: "ন্ত",
    142: "ণ্ড",
    143: "স্ব",
    144: "ণ্ণ",
    145: "ঙ্ঘ",
    146: "ব্ল",
    147: "ঙ্ক",
    148: "ত্ব",
    149: "ট্ট",
    150: "স্ন",
    151: "ঔ",
    152: "দ্দ",
    153: "ব্ধ",
    154: "ম্ব",
    155: "ণ্ঠ",
    156: "ম্ম",
    157: "ষ্ম",
    158: "ঃ",
    159: "ঊ",
    160: "ক্ল",
    161: "হ্ব",
    162: "প্প",
    163: "ঢ়",
    164: "ল্গ",
    165: "স্ফ",
    166: "ঢ",
    167: "ব্জ",
    168: "ক্স",
    169: "ল্ক",
    170: "দ্ভ",
    171: "ন্ট",
    172: "ল্ড",
    173: "ন্ঠ",
    174: "ক্ষ্ণ",
    175: "ল্ম",
    176: "্",
    177: "গ্ল",
    178: "দ্ঘ",
    179: "ন্জ",
    180: "ম্ল",
    181: "ত্ত্ব",
    182: "হ্ল",
    183: "ঘ্ন",
    184: "চ্ছ্ব",
    185: "ঞ",
    186: "ঐ",
    187: "ঈ",
    188: "গ্ধ",
    189: "প্স",
    190: "ফ্ট",
    191: "ল্ব",
    192: "জ্জ্ব",
    193: "স্ল",
    194: "হ্ম",
    195: "ঞ্ছ",
    196: "গ্ম",
    197: "ম্ফ",
    198: "শ্ছ",
    199: "ক্ব",
    200: "থ্ব",
    201: "স্খ",
    202: "স্প্ল",
    203: "হ্ণ",
    204: "ঞ্ঝ",
    205: "দ্গ",
    206: "চ্ঞ",
    207: "\u200c",
    208: "ঙ্ম",
    209: "ট্ব",
    210: "ধ্ন",
    211: "ল্স",
    212: "ক্ম",
    213: "জ্ঝ",
    214: "'",
    215: "ণ্ব",
    216: "স্ত্ব",
    217: "ক্ন",
    218: "ণ্ঢ",
    219: "ণ্ম",
    220: "দ্দ্ব",
    221: "ষ্ব",
    222: "?",
    223: ";",
    224: "(",
    225: ")",
    226: '"',
    227: ".",
    228: "!",
    229: "+",
    230: "=",
    231: "^",
    232: "|",
    233: "[",
    234: "\\",
    235: ":",
    236: "%",
    237: "‘",
    238: "’",
    239: "&",
    240: "@",
    241: "#",
    242: "/",
    243: "“",
    244: "”",
    245: "·",
    246: "–",
    247: "•",
    248: "*",
    249: "_",
}


class HandwrittenWordRecognizer:
    """Word recognizer for handwritten Bangla using CRNN."""

    def __init__(self, model_path: str, device: torch.device, num_gpus: int = 0) -> None:
        self.inv_grapheme_dict = inv_grapheme_dict_hw
        self.device = device
        # Use QAT-compatible CRNN to match saved checkpoint
        base_model = get_crnn(len(self.inv_grapheme_dict) + 1, qatconfig="fbgemm").to(
            self.device
        )
        # Load weights into the base (non-DataParallel) model first
        state = torch.load(model_path, map_location=self.device)
        base_model.load_state_dict(state)
        # Then optionally wrap with DataParallel for multi-GPU inference
        if num_gpus > 1 and self.device.type == "cuda":
            base_model = nn.DataParallel(base_model)
        self.model = base_model.eval()

    def _decode_prediction(self, preds: np.ndarray) -> Tuple[List[int], str]:
        decoded_string: List[str] = []
        decoded_label: List[int] = []

        for idx, pred in enumerate(preds):
            if pred != 0 and pred != 1 and not (idx > 0 and preds[idx - 1] == preds[idx]):
                decoded_string.append(self.inv_grapheme_dict.get(int(pred), ""))
                decoded_label.append(int(pred))

        return decoded_label, "".join(decoded_string)

    def _preprocess_twr_hwr(
        self, main_image: np.ndarray, word_coords: List[List[int]]
    ) -> np.ndarray:
        inp_h = 32
        inp_w = 128
        imgs = np.zeros((len(word_coords), 1, inp_h, inp_w), dtype=np.float32)

        for idx, data in enumerate(word_coords):
            try:
                x_min, x_max, y_min, y_max = data
                img = main_image[y_min:y_max, x_min:x_max]
                # Skip empty or degenerate crops to avoid OpenCV errors
                if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img_h, img_w = img.shape
                img = cv2.resize(
                    img,
                    (0, 0),
                    fx=inp_w / img_w,
                    fy=inp_h / img_h,
                    interpolation=cv2.INTER_CUBIC,
                )
                img = np.reshape(img, (inp_h, inp_w, 1))
                img = img.transpose(2, 0, 1)
                imgs[idx] = img
            except Exception as e:  # pragma: no cover - logging only
                logger.exception(e)

        return imgs

    def infer(
        self, image: np.ndarray, word_coords: List[List[int]]
    ) -> List[Dict[str, object]]:
        result: List[Dict[str, object]] = []
        processed_imgs = self._preprocess_twr_hwr(image, word_coords)

        processed_imgs_t = torch.tensor(processed_imgs, dtype=torch.float32).to(
            self.device
        )
        self.model.eval()
        with torch.no_grad():
            start_idx = 0
            batch_size = 128
            total = len(processed_imgs_t)
            while start_idx < total:
                end_idx = min(start_idx + batch_size, total)
                preds = self.model(processed_imgs_t[start_idx:end_idx, ...])
                preds_np = preds.contiguous().detach().cpu().numpy()

                for local_idx, pred in enumerate(preds_np):
                    _, decoded = self._decode_prediction(pred)
                    obj = {
                        "id": start_idx + local_idx + 1,
                        "text": decoded,
                    }
                    result.append(obj)

                start_idx += batch_size

        return result

