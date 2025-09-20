# version: 0.6.0  (SAM ViT-H + AMG пресеты + скоринг по вытянутости/площади/стабильности + ROI + антиблики)

# Требования:
#   pip install opencv-python numpy tqdm
#   pip install git+https://github.com/facebookresearch/segment-anything.git
#   (PyTorch + CUDA при наличии GPU)

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any, List
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ===================== КОНФИГ =====================
INPUT_FOLDER  = r"E:\data\004_640_eng"     # входные изображения
OUTPUT_FOLDER = r"E:\data\005_masks_5"   # бинарные маски .png (0/255)
PNG_ONLY = True
SAVE_OVERLAY = False
OVERLAY_ALPHA = 0.45

# Модель SAM
MODEL_TYPE = "vit_h"   # "vit_h" | "vit_l" | "vit_b"
SAM_CHECKPOINTS = {
    "vit_h": r"sam_vit_h_4b8939.pth",
    "vit_l": r"sam_vit_l_0b3195.pth",  # при наличии
    "vit_b": r"sam_vit_b_01ec64.pth",  # при наличии
}

# ROI: по вашим данным по краям ~128 px подложка
FIXED_PADDING_HINT = 128  # 0 — отключить

# Антиблики (металл)
USE_SPECULAR_SUPPRESSION = True
HSV_S_MAX = 0.25
HSV_V_MIN = 0.82
INPAINT_RADIUS = 3

# Жёсткие отсеки по площади маски в ROI
MIN_AREA_FRAC_ROI = 0.002         # >= 0.4% площади ROI
MAX_AREA_FRAC_ROI = 0.96         # <= 92% ROI

# Целевая площадь (для длинного узкого объекта) и допуск
TARGET_AREA_FRAC = 0.18           # целевая доля площади
AREA_SIGMA = 0.12                 # ширина «колокола»

# Минимальная вытянутость
ELONGATION_MIN_RATIO = 2.2        # min max(w,h)/min(w,h)

# Весовые коэффициенты скоринга
W_RATIO = 2.7
W_STAB  = 1.0
W_AREA  = 2.2
PENALTY_BORDER = 0.15             # штраф за касание каждой стороны ROI

# Постобработка
MORPH_KERNEL = 3                  # 0/1/3/5... (0 — выкл)
KEEP_ONLY_LARGEST = True

# Пресеты AMG
AMG_PRESETS: Dict[str, Dict[str, Any]] = {
    "quality": dict(points_per_side=48, pred_iou_thresh=0.7, stability_score_thresh=0.88,
                    crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100),
    "balanced": dict(points_per_side=32, pred_iou_thresh=0.7, stability_score_thresh=0.86,
                     crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100),
    "speed": dict(points_per_side=24, pred_iou_thresh=0.68, stability_score_thresh=0.84,
                  crop_n_layers=0, crop_n_points_downscale_factor=2, min_mask_region_area=120),
}
AMG_TRY_ORDER = ["quality", "balanced", "speed"]
# ===================================================


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def list_images(root, png_only=True):
    exts = ["*.png"] if png_only else ["*.png","*.jpg","*.jpeg","*.JPG","*.JPEG","*.PNG"]
    files = []
    for e in exts:
        files += list(Path(root).rglob(e))
    return sorted(files)


def get_roi_by_fixed_padding(img, pad):
    h, w = img.shape[:2]
    if pad <= 0:
        return (0, 0, w, h), img
    p = min(pad, w//3, h//3)
    x0, y0 = p, p
    x1, y1 = w - p, h - p
    x0 = max(0, min(x0, w-2)); x1 = max(x0+2, min(x1, w))
    y0 = max(0, min(y0, h-2)); y1 = max(y0+2, min(y1, h))
    return (x0, y0, x1, y1), img[y0:y1, x0:x1].copy()


def suppress_specular(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_f = s.astype(np.float32)/255.0
    v_f = v.astype(np.float32)/255.0
    spec = (s_f < HSV_S_MAX) & (v_f > HSV_V_MIN)
    spec_u8 = (spec.astype(np.uint8) * 255)
    if spec_u8.mean() < 1:
        return bgr
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    spec_u8 = cv2.dilate(spec_u8, k, iterations=1)
    return cv2.inpaint(bgr, spec_u8, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def mask_metrics(seg):
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return 0, 0, 0, 0.0, 0.0, (0,0,0,0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    area = int(seg.sum())
    # периметр
    cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = 0.0
    if cnts:
        perim = cv2.arcLength(cnts[0], True)
    ratio = (max(w, h) + 1e-6) / (min(w, h) + 1e-6)
    compactness = area / (perim + 1e-6)
    return area, w, h, ratio, compactness, (x0, y0, x1, y1)


def border_touch_penalty(bbox, roi_w, roi_h, margin=2):
    x0, y0, x1, y1 = bbox
    pen = 0.0
    if x0 <= margin: pen += PENALTY_BORDER
    if y0 <= margin: pen += PENALTY_BORDER
    if x1 >= roi_w - 1 - margin: pen += PENALTY_BORDER
    if y1 >= roi_h - 1 - margin: pen += PENALTY_BORDER
    return pen


def area_gaussian(frac, mu, sigma):
    # нормированная «оценка близости к целевой площади»: максимум 1.0 при frac=mu
    z = (frac - mu) / (sigma + 1e-6)
    return float(np.exp(-0.5 * z*z))


def postprocess_mask(mask_u8):
    if KEEP_ONLY_LARGEST:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            tmp = np.zeros_like(mask_u8)
            tmp[labels == idx] = 255
            mask_u8 = tmp
    if MORPH_KERNEL and MORPH_KERNEL >= 3 and MORPH_KERNEL % 2 == 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=1)
    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    return mask_u8


def overlay_mask(bgr, mask_u8, a=OVERLAY_ALPHA):
    color = np.zeros_like(bgr); color[:, :, 2] = 255
    m = mask_u8 > 0
    out = bgr.copy()
    out[m] = cv2.addWeighted(bgr[m], 1-a, color[m], a, 0)
    return out


class AMGEnsembler:
    def __init__(self, model_type: str, checkpoints: Dict[str, str], presets: Dict[str, Dict[str, Any]], try_order: List[str]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = checkpoints.get(model_type)
        if not ckpt or not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Не найден checkpoint для {model_type}: {ckpt}")
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to(device=self.device)
        self.generators = []
        for name in try_order:
            cfg = presets[name]
            self.generators.append((name, SamAutomaticMaskGenerator(sam, **cfg)))

    def generate_candidates(self, img_rgb):
        all_masks = []
        for name, gen in self.generators:
            masks = gen.generate(img_rgb)
            for m in masks:
                m["_preset"] = name
            all_masks.extend(masks)
        return all_masks


def select_best_mask(masks, roi_shape):
    H, W = roi_shape
    roi_area = H * W
    best = None
    best_score = -1e9
    for m in masks:
        seg = m["segmentation"]
        area, w, h, ratio, compact, bbox = mask_metrics(seg)
        if area == 0:
            continue
        frac = area / roi_area
        if not (MIN_AREA_FRAC_ROI <= frac <= MAX_AREA_FRAC_ROI):
            continue
        if ratio < ELONGATION_MIN_RATIO:
            continue

        # компоненты скоринга
        s_ratio = ratio                           # больше — лучше
        s_area  = area_gaussian(frac, TARGET_AREA_FRAC, AREA_SIGMA)  # ~[0..1]
        s_stab  = float(m.get("stability_score", 0.0))
        s_pen   = border_touch_penalty(bbox, W, H)                   # штраф

        score = W_RATIO*s_ratio + W_AREA*s_area + W_STAB*s_stab - s_pen

        if score > best_score:
            best_score = score
            best = seg
    return best


def process_image(ens: AMGEnsembler, in_path: Path, out_path: Path):
    bgr = cv2.imread(str(in_path))
    if bgr is None:
        return False

    # 1) ROI
    (x0, y0, x1, y1), roi = get_roi_by_fixed_padding(bgr, FIXED_PADDING_HINT)

    # 2) Антиблики
    roi_proc = suppress_specular(roi) if USE_SPECULAR_SUPPRESSION else roi

    # 3) Кандидаты SAM (все пресеты)
    img_rgb = cv2.cvtColor(roi_proc, cv2.COLOR_BGR2RGB)
    masks = ens.generate_candidates(img_rgb)

    # 4) Выбор лучшей маски по вытянутости/площади/стабильности
    best_seg = select_best_mask(masks, roi.shape[:2])

    # 5) Фолбэк: просто самая вытянутая из всех
    if best_seg is None and masks:
        best, best_ratio = None, -1
        for m in masks:
            seg = m["segmentation"]
            area, w, h, ratio, compact, bbox = mask_metrics(seg)
            if area == 0:
                continue
            if ratio > best_ratio:
                best_ratio = ratio
                best = seg
        best_seg = best

    # 6) Пусто — запишем ноль
    h, w = bgr.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)
    ensure_dir(out_path)
    if best_seg is None or best_seg.sum() == 0:
        cv2.imwrite(str(out_path), full)
        return False

    # 7) Постобработка в ROI и сборка в полный кадр
    mask_roi = (best_seg.astype(np.uint8))*255
    mask_roi = postprocess_mask(mask_roi)
    full[y0:y1, x0:x1] = mask_roi

    cv2.imwrite(str(out_path), full)
    if SAVE_OVERLAY:
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_overlay.jpg")), overlay_mask(bgr, full))
    return True


def main():
    files = list_images(INPUT_FOLDER, png_only=PNG_ONLY)
    if not files:
        print("Нет изображений для обработки.")
        return

    print(f"Файлов: {len(files)}; устройство: {'cuda' if torch.cuda.is_available() else 'cpu'}; модель: {MODEL_TYPE}")
    ens = AMGEnsembler(MODEL_TYPE, SAM_CHECKPOINTS, AMG_PRESETS, AMG_TRY_ORDER)

    ok, bad = 0, 0
    for p in tqdm(files, desc=f"SAM-{MODEL_TYPE} (elongation+area+stability)"):
        rel = p.relative_to(INPUT_FOLDER)
        out = Path(OUTPUT_FOLDER) / rel
        out = out.with_suffix(".png")
        try:
            if process_image(ens, p, out):
                ok += 1
            else:
                bad += 1
        except Exception:
            bad += 1
    print(f"Готово. Успешно: {ok}; не удалось/пустых: {bad}")
    print(f"Маски: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
