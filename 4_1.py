# version: 0.3.1  (SAM + подавление бликов + доклейка кончика)

# Требования:
#   pip install opencv-python numpy tqdm
#   pip install git+https://github.com/facebookresearch/segment-anything.git
#   (PyTorch + CUDA при наличии GPU)

import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# ========= НАСТРОЙКИ =========
INPUT_FOLDER  = r"E:\data\004_640_eng"
OUTPUT_FOLDER = r"E:\data\005_masks_eng"

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

PNG_ONLY = True
USE_AMG_FALLBACK = True
SAVE_OVERLAY = False
OVERLAY_ALPHA = 0.45

# ROI (подложка по краям ~128px)
FIXED_PADDING_HINT = 128

# Пороговые константы
MIN_AREA_FRAC = 0.001      # отсечь мусор <0.1% ROI
MAX_AREA_FRAC = 0.98       # защита от "весь кадр"
MORPH_KERNEL  = 3          # морф. закрытие

# Параметры подавления бликов
HSV_S_MAX = 0.22           # низкая насыщенность
HSV_V_MIN = 0.85           # высокая яркость
INPAINT_RADIUS = 3         # радиус Telea для заливки бликов

# Градиентная «доклейка» кончика
GRAD_BLUR = 3              # сглаживание перед градиентом
CANNY_T1, CANNY_T2 = 50, 150
GRAD_DILATE = 2            # расширение контура
TIP_MAX_FRACTION = 0.06    # максимум 6% площади ROI добавляем "как кончик"
# ============================


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def list_images(root, png_only=True):
    exts = ["*.png"] if png_only else ["*.png","*.jpg","*.jpeg","*.JPG","*.JPEG","*.PNG"]
    files = []
    for e in exts:
        files += list(Path(root).rglob(e))
    return sorted(files)


def roi_from_padding(img, pad):
    h, w = img.shape[:2]
    l = r = t = b = min(pad, w//3, h//3)
    x0, y0 = l, t
    x1, y1 = w - r, h - b
    return (x0, y0, x1, y1), img[y0:y1, x0:x1].copy()


def suppress_specular(bgr):
    """
    Подавление бликов:
      1) Находим пиксели с низкой S и высокой V в HSV (спекулярные).
      2) Расширяем маску бликов.
      3) Inpaint (Telea) по этой маске.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_f = s.astype(np.float32) / 255.0
    v_f = v.astype(np.float32) / 255.0

    spec = (s_f < HSV_S_MAX) & (v_f > HSV_V_MIN)
    spec_u8 = (spec.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    spec_u8 = cv2.dilate(spec_u8, k, iterations=1)

    # Inpaint по бликам
    inpainted = cv2.inpaint(bgr, spec_u8, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return inpainted, spec_u8  # второе — маска бликов (для контроля)


def largest_component(mask_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_u8)
    out[labels == idx] = 255
    return out


def fill_holes(mask_u8):
    ff = mask_u8.copy()
    ff_pad = cv2.copyMakeBorder(ff, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((ff_pad.shape[0]+2, ff_pad.shape[1]+2), np.uint8)
    cv2.floodFill(ff_pad, ff_mask, (0,0), 255)
    ff_nb = ff_pad[1:-1, 1:-1]
    inv_filled_bg = cv2.bitwise_not(ff_nb)
    holes = cv2.bitwise_and(inv_filled_bg, cv2.bitwise_not(mask_u8))
    return cv2.bitwise_or(mask_u8, holes)


def postprocess(mask_bool):
    mask = (mask_bool.astype(np.uint8))*255
    mask = largest_component(mask)
    if MORPH_KERNEL and MORPH_KERNEL >= 3 and MORPH_KERNEL % 2 == 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = fill_holes(mask)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def overlay_mask(bgr, mask_u8, a=OVERLAY_ALPHA):
    color = np.zeros_like(bgr); color[:,:,2] = 255
    m = mask_u8 > 0
    out = bgr.copy()
    out[m] = cv2.addWeighted(bgr[m], 1-a, color[m], a, 0)
    return out


class SamWorker:
    def __init__(self, ckpt, model_type):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.amg = SamAutomaticMaskGenerator(sam) if USE_AMG_FALLBACK else None

    def predict(self, roi_bgr):
        H, W = roi_bgr.shape[:2]
        img_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        # Подсказки: центр = объект, углы = фон
        pos = np.array([[W/2, H/2]], dtype=np.float32)
        neg = np.array([[4,4],[W-5,4],[4,H-5],[W-5,H-5]], dtype=np.float32)
        pts = np.vstack([pos, neg])
        lbl = np.array([1,0,0,0,0], dtype=np.int32)

        self.predictor.set_image(img_rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=pts, point_labels=lbl, box=None, multimask_output=True
        )
        best = masks[int(np.argmax(scores))].astype(bool)
        frac = best.mean()

        if (frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC) and self.amg is not None:
            amg = self.amg.generate(img_rgb)
            if amg:
                # выбираем маску с большой площадью, но не «всё изображение»
                HxW = H*W
                cand = [m for m in amg if 0.002*HxW < m["area"] < 0.95*HxW]
                if not cand:
                    cand = amg
                best = max(cand, key=lambda m: m["area"])["segmentation"].astype(bool)

        return best


def tip_patch_by_gradient(roi_bgr, base_mask_u8):
    """
    Добавка «кончика» по градиенту (для бликов):
      1) Canny на сглаженном ROI
      2) дилатация → бинарная область «тонкого объекта»
      3) из неё забираем компоненты, прилегающие к уже найденной маске
      4) ограничиваем верхний предел прироста площади
    """
    H, W = roi_bgr.shape[:2]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if GRAD_BLUR > 0:
        gray = cv2.GaussianBlur(gray, (GRAD_BLUR|1, GRAD_BLUR|1), 0)

    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2)
    if GRAD_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (GRAD_DILATE, GRAD_DILATE))
        edges = cv2.dilate(edges, k, iterations=1)

    # связные компоненты «тонких» областей
    n, labels, stats, _ = cv2.connectedComponentsWithStats((edges>0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return base_mask_u8

    base = base_mask_u8 > 0
    grow = np.zeros_like(base, dtype=np.uint8)
    hxw = H*W
    add_limit = int(TIP_MAX_FRACTION * hxw)

    added = 0
    for idx in range(1, n):
        comp = (labels == idx)
        # комп должен касаться базовой маски (сцепка)
        touch = cv2.dilate(base.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
        if (comp & (touch>0)).any():
            area = int(comp.sum())
            if added + area <= add_limit:
                grow[comp] = 255
                added += area

    combined = cv2.bitwise_or(base_mask_u8, grow)
    return combined


def process_one(worker, in_path: Path, out_path: Path):
    img = cv2.imread(str(in_path))
    if img is None:
        return False

    (x0, y0, x1, y1), roi = roi_from_padding(img, FIXED_PADDING_HINT)

    # 1) Подавляем блики в ROI
    roi_clean, spec_mask = suppress_specular(roi)

    # 2) SAM по очищенному ROI
    mask_bool = worker.predict(roi_clean)

    # 3) Постобработка + sanity
    mask_u8 = postprocess(mask_bool)
    frac = mask_u8.mean() / 255.0
    if frac > MAX_AREA_FRAC:  # безопасность
        mask_u8[:] = 0

    # 4) «Доклейка» кончика по градиенту (если кончик потерян)
    # Критерий: есть яркие блики рядом с контуром и площадь маски маленькая для «отвёртки»
    if spec_mask.mean() > 2 and frac < 0.25:  # эвристика, при необходимости подстройте
        mask_u8 = tip_patch_by_gradient(roi, mask_u8)
        mask_u8 = largest_component(mask_u8)  # подчистка хвостов

    # 5) Сборка в полный размер
    full = np.zeros(img.shape[:2], dtype=np.uint8)
    full[y0:y1, x0:x1] = mask_u8

    ensure_dir(out_path)
    cv2.imwrite(str(out_path), full)
    if SAVE_OVERLAY:
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_overlay.jpg")), overlay_mask(img, full))
    return True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = list_images(INPUT_FOLDER, png_only=PNG_ONLY)
    if not files:
        print("Нет изображений.")
        return

    print(f"Файлов: {len(files)}; устройство: {device}")
    worker = SamWorker(SAM_CHECKPOINT, MODEL_TYPE)

    ok, bad = 0, 0
    for p in tqdm(files, desc="Segm+SpecularFix"):
        rel = p.relative_to(INPUT_FOLDER)
        out = Path(OUTPUT_FOLDER) / rel
        out = out.with_suffix(".png")
        try:
            if process_one(worker, p, out):
                ok += 1
            else:
                bad += 1
        except Exception:
            bad += 1
    print(f"Готово. Успешно: {ok}; ошибок: {bad}")
    print(f"Маски: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
