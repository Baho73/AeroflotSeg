# u2net_infer.py
# Полная инференс-утилита: U²-Net (saliency) -> бинарные маски + постобработка + (опц.) GrabCut-доочистка

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ================= НАСТРОЙКИ =================
INPUT_FOLDER  = r"E:\data\004_640_eng"
OUTPUT_FOLDER = r"E:\data\005_masks_6"

U2NET_REPO_DIR = r"D:\Python\Aeroflot\U-2-Net"   # <- путь к склонированному репо (см. инструкции выше)
U2NET_WEIGHTS  = os.path.join(U2NET_REPO_DIR, "saved_models", "u2net", "u2net.pth")

PNG_ONLY = True                   # True: только *.png; False: jpg/jpeg/png
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Порог бинаризации saliency:
THRESH = 0.35                     # 0.25..0.5 — типичные значения

# Постобработка маски:
MORPH_KERNEL = 3                  # 0/1/3/5..., 0 — отключить
KEEP_ONLY_LARGEST = True

# (Опционально) лёгкая доочистка GrabCut поверх saliency-маски
USE_GRABCUT_REFINE = False        # включить при сложном фоне
GRABCUT_ITERS = 3
# ============================================


# --- импорт архитектуры U²-Net из репозитория ---
sys.path.append(U2NET_REPO_DIR)
from model.u2net import U2NET   # repo: xuebinqin/U-2-Net (salient object detection)  # noqa

def list_images(root, png_only=True):
    exts = ["*.png"] if png_only else ["*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG"]
    files = []
    for e in exts:
        files += list(Path(root).rglob(e))
    return sorted(files)

def load_model(weights_path):
    net = U2NET(3, 1)
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state)
    net.to(DEVICE)
    net.eval()
    return net

def _norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi + 1e-8)

def u2net_saliency(net, bgr):
    # вход: BGR uint8 -> RGB float32 [0,1], размер как есть
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # U²-Net обучен на 320x320; подаём ресайз, потом рескейлим назад
    inp = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    tens = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0)  # [1,3,320,320]
    tens = tens.to(DEVICE)

    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(tens)
        pred = _norm_pred(d1)   # финальный выход (repo README)  # noqa

    pred = pred.squeeze().cpu().numpy()  # [320,320], 0..1
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
    return pred  # float mask 0..1

def postprocess(mask_u8):
    if KEEP_ONLY_LARGEST:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = np.zeros_like(mask_u8)
            m[labels == idx] = 255
            mask_u8 = m
    if MORPH_KERNEL and MORPH_KERNEL >= 3 and MORPH_KERNEL % 2 == 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=1)
    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    return mask_u8

def refine_grabcut(bgr, init_mask_u8, iters=3):
    # init_mask_u8: 0/255. Сформируем метки GC: вероятный фон/объект
    h, w = init_mask_u8.shape
    gc_mask = np.zeros((h, w), np.uint8)
    # уверенный фон по низкой вероятности, уверенный объект по высокой
    gc_mask[init_mask_u8 < 10]  = cv2.GC_BGD
    gc_mask[init_mask_u8 > 245] = cv2.GC_FGD
    # остальное — вероятное
    unsure = (gc_mask != cv2.GC_BGD) & (gc_mask != cv2.GC_FGD)
    gc_mask[unsure] = cv2.GC_PR_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)  # OpenCV GrabCut  # noqa

    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out

def main():
    assert os.path.isfile(U2NET_WEIGHTS), f"Нет весов: {U2NET_WEIGHTS}\nСкачать u2net.pth из README репозитория (Google Drive/Baidu)."
    net = load_model(U2NET_WEIGHTS)

    files = list_images(INPUT_FOLDER, png_only=PNG_ONLY)
    if not files:
        print("Файлов не найдено.")
        return

    ok = bad = 0
    for p in tqdm(files, desc=f"U²-Net ({'cuda' if DEVICE=='cuda' else 'cpu'})"):
        rel = Path(p).relative_to(INPUT_FOLDER)
        out_path = Path(OUTPUT_FOLDER) / rel
        out_path = out_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            bgr = cv2.imread(str(p))
            if bgr is None:
                bad += 1
                # записываем нулевую маску, чтобы счёт совпадал
                cv2.imwrite(str(out_path), np.zeros((512,512), np.uint8))
                continue

            sal = u2net_saliency(net, bgr)           # float 0..1
            mask = (sal >= THRESH).astype(np.uint8) * 255

            mask = postprocess(mask)

            if USE_GRABCUT_REFINE:
                mask = refine_grabcut(bgr, mask, iters=GRABCUT_ITERS)
                mask = postprocess(mask)

            cv2.imwrite(str(out_path), mask)
            ok += 1
        except Exception:
            # при любой ошибке пишем «ноль», чтобы количество файлов совпадало
            h, w = (bgr.shape[:2] if 'bgr' in locals() and bgr is not None else (512, 512))
            cv2.imwrite(str(out_path), np.zeros((h, w), np.uint8))
            bad += 1

    print(f"Готово. Успешно: {ok}; ошибок: {bad}")
    print(f"Маски: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
