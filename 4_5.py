# u2net_tipfix.py
# U²-Net (saliency) + антиблики + восстановление тонкого жала по оси (PCA) + градиентный коридор

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
OUTPUT_FOLDER = r"E:\data\005_masks_7"

U2NET_REPO_DIR = r"D:\Python\Aeroflot\U-2-Net"   # путь к репозиторию
U2NET_WEIGHTS  = os.path.join(U2NET_REPO_DIR, "saved_models", "u2net", "u2net.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PNG_ONLY = True

# Порог бинаризации saliency
THRESH = 0.32

# Антиблики до U²-Net (HSV: низкая S, высокая V → inpaint Telea)
USE_SPECULAR_SUPPRESSION = True
HSV_S_MAX = 0.28
HSV_V_MIN = 0.78
INPAINT_RADIUS = 3

# Постобработка базовой маски
KEEP_ONLY_LARGEST = True
MORPH_KERNEL = 3  # 0/1/3/5..., 0 — выкл

# Восстановление жала (corridor-grow)
CORRIDOR_WIDTH_FRAC = 0.22   # ширина коридора в долях меньшей стороны bbox маски
CORRIDOR_LEN_FRAC   = 0.45   # длина коридора в долях большей стороны bbox маски
TIP_MAX_FRAC_ROI    = 0.10   # максимум, на сколько можно увеличить маску (от площади ROI)
EDGE_CANNY_T1, EDGE_CANNY_T2 = 40, 120
DILATE_EDGE = 2              # небольшая дилатация грани

# Жёсткие отсечки по площади (от площади ROI)
MIN_AREA_FRAC = 0.004
MAX_AREA_FRAC = 0.98
# ============================================


# --- импорт модели U²-Net ---
sys.path.insert(0, U2NET_REPO_DIR)
from model.u2net import U2NET  # noqa


def list_images(root, png_only=True):
    exts = ["*.png"] if png_only else ["*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG"]
    files = []
    for e in exts:
        files += list(Path(root).rglob(e))
    return sorted(files)


def load_model(weights_path):
    net = U2NET(3, 1)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state)
    net.to(DEVICE).eval()
    torch.backends.cudnn.benchmark = True
    return net


def _norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi + 1e-8)


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


def u2net_saliency(net, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    inp = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    tens = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(tens)
        pred = _norm_pred(d1)
    pred = pred.squeeze().detach().cpu().numpy()
    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_CUBIC)
    return pred  # float 0..1


def largest_component(mask_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_u8)
    out[labels == idx] = 255
    return out


def postprocess(mask_u8):
    if KEEP_ONLY_LARGEST:
        mask_u8 = largest_component(mask_u8)
    if MORPH_KERNEL and MORPH_KERNEL >= 3 and MORPH_KERNEL % 2 == 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=1)
    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    return mask_u8


def pca_axis(binary_u8):
    """Оценивает главную ось объекта по PCA (OpenCV PCA), возвращает центр и нормированный вектор оси."""
    ys, xs = np.where(binary_u8 > 0)
    if len(xs) < 50:
        return None, None
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean, eigenvectors = cv2.PCACompute(pts, mean=np.array([]))
    center = (float(mean[0,0]), float(mean[0,1]))
    v = eigenvectors[0]  # главное собственное направление
    v = v / (np.linalg.norm(v) + 1e-9)
    return center, v  # (cx,cy), (vx,vy)


def tip_endpoints(mask_u8, center, v):
    """Проекция точек маски на ось → получаем два конца (ручка/жало)."""
    ys, xs = np.where(mask_u8 > 0)
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    # вектор оси и орт
    vx, vy = v
    proj = (pts[:,0]-center[0])*vx + (pts[:,1]-center[1])*vy  # скаляры вдоль оси
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    p_min = (float(pts[i_min,0]), float(pts[i_min,1]))
    p_max = (float(pts[i_max,0]), float(pts[i_max,1]))
    return p_min, p_max  # два конца вдоль оси


def build_corridor_mask(shape, p0, p1, extend_len, width_px):
    """
    Строит «коридор» — вытянутый прямоугольник, ориентированный вдоль оси p0->p1,
    начиная от конца p1 наружу на длину extend_len и ширину width_px.
    """
    H, W = shape
    v = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    # продлеваем от p1 в сторону v
    p_start = np.array(p1, dtype=np.float32)
    p_end   = p_start + v * extend_len

    # нормаль для ширины
    n = np.array([-v[1], v[0]], dtype=np.float32)
    half = width_px / 2.0

    poly = np.array([
        p_start + n*half,
        p_end   + n*half,
        p_end   - n*half,
        p_start - n*half
    ], dtype=np.float32)

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
    return mask


def restore_tip_by_corridor(bgr, base_mask_u8):
    H, W = base_mask_u8.shape
    if base_mask_u8.sum() == 0:
        return base_mask_u8

    # 1) PCA ось и концы
    center, v = pca_axis(base_mask_u8)
    if center is None:
        return base_mask_u8
    p_min, p_max = tip_endpoints(base_mask_u8, center, v)

    # bbox базовой маски
    ys, xs = np.where(base_mask_u8 > 0)
    w = xs.max()-xs.min()+1
    h = ys.max()-ys.min()+1
    width_px = max(6, int(min(w, h) * CORRIDOR_WIDTH_FRAC))
    ext_len  = int(max(w, h) * CORRIDOR_LEN_FRAC)

    # 2) Строим коридоры для обоих концов и собираем «кандидатов» из границ/бликов
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1].astype(np.float32)/255.0
    vimg = hsv[:,:,2].astype(np.float32)/255.0
    spec = ((s < HSV_S_MAX) & (vimg > HSV_V_MIN)).astype(np.uint8) * 255

    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),(3,3),0),
                      EDGE_CANNY_T1, EDGE_CANNY_T2)
    if DILATE_EDGE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_EDGE, DILATE_EDGE))
        edges = cv2.dilate(edges, k, 1)

    # Коридоры
    corr1 = build_corridor_mask((H, W), center, p_max, ext_len, width_px)
    corr2 = build_corridor_mask((H, W), center, p_min, ext_len, width_px)

    cand1 = cv2.bitwise_or(cv2.bitwise_and(spec, corr1), cv2.bitwise_and(edges, corr1))
    cand2 = cv2.bitwise_or(cv2.bitwise_and(spec, corr2), cv2.bitwise_and(edges, corr2))

    # 3) Оставляем только то, что сцепляется с базовой маской (после небольшой дилатации моста)
    bridge = cv2.dilate(base_mask_u8, np.ones((3,3), np.uint8), 1)
    add1 = cv2.bitwise_and(cand1, bridge)
    add2 = cv2.bitwise_and(cand2, bridge)

    # Если сцепления нет, расширим кандидата на 1-2 пиксела
    if add1.sum() == 0:
        cand1 = cv2.dilate(cand1, np.ones((3,3), np.uint8), 1)
        add1 = cv2.bitwise_and(cand1, bridge)
    if add2.sum() == 0:
        cand2 = cv2.dilate(cand2, np.ones((3,3), np.uint8), 1)
        add2 = cv2.bitwise_and(cand2, bridge)

    # 4) Собираем итоговую «добавку» и ограничиваем прирост по площади
    corr_cand = cv2.bitwise_or(cand1, cand2)
    corr_cand = cv2.morphologyEx(corr_cand, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    _, corr_cand = cv2.threshold(corr_cand, 1, 255, cv2.THRESH_BINARY)

    base_area = int((base_mask_u8 > 0).sum())
    roi_area  = H*W
    max_add   = int(TIP_MAX_FRAC_ROI * roi_area)

    # оставляем только компоненты, которые касаются base и укладываются в лимит площади
    n, labels, stats, _ = cv2.connectedComponentsWithStats((corr_cand > 0).astype(np.uint8), connectivity=8)
    grow = np.zeros_like(base_mask_u8)
    added = 0
    touch = cv2.dilate(base_mask_u8, np.ones((3,3), np.uint8), 1)
    for idx in range(1, n):
        comp = (labels == idx).astype(np.uint8)*255
        if (cv2.bitwise_and(comp, touch) > 0).any():
            area = int((comp > 0).sum())
            if added + area <= max_add:
                grow = cv2.bitwise_or(grow, comp)
                added += area

    out = cv2.bitwise_or(base_mask_u8, grow)
    return out


def process_one(net, in_path: Path, out_path: Path):
    bgr = cv2.imread(str(in_path))
    if bgr is None:
        return False

    # Антиблики до saliency
    if USE_SPECULAR_SUPPRESSION:
        bgr_in = suppress_specular(bgr)
    else:
        bgr_in = bgr

    sal = u2net_saliency(net, bgr_in)
    mask = (sal >= THRESH).astype(np.uint8) * 255
    mask = postprocess(mask)

    # sanity по площади
    frac = mask.mean()/255.0
    if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
        # в спорных случаях попробуем восстановить по коридору напрямую
        mask = np.zeros_like(mask)

    # Восстановление тонкого жала вдоль главной оси
    mask = restore_tip_by_corridor(bgr, mask)
    mask = postprocess(mask)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)
    return True


def main():
    assert os.path.isfile(U2NET_WEIGHTS), f"Нет весов: {U2NET_WEIGHTS}"
    net = load_model(U2NET_WEIGHTS)

    files = list_images(INPUT_FOLDER, png_only=PNG_ONLY)
    if not files:
        print("Файлов не найдено.")
        return

    ok = bad = 0
    for p in tqdm(files, desc=f"U²-Net+TipFix ({'cuda' if DEVICE=='cuda' else 'cpu'})"):
        rel = Path(p).relative_to(INPUT_FOLDER)
        out_path = Path(OUTPUT_FOLDER) / rel
        out_path = out_path.with_suffix(".png")
        try:
            if process_one(net, p, out_path):
                ok += 1
            else:
                bad += 1
        except Exception:
            # гарантия совпадения количества выходов с входами
            H, W = (cv2.imread(str(p)).shape[:2] if os.path.isfile(p) else (512,512))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), np.zeros((H, W), np.uint8))
            bad += 1

    print(f"Готово. Успешно: {ok}; ошибок: {bad}")
    print(f"Маски: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
