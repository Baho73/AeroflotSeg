# AeroflotSeg

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-CV-EE4C2C?logo=pytorch) ![License: MIT](https://img.shields.io/github/license/Baho73/AeroflotSeg)

Пайплайн сегментации объектов на фотографиях для Аэрофлота. Задача — точно вырезать металлические инструменты и детали с фона, включая объекты с бликами и тонкими кончиками.

## Пайплайн

| Этап | Скрипт | Описание |
|------|--------|----------|
| 1 | `01_generate_bbox_previews.py` | Удаление фона (rembg) и генерация превью с bounding box |
| 2 | `02_crop_by_bbox.py` | Кроп оригиналов по найденным bbox |
| 3 | `03_resize_to_640.py` | Пропорциональный ресайз (длинная сторона → 640px) |
| 4 | `04_rembg_remove_background.py` | Финальное удаление фона через rembg |
| 4a | `04a_sam_segment_with_tip_fix.py` | SAM ViT-H + подавление бликов + восстановление кончиков |
| 4b | `04b_sam_amg_ensemble_scoring.py` | SAM AMG ансамбль с оценкой по элонгации/площади/стабильности |
| 4c | `04c_u2net_saliency_segmentation.py` | U2-Net сегментация значимых объектов |
| 4d | `04d_u2net_with_pca_tip_recovery.py` | U2-Net + PCA-коридор для восстановления тонких концов |

## Ключевые решения

- **Подавление бликов** — металлические объекты дают спекулярные блики, которые ломают стандартные модели. HSV-анализ + inpainting перед сегментацией
- **Восстановление кончиков** — тонкие концы инструментов (жала отвёрток) теряются при сегментации. PCA-анализ маски определяет главную ось, затем «коридорный рост» вдоль оси восстанавливает потерянные пиксели
- **Ансамбль моделей** — 4 подхода к сегментации (rembg, SAM point prompts, SAM AMG ensemble, U2-Net) с возможностью выбора лучшего для конкретного типа объектов

## Стек

`Python` `OpenCV` `PyTorch` `Segment Anything (SAM)` `U2-Net` `rembg` `NumPy` `multiprocessing`
