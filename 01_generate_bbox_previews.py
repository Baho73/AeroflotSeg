import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
from rembg import remove

# --- НАСТРОЙКИ ---
# Папка с ИСХОДНЫМИ фотографиями
INPUT_FOLDER = r"E:\data\001" 
# Папка, куда будут сохраняться НОВЫЕ результаты этой сессии
OUTPUT_FOLDER = r"E:\data\002_validation_previews_2"
# Папка со СТАРЫМИ, частично выполненными результатами (для проверки)
EXISTING_OUTPUT_FOLDER = r"E:\data\002_validation_previews"

# --- НАСТРОЙКИ ОБРАБОТКИ (остаются без изменений) ---
BOX_COLOR = (0, 0, 255)
BOX_THICKNESS = 16
PADDING = 128
NUM_PROCESSES = 10
# --- КОНЕЦ НАСТРОЕК ---

def check_and_download_model():
    """
    Проверяет и скачивает модель rembg.
    """
    model_name = "u2net.onnx"
    model_path = Path.home() / ".u2net" / model_name
    if not model_path.exists():
        print(f"Модель '{model_name}' не найдена. Запускаю скачивание...")
        remove(np.zeros((1, 1, 3), np.uint8))
        print("Скачивание завершено.")
    else:
        print(f"Модель '{model_name}' на месте.")

def process_image(task):
    """
    Функция-обработчик для одного файла (остается без изменений).
    """
    input_path, output_path = task
    try:
        with open(input_path, 'rb') as i:
            input_data = i.read()
        output_data = remove(input_data)
        image_bgra = cv2.imdecode(np.frombuffer(output_data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image_bgra is None: return
        alpha_channel = image_bgra[:, :, 3]
        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            original_image = cv2.imdecode(np.frombuffer(input_data, np.uint8), cv2.IMREAD_COLOR)
            img_height, img_width, _ = original_image.shape
            x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
            x2, y2 = min(img_width, x + w + PADDING), min(img_height, y + h + PADDING)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            _, buf = cv2.imencode(".png", original_image)
            buf.tofile(output_path)
    except Exception as e:
        print(f"ERROR in {input_path}: {e}")

def main():
    """
    Главная функция с логикой возобновления (resume).
    """
    print(f"Начинаем создание превью (v15, Отказоустойчивая версия)...")
    check_and_download_model()

    print("\nПодготовка к обработке...")
    print(f"Исходная папка: {os.path.abspath(INPUT_FOLDER)}")
    print(f"Папка для проверки: {os.path.abspath(EXISTING_OUTPUT_FOLDER)}")
    print(f"Папка для новых результатов: {os.path.abspath(OUTPUT_FOLDER)}")

    # --- НОВАЯ ЛОГИКА: ФИЛЬТРАЦИЯ ЗАДАЧ ---
    # 1. Собираем список ВСЕХ исходных файлов
    all_input_files = []
    for root, _, files in os.walk(INPUT_FOLDER):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_input_files.append(os.path.join(root, filename))

    if not all_input_files:
        print("В исходной папке не найдены изображения.")
        return

    # 2. Формируем список задач, пропуская уже выполненные
    tasks_to_run = []
    skipped_count = 0
    print("\nПроверка существующих файлов для возобновления работы...")
    for input_path in tqdm(all_input_files, desc="Анализ"):
        relative_path = os.path.relpath(input_path, INPUT_FOLDER)
        base, _ = os.path.splitext(relative_path)
        
        # Проверяем наличие файла в СТАРОЙ папке
        check_path = os.path.join(EXISTING_OUTPUT_FOLDER, base + ".png")

        if os.path.exists(check_path):
            skipped_count += 1
            continue  # Пропускаем, работа уже сделана

        # Если файла нет, добавляем задачу в список, указывая НОВУЮ папку для вывода
        output_path = os.path.join(OUTPUT_FOLDER, base + ".png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tasks_to_run.append((input_path, output_path))
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    print(f"\nОбнаружено {len(all_input_files)} исходных файлов.")
    print(f"Пропущено {skipped_count} уже обработанных файлов.")
    
    if not tasks_to_run:
        print("Все файлы уже обработаны. Завершение работы.")
        return
        
    print(f"Осталось обработать: {len(tasks_to_run)} файлов.")

    if NUM_PROCESSES == 0:
        num_to_run = cpu_count()
    else:
        num_to_run = NUM_PROCESSES
    
    print(f"Всего доступно ядер: {cpu_count()}. Запускаем обработку на {num_to_run} ядрах.")

    with Pool(processes=num_to_run) as p:
        list(tqdm(p.imap_unordered(process_image, tasks_to_run), total=len(tasks_to_run)))

    print(f"\nОбработка завершена.")
    print(f"Новые результаты сохранены в папку: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == '__main__':
    main()