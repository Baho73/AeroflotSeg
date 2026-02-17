import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
from rembg import remove

# --- НАСТРОЙКИ ---
# Папка с уменьшенными, пропорциональными изображениями (ВХОД)
INPUT_FOLDER = r"E:\data\004_640" 
# Папка для сохранения изображений без фона ("цифровых стикеров") (ВЫХОД)
OUTPUT_FOLDER = r"E:\data\005"
# Количество потоков для обработки. 0 - использовать все ядра.
NUM_PROCESSES = 12
# --- КОНЕЦ НАСТРОЕК ---

def check_and_download_model():
    """
    Проверяет наличие модели rembg. Если ее нет - скачивает в одном потоке.
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
    Удаляет фон с изображения с помощью rembg.
    """
    input_path, output_path = task
    try:
        # Читаем исходный PNG-файл в бинарном виде
        with open(input_path, 'rb') as i:
            input_data = i.read()

        # Удаляем фон
        output_data = remove(input_data)
        
        # Записываем результат (PNG с прозрачным фоном)
        with open(output_path, 'wb') as o:
            o.write(output_data)

    except Exception as e:
        print(f"Ошибка при обработке файла {input_path}: {e}")

def main():
    """
    Главная функция для параллельного удаления фона.
    """
    print(f"Начинаем Этап 4: Удаление фона...")
    check_and_download_model()

    print("\nПодготовка к обработке...")
    print(f"Исходная папка: {os.path.abspath(INPUT_FOLDER)}")
    print(f"Папка для результатов: {os.path.abspath(OUTPUT_FOLDER)}")

    tasks = []
    for root, _, files in os.walk(INPUT_FOLDER):
        for filename in files:
            if filename.lower().endswith('.png'):
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, INPUT_FOLDER)
                output_path_png = os.path.join(OUTPUT_FOLDER, relative_path)
                os.makedirs(os.path.dirname(output_path_png), exist_ok=True)
                tasks.append((input_path, output_path_png))

    if not tasks:
        print("В исходной папке не найдены изображения.")
        return

    if NUM_PROCESSES == 0:
        num_to_run = cpu_count()
    else:
        num_to_run = NUM_PROCESSES
        
    print(f"Найдено {len(tasks)} изображений. Запускаем обработку на {num_to_run} ядрах.")
        
    with Pool(processes=num_to_run) as p:
        list(tqdm(p.imap_unordered(process_image, tasks), total=len(tasks)))

    print(f"\nУдаление фона завершено. 'Цифровые стикеры' готовы в папке {os.path.basename(OUTPUT_FOLDER)}.")

if __name__ == '__main__':
    main()