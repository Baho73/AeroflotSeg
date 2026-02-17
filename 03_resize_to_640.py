import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- НАСТРОЙКИ ---
# Папка с обрезанными изображениями
INPUT_FOLDER = r"E:\data\003" 
# Папка для сохранения уменьшенных изображений
OUTPUT_FOLDER = r"E:\data\004_640"
# Целевой размер для длинной стороны изображения
TARGET_LONG_SIDE = 640
# Количество потоков для обработки. 0 - использовать все ядра.
NUM_PROCESSES = 12
# --- КОНЕЦ НАСТРОЕК ---

def resize_proportional(task):
    """
    Пропорционально изменяет размер изображения по длинной стороне.
    """
    input_path, output_path = task
    try:
        # Используем наш надежный метод чтения
        n = np.fromfile(input_path, np.uint8)
        image = cv2.imdecode(n, cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED для поддержки PNG с прозрачностью
        if image is None: return

        h, w = image.shape[:2]

        # Если изображение уже меньше целевого размера, ничего не делаем, просто копируем
        if max(h, w) <= TARGET_LONG_SIDE:
            with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
            return

        # Вычисляем коэффициент масштабирования
        scale = TARGET_LONG_SIDE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Уменьшаем изображение, используя INTER_AREA - лучший метод для сжатия
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Сохраняем результат в PNG, чтобы не потерять прозрачность, если она будет
        _, buf = cv2.imencode(".png", resized_image)
        buf.tofile(output_path)

    except Exception as e:
        print(f"Ошибка при обработке файла {input_path}: {e}")


def main():
    """
    Главная функция для параллельного изменения размера.
    """
    print(f"Начинаем пропорциональное изменение размера до {TARGET_LONG_SIDE}px...")
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
        list(tqdm(p.imap_unordered(resize_proportional, tasks), total=len(tasks)))

    print(f"\nИзменение размера завершено.")

if __name__ == '__main__':
    main()