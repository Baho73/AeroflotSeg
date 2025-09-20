import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- НАСТРОЙКИ ---
# Папка, где лежат PNG с нарисованными красными рамками
PREVIEW_FOLDER = r"E:\data\002" 
# Папка, где лежат ОРИГИНАЛЬНЫЕ исходные изображения (JPG)
INPUT_FOLDER = r"E:\data\001" 
# Папка, куда будут сохраняться ОБРЕЗАННЫЕ изображения
OUTPUT_FOLDER_CROPPED = r"E:\data\003"

# --- НАСТРОЙКИ ПОИСКА РАМКИ ---
# Точный красный цвет в формате (B, G, R). Должен совпадать с BOX_COLOR из скрипта-валидатора.
TARGET_COLOR_BGR = [0, 0, 255] 
# --- КОНЕЦ НАСТРОЕК ---

# Количество потоков для обработки. 0 - использовать все ядра.
NUM_PROCESSES = 6

def crop_by_red_box(task):
    """
    Находит красную рамку на превью и использует ее координаты для обрезки оригинала.
    """
    preview_path, original_path, output_path = task
    try:
        # 1. Загружаем превью с рамкой
        n_preview = np.fromfile(preview_path, np.uint8)
        preview_image = cv2.imdecode(n_preview, cv2.IMREAD_COLOR)
        if preview_image is None: 
            print(f"WARN: Не удалось прочитать превью: {preview_path}")
            return

        # 2. Находим красную рамку
        # Создаем маску для идеально красного цвета
        lower_red = np.array(TARGET_COLOR_BGR)
        upper_red = np.array(TARGET_COLOR_BGR)
        red_mask = cv2.inRange(preview_image, lower_red, upper_red)

        # Находим контуры на этой маске. Там должна быть только рамка.
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Находим самый большой контур (на случай мелкого шума)
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 3. Загружаем ОРИГИНАЛЬНОЕ изображение
            n_original = np.fromfile(original_path, np.uint8)
            original_image = cv2.imdecode(n_original, cv2.IMREAD_COLOR)
            if original_image is None: 
                print(f"WARN: Не удалось прочитать оригинал: {original_path}")
                return

            # 4. ВЫРЕЗАЕМ ОРИГИНАЛ по найденным координатам
            cropped_image = original_image[y:y+h, x:x+w]
            
            # 5. Сохраняем результат в PNG (lossless)
            _, buf = cv2.imencode(".png", cropped_image)
            buf.tofile(output_path)

    except Exception as e:
        print(f"Ошибка при обработке файла {preview_path}: {e}")


def main():
    """
    Главная функция для параллельной обрезки по превью.
    """
    print(f"Начинаем быструю обрезку по готовым превью...")
    print(f"Папка с превью: {os.path.abspath(PREVIEW_FOLDER)}")
    print(f"Папка с оригиналами: {os.path.abspath(INPUT_FOLDER)}")
    print(f"Папка для результатов: {os.path.abspath(OUTPUT_FOLDER_CROPPED)}")

    tasks = []
    # Идем по папке с ГОТОВЫМИ ПРЕВЬЮ
    for root, _, files in os.walk(PREVIEW_FOLDER):
        for filename in files:
            if filename.lower().endswith('.png'):
                preview_path = os.path.join(root, filename)
                
                # --- ВАЖНЫЙ ШАГ: Находим соответствующий оригинальный файл ---
                relative_path = os.path.relpath(preview_path, PREVIEW_FOLDER)
                base, _ = os.path.splitext(relative_path)
                
                # Собираем пути для всех возможных расширений оригинала
                possible_originals = [
                    os.path.join(INPUT_FOLDER, base + ".jpg"),
                    os.path.join(INPUT_FOLDER, base + ".jpeg"),
                    os.path.join(INPUT_FOLDER, base + ".png")
                ]

                original_path = None
                for path in possible_originals:
                    if os.path.exists(path):
                        original_path = path
                        break
                
                if original_path is None:
                    print(f"Не найден оригинал для {preview_path}")
                    continue
                # --- КОНЕЦ ВАЖНОГО ШАГА ---
                
                output_path_png = os.path.join(OUTPUT_FOLDER_CROPPED, base + ".png")
                os.makedirs(os.path.dirname(output_path_png), exist_ok=True)
                tasks.append((preview_path, original_path, output_path_png))

    if not tasks:
        print("В папке с превью не найдены изображения.")
        return

    if NUM_PROCESSES == 0:
        num_to_run = cpu_count()
    else:
        num_to_run = NUM_PROCESSES
        
    print(f"Найдено {len(tasks)} превью для обработки. Запускаем обрезку на {num_to_run} ядрах.")
        
    with Pool(processes=num_to_run) as p:
        list(tqdm(p.imap_unordered(crop_by_red_box, tasks), total=len(tasks)))

    print(f"\nФинальная обрезка по превью завершена.")

if __name__ == '__main__':
    main()