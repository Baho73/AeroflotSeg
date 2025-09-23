import os
import numpy as np
#from tqdm import tqdm
#from multiprocessing import Pool, cpu_count
from pathlib import Path
import cv2

# --- НАСТРОЙКИ ---
# Папка с уменьшенными, пропорциональными изображениями (ВХОД)
INPUT_FOLDER = r"C:\Users\dponomarev5\Desktop\data\004\1" 
# Папка для сохранения изображений без фона ("цифровых стикеров") (ВЫХОД)
OUTPUT_FOLDER = r"C:\Users\dponomarev5\Desktop\data\005\1"


kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) #Ядро для замыкания. Увеличить если останутся дырки от бликов на изображении
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)) #Ядро для размыкания. Увеличить если остаются артефакты помимо объекта. Уменьшить если начали сам объект подрезать
border_expansion = 20 #Должен быть чуть больше чем размер ядра для замыкания




def remove_background_with_glare_handling(task, kernel_close, kernel_open, border_expansion):
    input_path, output_path = task
    
    try:
        # 1. Чтение изображения
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Ошибка: не удалось загрузить изображение {input_path}")
            return False
        
        # 2. Конвертация в grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Вычисляем градиент
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient = np.uint8(255 * gradient / gradient.max())

        # Порог по градиенту - хорошо видит границы металлических объектов
        _, mask_grad = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)

        # Простой порог Оцу - видит внутренности объекта
        _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Объединяем две маски
        mask = cv2.bitwise_or(mask_otsu, mask_grad)

        # 4. РАСШИРЕНИЕ МАСКИ за границы объекта
        kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_expansion, border_expansion))
        expanded_mask = cv2.dilate(mask, kernel_expand, iterations=1)

        # 5. ЗАКРЫТИЕ (CLOSING) на расширенной маске
        mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, kernel_close)

        # 6. Размыкание
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # 7. СЖАТИЕ МАСКИ обратно к исходным границам
        kernel_contract = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_expansion, border_expansion))
        mask = cv2.erode(mask, kernel_contract, iterations=1)

        # 8. Применяем маску к изображению
        result = cv2.bitwise_and(img, img, mask=mask)
        
        # 9. Создаем прозрачный фон
        b, g, r = cv2.split(result)
        alpha = mask
        rgba = cv2.merge([b, g, r, alpha])
        
        # 10. Сохраняем результат
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), rgba)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке {input_path}: {str(e)}")
        return False




def main():
    """
    Главная функция для удаления фона.
    """
    print(f"Начинаем Этап 4: Удаление фона...")

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

        
    print(f"Найдено {len(tasks)} изображений.")
    
    
    for task in tasks:
        remove_background_with_glare_handling(task, kernel_close, kernel_open, border_expansion)
    print('Обработка завершена')

if __name__ == '__main__':
    main()
