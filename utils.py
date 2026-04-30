# -*- coding: utf-8 -*-

import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

def load_tiff_to_3d_array(folder_path):
    """
    Загружает TIFF изображения из указанной папки в трехмерный numpy массив.
    
    Args:
        folder_path (str): Путь к папке с TIFF изображениями
        
    Returns:
        numpy.ndarray: Трехмерный массив с изображениями
    """
    # Получаем список всех TIFF файлов в папке
    tiff_files = glob.glob(os.path.join(folder_path, "*.tif")) + glob.glob(os.path.join(folder_path, "*.tiff"))
    
    # Сортируем файлы по имени
    tiff_files.sort()
    
    if not tiff_files:
        raise ValueError(f"В папке {folder_path} не найдены TIFF изображения")
    
    # Открываем первое изображение, чтобы узнать размеры
    sample_img = np.array(Image.open(tiff_files[0]))
    height, width = sample_img.shape
    
    # Создаем трехмерный numpy массив с нужными размерами
    num_images = len(tiff_files)
    image_stack = np.empty((num_images, height, width), dtype=sample_img.dtype)
    
    # Загружаем все изображения в массив
    for i, tiff_file in tqdm(enumerate(tiff_files)):
        img = np.array(Image.open(tiff_file))
        if img.shape != (height, width):
            raise ValueError(f"Размер изображения {tiff_file} не соответствует размеру первого изображения")
        image_stack[i] = img
    
    return image_stack


def safe_path(file_path, allow_rewrite = True):
    if allow_rewrite is False and os.path.exists(file_path):
        raise(IOError, f'File {file_path=} exists. Set allow_rewrite to False to rewite')
        
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    return file_path

def save_amira(in_array, out_path, name, reshape=3, pixel_size=9.0e-3):
    data_path = str(out_path)
    os.makedirs(data_path, exist_ok=True)
    name = name.replace(' ', '_')

    if reshape != 1:
        reshaped_vol = reshape_volume(in_array, reshape)
    else:
        reshaped_vol = in_array
        
    file_shape = reshaped_vol.shape
    shape_str = f'{file_shape[0]}_{file_shape[1]}_{file_shape[2]}'
    out_name = f'{name}.{shape_str}.{reshape}.raw'
    with open(os.path.join(data_path, out_name), 'wb') as amira_file:
        reshaped_vol.tofile(amira_file)
            
    with open(os.path.join(data_path, f'tomo.{name}.{reshape}.hx'), 'w') as af:
        af.write('# Amira Script\n')
        # af.write('remove -all\n')
        template_str = '[ load -unit mm -raw ${{SCRIPTDIR}}/{} ' + \
                       'little xfastest float 1 {} {} {}  0 {} 0 {} 0 {} ] setLabel {}\n'
        af.write(template_str.format(
            out_name,
            file_shape[2], file_shape[1], file_shape[0],
            pixel_size * reshape * (file_shape[2] - 1),
            pixel_size * reshape * (file_shape[1] - 1),
            pixel_size * reshape * (file_shape[0] - 1),
            out_name)
        )


def reshape_volume(array_3d, binning_factor):
    """
    Более эффективная реализация pixel binning для трёхмерного numpy массива
    по всем трём осям.
    
    Параметры:
    array_3d (numpy.ndarray): Входной трёхмерный массив
    binning_factor (int): Коэффициент сжатия (должен быть целым числом > 0)
    
    Возвращает:
    numpy.ndarray: Сжатый массив
    """
    # Проверка входных данных
    if not isinstance(array_3d, np.ndarray) or array_3d.ndim != 3:
        raise ValueError("Входные данные должны быть трёхмерным numpy массивом")
    
    if not isinstance(binning_factor, int) or binning_factor <= 0:
        raise ValueError("Коэффициент сжатия должен быть положительным целым числом")
    
    # Получаем размеры исходного массива
    height, width, depth = array_3d.shape
    
    # Вычисляем новые размеры
    new_height = height // binning_factor
    new_width = width // binning_factor
    new_depth = depth // binning_factor
    
    # Обрезаем массив до размеров, кратных binning_factor
    trimmed_array = array_3d[:new_height*binning_factor, 
                             :new_width*binning_factor, 
                             :new_depth*binning_factor]
    
    # Изменяем форму массива для группировки вокселей
    reshaped = trimmed_array.reshape(new_height, binning_factor, 
                                     new_width, binning_factor, 
                                     new_depth, binning_factor)
    
    # Вычисляем среднее по группам вокселей
    result = reshaped.mean(axis=(1, 3, 5), dtype='float32')
    return result