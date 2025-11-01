from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import io
import base64
import json
import os
import math

app = Flask(__name__)

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.image_array = None
        self.history = []
        
    def load_image(self, image_file):
        """Загрузка изображения"""
        self.original_image = Image.open(image_file).convert('RGB')
        self.current_image = self.original_image.copy()
        self.image_array = np.array(self.current_image, dtype=np.float32)
        self.history = ["original"]
        return self.get_image_info()
    
    def get_image_info(self):
        """Получение информации об изображении"""
        if self.current_image is None:
            return None
        return {
            'width': self.current_image.width,
            'height': self.current_image.height,
            'mode': self.current_image.mode
        }
    
    def _calculate_brightness(self, image_array):
        """Вычисление яркости каждого пикселя как среднее по каналам RGB"""
        height, width, channels = image_array.shape
        brightness = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                r = image_array[i, j, 0]
                g = image_array[i, j, 1] 
                b = image_array[i, j, 2]
                brightness[i, j] = (r + g + b) / 3.0
                
        return brightness
    
    def _find_max_brightness(self, image_array):
        """Нахождение максимальной яркости в изображении"""
        brightness = self._calculate_brightness(image_array)
        max_val = 0
        
        height, width = brightness.shape
        for i in range(height):
            for j in range(width):
                if brightness[i, j] > max_val:
                    max_val = brightness[i, j]
                    
        return max_val
    
    # Преобразования цветности
    def logarithmic_transform(self):
        """Логарифмическое преобразование"""
        if self.image_array is None:
            return False

        max_brightness = self._find_max_brightness(self.image_array)
        c = 255.0 / math.log(1 + max_brightness)
        
        height, width, channels = self.image_array.shape
        new_array = np.zeros_like(self.image_array)
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    pixel_value = self.image_array[i, j, k]
                    new_value = c * math.log(1 + pixel_value)
                    new_array[i, j, k] = max(0, min(255, new_value))
        
        self.image_array = new_array
        self._update_current_image()
        return True
    
    def power_transform(self, gamma):
        """Степенное преобразование"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        new_array = np.zeros_like(self.image_array)
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    normalized = self.image_array[i, j, k] / 255.0
                    transformed = math.pow(normalized, gamma)
                    new_value = transformed * 255.0
                    new_array[i, j, k] = max(0, min(255, new_value))
        
        self.image_array = new_array
        self._update_current_image()
        return True
    
    def binary_transform(self, threshold):
        """Бинарное преобразование"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        new_array = np.zeros_like(self.image_array)
        
        for i in range(height):
            for j in range(width):
                r = self.image_array[i, j, 0]
                g = self.image_array[i, j, 1]
                b = self.image_array[i, j, 2]
                brightness = (r + g + b) / 3.0
                
                if brightness > threshold:
                    binary_value = 255
                else:
                    binary_value = 0
                
                for k in range(channels):
                    new_array[i, j, k] = binary_value
        
        self.image_array = new_array
        self._update_current_image()
        return True
    
    def intensity_range_cut(self, min_val, max_val, constant_value=0, keep_original=False):
        """Вырезание диапазона яркостей"""
        if self.image_array is None:
            return False
        
        if min_val > max_val:
            return False
            
        height, width, channels = self.image_array.shape
        new_array = np.zeros_like(self.image_array)
        
        for i in range(height):
            for j in range(width):
                r = self.image_array[i, j, 0]
                g = self.image_array[i, j, 1]
                b = self.image_array[i, j, 2]
                brightness = (r + g + b) / 3.0
                
                if min_val <= brightness <= max_val:
                    for k in range(channels):
                        new_array[i, j, k] = self.image_array[i, j, k]
                else:
                    if keep_original:
                        for k in range(channels):
                            new_array[i, j, k] = self.image_array[i, j, k]
                    else:
                        for k in range(channels):
                            new_array[i, j, k] = constant_value
        
        self.image_array = new_array
        self._update_current_image()
        return True
    
    # Фильтры сглаживания
    def rectangular_filter(self, kernel_size=3):
        """Прямоугольный фильтр"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        new_array = np.zeros_like(self.image_array)
        pad = kernel_size // 2
        
        mean_value = np.mean(self.image_array)
        padded = np.full((height + 2*pad, width + 2*pad, channels), mean_value)
        padded[pad:pad+height, pad:pad+width, :] = self.image_array
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    window = padded[i:i+kernel_size, j:j+kernel_size, k]
                    new_value = np.mean(window)
                    new_array[i, j, k] = new_value
        
        self.image_array = new_array
        self._update_current_image()
        return True

    def median_filter(self, kernel_size=3):
        """Медианный фильтр"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        new_array = np.zeros_like(self.image_array)
        pad = kernel_size // 2
        
        mean_value = np.mean(self.image_array)
        padded = np.full((height + 2*pad, width + 2*pad, channels), mean_value)
        padded[pad:pad+height, pad:pad+width, :] = self.image_array
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    window = padded[i:i+kernel_size, j:j+kernel_size, k]
                    values = window.flatten()
                    new_value = np.median(values)
                    new_array[i, j, k] = new_value
        
        self.image_array = new_array
        self._update_current_image()
        return True
    
    def _create_gaussian_kernel_simple(self, size, sigma):
        """Создание ядра Гаусса"""
        kernel = np.zeros((size, size))
        center = size // 2
        total = 0.0
        
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                value = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
                kernel[i, j] = value
                total += value
        
        for i in range(size):
            for j in range(size):
                kernel[i, j] /= total
                
        return kernel
    
    def gaussian_filter(self, sigma):
        """Фильтр Гаусса"""
        if self.image_array is None:
            return False
            
        kernel_size = int(3 * sigma) * 2 + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        kernel = self._create_gaussian_kernel_simple(kernel_size, sigma)
        
        height, width, channels = self.image_array.shape
        new_array = np.zeros_like(self.image_array)
        pad = kernel_size // 2
        
        mean_value = np.mean(self.image_array)
        padded = np.full((height + 2*pad, width + 2*pad, channels), mean_value)
        padded[pad:pad+height, pad:pad+width, :] = self.image_array
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    total = 0.0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            pixel_value = padded[i+ki, j+kj, k]
                            weight = kernel[ki, kj]
                            total += pixel_value * weight
                    
                    new_array[i, j, k] = max(0, min(255, total))
        
        self.image_array = new_array
        self._update_current_image()
        return True
    
    def sigma_filter(self, sigma):
        """Сигма-фильтр"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        new_array = np.zeros_like(self.image_array)
        kernel_size = 3
        pad = kernel_size // 2
        
        mean_value = np.mean(self.image_array)
        padded = np.full((height + 2*pad, width + 2*pad, channels), mean_value)
        padded[pad:pad+height, pad:pad+width, :] = self.image_array
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    center_value = padded[i+pad, j+pad, k]
                    window = padded[i:i+kernel_size, j:j+kernel_size, k]
                    
                    valid_values = []
                    for wi in range(kernel_size):
                        for wj in range(kernel_size):
                            pixel_value = window[wi, wj]
                            if abs(pixel_value - center_value) <= sigma:
                                valid_values.append(pixel_value)
                    
                    if valid_values:
                        new_value = sum(valid_values) / len(valid_values)
                    else:
                        new_value = center_value
                    
                    new_array[i, j, k] = max(0, min(255, new_value))
        
        self.image_array = new_array
        self._update_current_image()
        return True
    
    def add_noise(self, noise_type='gaussian', amount=0.1):
        """Добавление шума"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        noisy_array = self.image_array.copy()
        
        if noise_type == 'gaussian':
            for i in range(height):
                for j in range(width):
                    for k in range(channels):
                        noise = np.random.normal(0, amount * 255)
                        noisy_array[i, j, k] += noise
        elif noise_type == 'salt_pepper':
            for i in range(height):
                for j in range(width):
                    rand_val = np.random.random()
                    if rand_val < amount:
                        noisy_array[i, j, :] = 0
                    elif rand_val < 2 * amount:
                        noisy_array[i, j, :] = 255
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    noisy_array[i, j, k] = max(0, min(255, noisy_array[i, j, k]))
        
        self.image_array = noisy_array
        self._update_current_image()
        return True
    
    def get_difference_map(self, original_array):
        """Карта разности с оригиналом"""
        if self.image_array is None or original_array is None:
            return None
            
        height, width, channels = self.image_array.shape
        diff_array = np.zeros_like(self.image_array)
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    diff = abs(self.image_array[i, j, k] - original_array[i, j, k])
                    diff_array[i, j, k] = diff
        
        diff_image = Image.fromarray(diff_array.astype(np.uint8))
        img_io = io.BytesIO()
        diff_image.save(img_io, 'PNG')
        img_io.seek(0)
        return base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    # Увеличение резкости
    def unsharp_masking(self, k_size=3, lambda_val=1.0):
        """Нерезкое маскирование"""
        if self.image_array is None:
            return False
            
        original_array = self.image_array.copy()
        
        sigma = max(0.5, k_size / 3.0)
        self.gaussian_filter(sigma=sigma)
        blurred_array = self.image_array.copy()
        
        self.image_array = original_array.copy()
        
        height, width, channels = self.image_array.shape
        sharpened_array = np.zeros_like(self.image_array)
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    mask = original_array[i, j, k] - blurred_array[i, j, k]
                    sharpened = original_array[i, j, k] + lambda_val * mask
                    sharpened_array[i, j, k] = max(0, min(255, sharpened))
        
        self.image_array = sharpened_array
        self._update_current_image()
        return True
    
    # 1. High-pass фильтр
    def high_pass_filter(self, blur_type='average', kernel_size=3, c=1.0):
        """High-pass фильтр"""
        if self.image_array is None:
            return False
            
        original_array = self.image_array.copy()
        
        if blur_type == 'average':
            self.rectangular_filter(kernel_size)
        elif blur_type == 'gaussian':
            sigma = max(0.5, kernel_size / 3.0)
            self.gaussian_filter(sigma)
        
        blurred_array = self.image_array.copy()
        
        self.image_array = original_array.copy()
        
        # ВЧ = ИСХ - РАЗМ*c
        height, width, channels = self.image_array.shape
        high_pass_array = np.zeros_like(self.image_array)
        
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    high_pass_value = original_array[i, j, k] - blurred_array[i, j, k] * c
                    high_pass_array[i, j, k] = max(0, min(255, high_pass_value + 128))
        
        self.image_array = high_pass_array
        self._update_current_image()
        return True
    
    # 2. Свёртка с произвольной матрицей
    def apply_convolution(self, kernel, normalize=True, add_128=False):
        """Применение свёртки с произвольной матрицей"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        kernel_size = len(kernel)

        if kernel_size % 2 == 0:
            return False 
        
        pad = kernel_size // 2
        
        new_array = np.zeros_like(self.image_array)
        
        if normalize:
            kernel_sum = np.sum(kernel)
            if kernel_sum != 0:
                kernel = kernel / kernel_sum
        
        for k in range(channels):
            padded = np.pad(self.image_array[:, :, k], pad, mode='edge')
            
            for i in range(height):
                for j in range(width):
                    total = 0.0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            pixel_value = padded[i + ki, j + kj]
                            weight = kernel[ki][kj]
                            total += pixel_value * weight
                    
                    if add_128:
                        total += 128
                    
                    new_array[i, j, k] = max(0, min(255, total))
        
        
        self.image_array = new_array
        self._update_current_image()
        return True
    
    # 3. Метод выделения углов 
    def harris_corner_detection(self, k=0.04, threshold=0.01):
        """Детектор углов Харриса"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        gray = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                r = self.image_array[i, j, 0]
                g = self.image_array[i, j, 1]
                b = self.image_array[i, j, 2]
                gray[i, j] = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray)) * 255
        
        Ix = np.zeros_like(gray)
        Iy = np.zeros_like(gray)
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        pad = 1
        padded_gray = np.pad(gray, pad, mode='edge')
        
        for i in range(height):
            for j in range(width):
                gx = 0.0
                for ki in range(3):
                    for kj in range(3):
                        gx += padded_gray[i + ki, j + kj] * sobel_x[ki, kj]
                Ix[i, j] = gx
                
                gy = 0.0
                for ki in range(3):
                    for kj in range(3):
                        gy += padded_gray[i + ki, j + kj] * sobel_y[ki, kj]
                Iy[i, j] = gy
        
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy
        
        def gaussian_filter_for_harris(image, sigma=1.0):
            """Гауссов фильтр"""
            size = int(3 * sigma) * 2 + 1
            if size % 2 == 0:
                size += 1
            pad = size // 2
            
            kernel = np.zeros((size, size))
            total = 0.0
            for i in range(size):
                for j in range(size):
                    x = i - pad
                    y = j - pad
                    value = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
                    kernel[i, j] = value
                    total += value
            
            kernel /= total
            
            h, w = image.shape
            padded = np.pad(image, pad, mode='edge')
            result = np.zeros_like(image)
            
            for i in range(h):
                for j in range(w):
                    total_val = 0.0
                    for ki in range(size):
                        for kj in range(size):
                            total_val += padded[i + ki, j + kj] * kernel[ki, kj]
                    result[i, j] = total_val
            
            return result
        
        Ix2 = gaussian_filter_for_harris(Ix2, sigma=1.0)
        Iy2 = gaussian_filter_for_harris(Iy2, sigma=1.0)
        Ixy = gaussian_filter_for_harris(Ixy, sigma=1.0)
        
        det = Ix2 * Iy2 - Ixy * Ixy
        trace = Ix2 + Iy2
        
        R = det - k * trace * trace
        
        R_flat = R.flatten()
        R_flat_sorted = np.sort(R_flat)
        
        if threshold < 0.1:  
            auto_threshold = np.percentile(R_flat_sorted, 95)  
        else:
            auto_threshold = np.percentile(R_flat_sorted, 100 * (1 - threshold))
        
        corners = R > auto_threshold
        
        result_array = self.image_array.copy()
        
        corner_count = 0
        for i in range(height):
            for j in range(width):
                if corners[i, j]:
                    corner_count += 1
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                result_array[ni, nj, 0] = 255  
                                result_array[ni, nj, 1] = 0
                                result_array[ni, nj, 2] = 0
        
        self.image_array = result_array
        self._update_current_image()
        return True

    def shi_tomasi_corner_detection(self, threshold=0.01):
        """Детектор углов Ши-Томаси"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        gray = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                r = self.image_array[i, j, 0]
                g = self.image_array[i, j, 1]
                b = self.image_array[i, j, 2]
                gray[i, j] = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray)) * 255
        
        Ix = np.zeros_like(gray)
        Iy = np.zeros_like(gray)
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        pad = 1
        padded_gray = np.pad(gray, pad, mode='edge')
        
        for i in range(height):
            for j in range(width):
                gx = 0.0
                for ki in range(3):
                    for kj in range(3):
                        gx += padded_gray[i + ki, j + kj] * sobel_x[ki, kj]
                Ix[i, j] = gx
                
                gy = 0.0
                for ki in range(3):
                    for kj in range(3):
                        gy += padded_gray[i + ki, j + kj] * sobel_y[ki, kj]
                Iy[i, j] = gy
        
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy
        
        def gaussian_filter_for_shi_tomasi(image, sigma=1.0):
            """Гауссов фильтр"""
            size = int(3 * sigma) * 2 + 1
            if size % 2 == 0:
                size += 1
            pad = size // 2
            
            kernel = np.zeros((size, size))
            total = 0.0
            for i in range(size):
                for j in range(size):
                    x = i - pad
                    y = j - pad
                    value = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
                    kernel[i, j] = value
                    total += value
            
            kernel /= total
            
            h, w = image.shape
            padded = np.pad(image, pad, mode='edge')
            result = np.zeros_like(image)
            
            for i in range(h):
                for j in range(w):
                    total_val = 0.0
                    for ki in range(size):
                        for kj in range(size):
                            total_val += padded[i + ki, j + kj] * kernel[ki, kj]
                    result[i, j] = total_val
            
            return result
        
        Ix2 = gaussian_filter_for_shi_tomasi(Ix2, sigma=1.0)
        Iy2 = gaussian_filter_for_shi_tomasi(Iy2, sigma=1.0)
        Ixy = gaussian_filter_for_shi_tomasi(Ixy, sigma=1.0)
        
        det = Ix2 * Iy2 - Ixy * Ixy
        trace = Ix2 + Iy2
        
        lambda_min = (trace - np.sqrt(trace * trace - 4 * det)) / 2
        
        R = lambda_min
        
        R_flat = R.flatten()
        R_flat_sorted = np.sort(R_flat)
        
        if threshold < 0.1:  
            auto_threshold = np.percentile(R_flat_sorted, 95)  
        else:
            auto_threshold = np.percentile(R_flat_sorted, 100 * (1 - threshold))
        
        corners = R > auto_threshold
        
        result_array = self.image_array.copy()
        
        corner_count = 0
        for i in range(height):
            for j in range(width):
                if corners[i, j]:
                    corner_count += 1
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                result_array[ni, nj, 0] = 0    
                                result_array[ni, nj, 1] = 0    
                                result_array[ni, nj, 2] = 255  
        
        self.image_array = result_array
        self._update_current_image()
        return True
    
    # 4. Методы выделения границ
    def sobel_edge_detection(self):
        """Оператор Собеля для выделения границ"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        gray = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                r = self.image_array[i, j, 0]
                g = self.image_array[i, j, 1]
                b = self.image_array[i, j, 2]
                gray[i, j] = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        def convolve(image, kernel):
            """Свертка"""
            h, w = image.shape
            k_size = kernel.shape[0]
            pad = k_size // 2
            
            padded = np.pad(image, pad, mode='edge')
            result = np.zeros_like(image)
            
            for i in range(h):
                for j in range(w):
                    total = 0.0
                    for ki in range(k_size):
                        for kj in range(k_size):
                            total += padded[i + ki, j + kj] * kernel[ki, kj]
                    result[i, j] = total
            return result
        
        edges_x = convolve(gray, sobel_x)
        edges_y = convolve(gray, sobel_y)
        
        gradient_magnitude = np.zeros_like(gray)
        for i in range(height):
            for j in range(width):
                gx = edges_x[i, j]
                gy = edges_y[i, j]
                gradient_magnitude[i, j] = math.sqrt(gx*gx + gy*gy)
        
        max_val = np.max(gradient_magnitude)
        if max_val > 0:
            gradient_magnitude = (gradient_magnitude / max_val) * 255
        
        result_array = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                value = gradient_magnitude[i, j]
                result_array[i, j, 0] = value
                result_array[i, j, 1] = value
                result_array[i, j, 2] = value
        
        self.image_array = result_array
        self._update_current_image()
        return True

    def canny_edge_detection(self, low_threshold=50, high_threshold=150):
        """Детектор границ Канни"""
        if self.image_array is None:
            return False
            
        height, width, channels = self.image_array.shape
        gray = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                r = self.image_array[i, j, 0]
                g = self.image_array[i, j, 1]
                b = self.image_array[i, j, 2]
                gray[i, j] = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        def gaussian_filter_for_canny(image, sigma=1.4):
            size = int(3 * sigma) * 2 + 1
            if size % 2 == 0:
                size += 1
            pad = size // 2
            
            kernel = np.zeros((size, size))
            total = 0.0
            for i in range(size):
                for j in range(size):
                    x = i - pad
                    y = j - pad
                    value = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
                    kernel[i, j] = value
                    total += value
            
            kernel /= total
            
            h, w = image.shape
            padded = np.pad(image, pad, mode='edge')
            result = np.zeros_like(image)
            
            for i in range(h):
                for j in range(w):
                    total_val = 0.0
                    for ki in range(size):
                        for kj in range(size):
                            total_val += padded[i + ki, j + kj] * kernel[ki, kj]
                    result[i, j] = total_val
            
            return result
        
        blurred = gaussian_filter_for_canny(gray, sigma=1.4)
        
        def sobel_for_canny(image):
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            h, w = image.shape
            padded = np.pad(image, 1, mode='edge')
            
            grad_x = np.zeros_like(image)
            grad_y = np.zeros_like(image)
            
            for i in range(h):
                for j in range(w):
                    gx = 0.0
                    for ki in range(3):
                        for kj in range(3):
                            gx += padded[i + ki, j + kj] * sobel_x[ki, kj]
                    grad_x[i, j] = gx
                    
                    gy = 0.0
                    for ki in range(3):
                        for kj in range(3):
                            gy += padded[i + ki, j + kj] * sobel_y[ki, kj]
                    grad_y[i, j] = gy
            
            return grad_x, grad_y
        
        sobel_x, sobel_y = sobel_for_canny(blurred)
        
        gradient_magnitude = np.zeros_like(gray)
        gradient_direction = np.zeros_like(gray)
        
        for i in range(height):
            for j in range(width):
                gx = sobel_x[i, j]
                gy = sobel_y[i, j]
                gradient_magnitude[i, j] = math.sqrt(gx*gx + gy*gy)
                if gx != 0:
                    gradient_direction[i, j] = math.atan2(gy, gx)
                else:
                    gradient_direction[i, j] = math.pi / 2 if gy > 0 else -math.pi / 2
        
        suppressed = np.zeros_like(gradient_magnitude)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                angle = gradient_direction[i, j] * 180 / math.pi
                if angle < 0:
                    angle += 180
                
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180): # горизонталь
                    neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]]
                elif 22.5 <= angle < 67.5: # диагональ 
                    neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
                elif 67.5 <= angle < 112.5: # вертикаль
                    neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
                elif 112.5 <= angle < 157.5: # диагональ
                    neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
                else:
                    neighbors = [0, 0]
                
                if gradient_magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = gradient_magnitude[i, j]
        
        strong_edges = suppressed > high_threshold
        weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
        
        edges = strong_edges.copy()
        
        changed = True
        while changed:
            changed = False
            for i in range(1, height-1):
                for j in range(1, width-1):
                    if weak_edges[i, j] and not edges[i, j]:
                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < height and 0 <= nj < width:
                                    if edges[ni, nj]:
                                        edges[i, j] = True
                                        changed = True
                                        break
                            if changed:
                                break
        
        result_array = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                if edges[i, j]:
                    value = 255
                else:
                    value = 0
                result_array[i, j, 0] = value
                result_array[i, j, 1] = value
                result_array[i, j, 2] = value
        
        self.image_array = result_array
        self._update_current_image()
        return True
    
    def apply_transformation(self, transform_type, **params):
        """Универсальный метод применения преобразований"""
        if self.image_array is None:
            return False
            
        success = False
        
        if transform_type == 'logarithmic':
            success = self.logarithmic_transform()
        elif transform_type == 'power':
            success = self.power_transform(params.get('gamma', 1.0))
        elif transform_type == 'binary':
            success = self.binary_transform(params.get('threshold', 128))
        elif transform_type == 'intensity_range':
            success = self.intensity_range_cut(
                params.get('min_val', 0),
                params.get('max_val', 255),
                params.get('constant', 0),
                params.get('keep_original', False)
            )
        elif transform_type == 'rectangular_filter':
            success = self.rectangular_filter(params.get('kernel_size', 3))
        elif transform_type == 'median_filter':
            success = self.median_filter(params.get('kernel_size', 3))
        elif transform_type == 'gaussian_filter':
            success = self.gaussian_filter(params.get('sigma', 1.0))
        elif transform_type == 'sigma_filter':
            success = self.sigma_filter(params.get('sigma', 1.0))
        elif transform_type == 'add_noise':
            success = self.add_noise(
                params.get('noise_type', 'gaussian'),
                params.get('amount', 0.1)
            )
        elif transform_type == 'unsharp_masking':
            success = self.unsharp_masking(
                params.get('k_size', 3),
                params.get('lambda_val', 1.0)
            )
        elif transform_type == 'high_pass':
            success = self.high_pass_filter(
                params.get('blur_type', 'average'),
                params.get('kernel_size', 3),
                params.get('c', 1.0)
            )
        elif transform_type == 'convolution':
            success = self.apply_convolution(
                params.get('kernel'),
                params.get('normalize', True),
                params.get('add_128', False)
            )
        elif transform_type == 'harris_corners':
            success = self.harris_corner_detection(
                params.get('k', 0.04),
                params.get('threshold', 0.01)
            )
        elif transform_type == 'shi_tomasi_corners':
            success = self.shi_tomasi_corner_detection(
                params.get('threshold', 0.01)
            )
        elif transform_type == 'sobel_edges':
            success = self.sobel_edge_detection()
        elif transform_type == 'canny_edges':
            success = self.canny_edge_detection(
                params.get('low_threshold', 50),
                params.get('high_threshold', 150)
            )
        elif transform_type == 'reset':
            success = self.reset_image()
        
        if success:
            history_entry = transform_type
            if params:
                history_entry += f"({', '.join(f'{k}={v}' for k, v in params.items())})"
            self.history.append(history_entry)
            return True
        return False
    
    def get_processing_history(self):
        """Получить историю преобразований"""
        return self.history
    
    def reset_image(self):
        """Сброс к исходному изображению"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.image_array = np.array(self.current_image, dtype=np.float32)
            self.history = ["original"]
            return True
        return False
    
    def save_image(self, filename):
        """Сохранение изображения"""
        if self.current_image is not None:
            os.makedirs('saves', exist_ok=True)
            save_path = os.path.join('saves', filename)
            self.current_image.save(save_path)
            return True
        return False
    
    def get_image_base64(self):
        """Получение изображения в base64"""
        if self.current_image is None:
            return None
            
        img_io = io.BytesIO()
        self.current_image.save(img_io, 'PNG')
        img_io.seek(0)
        return base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    def _update_current_image(self):
        """Обновление текущего изображения из массива"""
        self.current_image = Image.fromarray(self.image_array.astype(np.uint8))

processor = ImageProcessor()

def convert_numpy_types(obj):
    """Рекурсивно преобразует NumPy типы в стандартные Python типы"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        info = processor.load_image(file)
        image_data = processor.get_image_base64()
        
        response_data = {
            'success': True,
            'info': convert_numpy_types(info),
            'image': image_data,
            'history': processor.get_processing_history()
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_transform', methods=['POST'])
def apply_transform():
    data = request.json
    transform_type = data.get('type')
    params = data.get('params', {})
    
    try:
        before_array = processor.image_array.copy() if processor.image_array is not None else None
        
        success = processor.apply_transformation(transform_type, **params)
        
        if success:
            image_data = processor.get_image_base64()
            
            diff_map = None
            if before_array is not None and processor.image_array is not None:
                diff_map = processor.get_difference_map(before_array)
            
            response_data = {
                'success': True,
                'image': image_data,
                'history': processor.get_processing_history(),
                'diff_map': diff_map
            }
            
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Transform application failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_image():
    try:
        success = processor.reset_image()
        if success:
            image_data = processor.get_image_base64()
            response_data = {
                'success': True,
                'image': image_data,
                'history': processor.get_processing_history()
            }
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Reset failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save', methods=['POST'])
def save_image():
    data = request.json
    filename = data.get('filename', 'processed_image.png')
    
    try:
        success = processor.save_image(filename)
        if success:
            return jsonify({'success': True, 'filename': filename})
        else:
            return jsonify({'error': 'Save failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('saves', exist_ok=True)
    app.run(debug=True, port=5000)