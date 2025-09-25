import cv2
import numpy as np
import pyautogui
import time
import logging
from datetime import datetime
import os
import json
from threading import Thread
from queue import Queue
import pytesseract
from PIL import Image
import hashlib
import tkinter as tk
from tkinter import messagebox
import re
import glob
import shlex

from calibration_utils import import_reference_templates
from screen_state import OCRToken, ScreenState, analyze_tokens

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_detector.log'),
        logging.StreamHandler()
    ]
)


class RegionSelector:
    """
    Clase para seleccionar visualmente la región de BlueStacks
    """

    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.selecting = False

    def select_region(self):
        """
        Permite al usuario seleccionar un rectángulo en la pantalla
        """
        print("\n" + "="*50)
        print("    SELECCIÓN DE REGIÓN DE BLUESTACKS")
        print("="*50)

        print("\n📍 PASO 1:")
        print("   Mueve el mouse a la ESQUINA SUPERIOR IZQUIERDA de BlueStacks")
        print("   y presiona ENTER cuando estés listo")
        input("\n   👉 Presiona ENTER: ")
        self.start_x, self.start_y = pyautogui.position()
        print(
            f"   ✓ Esquina superior izquierda: ({self.start_x}, {self.start_y})")

        print("\n📍 PASO 2:")
        print("   Mueve el mouse a la ESQUINA INFERIOR DERECHA de BlueStacks")
        print("   y presiona ENTER cuando estés listo")
        input("\n   👉 Presiona ENTER: ")
        self.end_x, self.end_y = pyautogui.position()
        print(f"   ✓ Esquina inferior derecha: ({self.end_x}, {self.end_y})")

        # Calcular región
        x = min(self.start_x, self.end_x)
        y = min(self.start_y, self.end_y)
        width = abs(self.end_x - self.start_x)
        height = abs(self.end_y - self.start_y)

        print("\n" + "="*50)
        print(f"📐 REGIÓN CONFIGURADA:")
        print(f"   Posición: ({x}, {y})")
        print(f"   Tamaño: {width} x {height} píxeles")
        print("="*50)

        # Mostrar preview (con manejo de errores mejorado)
        self.show_preview(x, y, width, height)

        return (x, y, width, height)

    def show_preview(self, x, y, width, height):
        """
        Muestra un preview de la región seleccionada
        """
        try:
            print("\n📸 Capturando preview de la región...")

            # Tomar screenshot de la región
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

            # Dibujar marco
            cv2.rectangle(screenshot_cv, (5, 5),
                          (width-5, height-5), (0, 255, 0), 2)
            cv2.putText(screenshot_cv, "REGION CAPTURADA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Marcar zona inferior donde estará el botón de monedas
            button_y_start = int(height * 0.7)  # 70% inferior
            cv2.rectangle(screenshot_cv, (10, button_y_start),
                          (width-10, height-10), (255, 0, 0), 2)
            cv2.putText(screenshot_cv, "Zona boton monedas", (15, button_y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Guardar preview como imagen
            preview_path = "region_preview.png"
            cv2.imwrite(preview_path, screenshot_cv)
            print(f"✓ Preview guardado en: {preview_path}")

            # Intentar mostrar la ventana con timeout
            print("\n🖼️ Mostrando preview...")
            print("➡️ Presiona CUALQUIER TECLA en la ventana de preview para continuar")
            print("   (Si la ventana no aparece, ciérrala o presiona Ctrl+C)\n")

            window_name = "Region BlueStacks - Presiona cualquier tecla para continuar"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(800, width), min(600, height))
            cv2.imshow(window_name, screenshot_cv)

            # Esperar con timeout de 10 segundos
            key = cv2.waitKey(10000)  # 10 segundos de timeout
            cv2.destroyAllWindows()

            if key == -1:
                print("⏱️ Timeout del preview - continuando...")
            else:
                print("✓ Preview cerrado")

        except Exception as e:
            print(f"⚠️ No se pudo mostrar el preview visual: {e}")
            print("   El preview se guardó en 'region_preview.png'")
            print("   Continuando con la configuración...")


class XDetectorBot:
    def __init__(self, config_file='bot_config.json'):
        """
        Inicializa el bot detector con configuración personalizable
        """
        self.running = False
        self.click_queue = Queue()
        self.last_click_hash = None
        self.click_count = 0
        self.ad_count = 0
        self.last_x_click_time = 0
        self.last_special_state = ScreenState.UNKNOWN
        self.last_special_hash = None
        self.last_special_time = 0.0

        # Configuración por defecto
        self.config = {
            "screenshot_interval": 0.5,
            "confidence_threshold": 0.75,
            "click_delay": 0.5,
            "safe_mode": True,
            "detection_methods": ["template", "contour", "ocr", "color"],
            "templates_folder": "templates",
            "click_offset": {"x": 0, "y": 0},
            # None para pantalla completa o (x, y, width, height)
            "region": None,
            "max_clicks_per_minute": 60,
            "enable_anti_detection": True,
            # Zona inferior donde buscar el botón (70% hacia abajo)
            "money_button_zone": 0.7,
            "wait_after_x": 1.0,  # Espera después de cerrar X
            "auto_click_money": True  # Activar clic automático en botón de monedas
        }

        # Cargar configuración desde archivo si existe
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))

        # Crear carpeta de templates si no existe
        if not os.path.exists(self.config['templates_folder']):
            os.makedirs(self.config['templates_folder'])

        # Cargar templates de X y botón de monedas
        self.load_templates()

        # Configurar PyAutoGUI
        pyautogui.FAILSAFE = self.config['safe_mode']
        pyautogui.PAUSE = 0.1

        logging.info("Bot inicializado con configuración: %s", self.config)

    def load_templates(self):
        """
        Carga templates de imágenes de X y botón de monedas
        """
        self.x_templates = []
        self.money_templates = []

        if not os.path.exists(self.config['templates_folder']):
            return

        template_files = [f for f in os.listdir(self.config['templates_folder'])
                          if f.endswith(('.png', '.jpg', '.jpeg'))]

        for template_file in template_files:
            path = os.path.join(self.config['templates_folder'], template_file)
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                # Clasificar templates
                if 'money' in template_file.lower() or 'coin' in template_file.lower() or 'moneda' in template_file.lower():
                    self.money_templates.append({
                        'name': template_file,
                        'image': template,
                        'shape': template.shape
                    })
                    logging.info(
                        f"Template de monedas cargado: {template_file}")
                else:
                    self.x_templates.append({
                        'name': template_file,
                        'image': template,
                        'shape': template.shape
                    })
                    logging.info(f"Template de X cargado: {template_file}")

        if not self.x_templates:
            logging.warning(
                "No se encontraron templates de X. Usando solo detección por contornos y OCR")

    def setup_region(self):
        """
        Configura la región de captura de BlueStacks
        """
        try:
            selector = RegionSelector()
            region = selector.select_region()

            # Validar región
            if region[2] < 100 or region[3] < 100:
                print("\n⚠️ La región seleccionada es muy pequeña.")
                print("   Asegúrate de seleccionar toda la ventana de BlueStacks.")
                return None

            # Guardar en configuración
            self.config['region'] = region

            # Guardar en archivo
            config_file = 'bluestacks_region.json'
            with open(config_file, 'w') as f:
                json.dump({
                    'region': region,
                    'money_button_zone': self.config['money_button_zone'],
                    'timestamp': datetime.now().isoformat()
                }, f, indent=4)

            print(
                f"\n✅ Configuración guardada exitosamente en '{config_file}'")
            print(f"   Puedes usar 'python bot_detector.py quick' para inicio rápido")

            return region

        except KeyboardInterrupt:
            print("\n\n⚠️ Configuración cancelada por el usuario")
            return None
        except Exception as e:
            print(f"\n❌ Error configurando región: {e}")
            return None

    def take_screenshot(self):
        """
        Toma un screenshot de la región especificada
        """
        try:
            if self.config['region']:
                screenshot = pyautogui.screenshot(region=self.config['region'])
            else:
                screenshot = pyautogui.screenshot()

            # Convertir a formato OpenCV
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

            return screenshot_cv
        except Exception as e:
            logging.error(f"Error tomando screenshot: {e}")
            return None

    def detect_x_template_matching(self, screenshot):
        """
        Detecta X usando template matching con múltiples escalas
        """
        detections = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        for template_data in self.x_templates:
            template = template_data['image']

            # Probar múltiples escalas
            scales = np.linspace(0.5, 1.5, 20)

            for scale in scales:
                # Redimensionar template
                width = int(template.shape[1] * scale)
                height = int(template.shape[0] * scale)

                if width <= 0 or height <= 0:
                    continue

                resized = cv2.resize(template, (width, height))

                # Omitir si el template redimensionado es más grande que la imagen
                if (resized.shape[0] > gray.shape[0] or
                        resized.shape[1] > gray.shape[1]):
                    continue

                # Template matching
                result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)

                # Encontrar ubicaciones con alta confianza
                locations = np.where(
                    result >= self.config['confidence_threshold'])

                for pt in zip(*locations[::-1]):
                    detections.append({
                        'x': pt[0] + width // 2,
                        'y': pt[1] + height // 2,
                        'confidence': result[pt[1], pt[0]],
                        'method': 'template',
                        'type': 'x',
                        'size': (width, height)
                    })

        return detections

    def detect_money_button(self, screenshot):
        """
        Detecta el botón de monedas en la zona inferior
        """
        detections = []
        height, width = screenshot.shape[:2]

        # Definir zona de búsqueda (parte inferior)
        search_zone_y = int(height * self.config['money_button_zone'])
        search_zone = screenshot[search_zone_y:height, :]
        search_zone_gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)

        # 1. Buscar por template si existe
        for template_data in self.money_templates:
            template = template_data['image']
            if (template.shape[0] > search_zone_gray.shape[0] or
                    template.shape[1] > search_zone_gray.shape[1]):
                logging.debug(
                    "Template '%s' omitido: tamaño (%s, %s) mayor que zona (%s, %s)",
                    template_data['name'], template.shape[1], template.shape[0],
                    search_zone_gray.shape[1], search_zone_gray.shape[0])
                continue

            result = cv2.matchTemplate(
                search_zone_gray, template, cv2.TM_CCOEFF_NORMED)

            locations = np.where(result >= 0.7)
            for pt in zip(*locations[::-1]):
                detections.append({
                    'x': pt[0] + template.shape[1] // 2,
                    'y': search_zone_y + pt[1] + template.shape[0] // 2,
                    'confidence': result[pt[1], pt[0]],
                    'method': 'template',
                    'type': 'money'
                })

        # 2. Buscar por OCR (texto que contenga "moneda", "coin", "+200", etc.)
        try:
            pil_image = Image.fromarray(
                cv2.cvtColor(search_zone, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT,
                                             config='--psm 11', lang='spa+eng')

            for i in range(len(data['text'])):
                text = data['text'][i].strip().lower()
                conf = int(data['conf'][i])

                # Buscar palabras clave relacionadas con monedas/rewards
                money_keywords = ['moneda', 'coin', '200', 'reward', 'premio', 'ver', 'anuncio',
                                  'watch', 'ad', 'gratis', 'free', 'ganar', 'earn', '+']

                if any(keyword in text for keyword in money_keywords) and conf > 50:
                    x = data['left'][i] + data['width'][i] // 2
                    y = search_zone_y + data['top'][i] + data['height'][i] // 2

                    detections.append({
                        'x': x,
                        'y': y,
                        'confidence': conf / 100.0,
                        'method': 'ocr',
                        'type': 'money',
                        'text': text
                    })

        except Exception as e:
            logging.debug(f"Error en OCR para botón de monedas: {e}")

        # 3. Buscar botones grandes por color/forma en la zona inferior
        # Buscar rectángulos grandes que podrían ser botones
        edges = cv2.Canny(search_zone_gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Botones grandes
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                if 2.0 <= aspect_ratio <= 6.0:  # Botones rectangulares anchos
                    detections.append({
                        'x': x + w // 2,
                        'y': search_zone_y + y + h // 2,
                        'confidence': 0.5,
                        'method': 'contour',
                        'type': 'money'
                    })

        return detections

    def detect_x_contours(self, screenshot):
        """
        Detecta X usando detección de contornos y formas geométricas
        """
        detections = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Aplicar filtros para mejorar detección
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Encontrar contornos
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filtrar por área
            area = cv2.contourArea(contour)
            if area < 100 or area > 5000:
                continue

            # Aproximar polígono
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Buscar formas que podrían ser X (4-8 vértices, aspecto cuadrado)
            if 4 <= len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                if 0.7 <= aspect_ratio <= 1.3:  # Forma aproximadamente cuadrada
                    # Verificar si tiene líneas diagonales (característica de X)
                    roi = gray[y:y+h, x:x+w]
                    if self.has_diagonal_lines(roi):
                        detections.append({
                            'x': x + w // 2,
                            'y': y + h // 2,
                            'confidence': 0.6,
                            'method': 'contour',
                            'type': 'x',
                            'size': (w, h)
                        })

        return detections

    def has_diagonal_lines(self, roi):
        """
        Verifica si una región tiene líneas diagonales (característica de X)
        """
        if roi.size == 0:
            return False

        # Detectar líneas usando Hough
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=20)

        if lines is None:
            return False

        # Buscar líneas diagonales (ángulos cerca de 45° o 135°)
        diagonal_count = 0
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)

            if (35 <= angle <= 55) or (125 <= angle <= 145):
                diagonal_count += 1

        return diagonal_count >= 2

    def detect_x_ocr(self, screenshot):
        """
        Detecta X usando OCR (reconocimiento de texto)
        """
        detections = []

        # Convertir a PIL para pytesseract
        pil_image = Image.fromarray(
            cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))

        # OCR con configuración para detectar caracteres individuales
        try:
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT,
                                             config='--psm 11 -c tessedit_char_whitelist=Xx×✕╳')

            for i in range(len(data['text'])):
                text = data['text'][i].strip().upper()
                conf = int(data['conf'][i])

                if text in ['X', '×', '✕', '╳'] and conf > 60:
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2

                    detections.append({
                        'x': x,
                        'y': y,
                        'confidence': conf / 100.0,
                        'method': 'ocr',
                        'type': 'x',
                        'size': (data['width'][i], data['height'][i])
                    })
        except Exception as e:
            logging.debug(f"Error en OCR: {e}")

        return detections

    def detect_x_color_based(self, screenshot):
        """
        Detecta X basándose en colores típicos de botones de cierre
        """
        detections = []
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # Rangos de colores comunes para botones de cierre
        color_ranges = [
            # Rojo
            {'lower': np.array([0, 50, 50]),
             'upper': np.array([10, 255, 255])},
            {'lower': np.array([170, 50, 50]),
             'upper': np.array([180, 255, 255])},
            # Gris/Negro
            {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 30, 100])},
            # Blanco
            {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])}
        ]

        for color_range in color_ranges:
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])

            # Operaciones morfológicas para limpiar
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Encontrar contornos
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 <= area <= 2000:  # Tamaño típico de botón
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h

                    if 0.7 <= aspect_ratio <= 1.3:  # Forma cuadrada
                        detections.append({
                            'x': x + w // 2,
                            'y': y + h // 2,
                            'confidence': 0.5,
                            'method': 'color',
                            'type': 'x',
                            'size': (w, h)
                        })

        return detections

    def merge_detections(self, all_detections):
        """
        Fusiona detecciones cercanas y elimina duplicados
        """
        if not all_detections:
            return []

        # Separar por tipo
        x_detections = [d for d in all_detections if d.get('type') == 'x']
        money_detections = [
            d for d in all_detections if d.get('type') == 'money']

        merged = []

        # Fusionar detecciones de X
        if x_detections:
            used = [False] * len(x_detections)

            for i, det1 in enumerate(x_detections):
                if used[i]:
                    continue

                cluster = [det1]
                used[i] = True

                # Buscar detecciones cercanas
                for j, det2 in enumerate(x_detections):
                    if used[j]:
                        continue

                    dist = np.sqrt((det1['x'] - det2['x'])
                                   ** 2 + (det1['y'] - det2['y'])**2)

                    if dist < 30:  # Umbral de distancia
                        cluster.append(det2)
                        used[j] = True

                # Calcular centro del cluster y confianza promedio
                avg_x = sum(d['x'] for d in cluster) / len(cluster)
                avg_y = sum(d['y'] for d in cluster) / len(cluster)
                avg_conf = sum(d['confidence'] for d in cluster) / len(cluster)

                # Bonus de confianza por múltiples métodos
                methods = set(d['method'] for d in cluster)
                confidence_boost = len(methods) * 0.1

                merged.append({
                    'x': int(avg_x),
                    'y': int(avg_y),
                    'confidence': min(avg_conf + confidence_boost, 1.0),
                    'methods': list(methods),
                    'type': 'x',
                    'cluster_size': len(cluster)
                })

        # Agregar detecciones de monedas (sin fusionar por ahora)
        merged.extend(money_detections)

        # Ordenar por confianza
        merged.sort(key=lambda x: x['confidence'], reverse=True)

        return merged

    def _extract_ocr_tokens(self, screenshot):
        """Convert pytesseract raw output into ``OCRToken`` instances."""

        try:
            pil_image = Image.fromarray(
                cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                lang='spa+eng')
        except Exception as exc:
            logging.debug("Error extrayendo OCR: %s", exc)
            return []

        tokens = []
        texts = data.get('text', [])
        n = len(texts)

        for i in range(n):
            text = (texts[i] or '').strip()
            if not text:
                continue

            try:
                confidence_str = data.get('conf', [])[i]
            except IndexError:
                confidence_str = '-1'

            try:
                confidence = float(confidence_str)
            except Exception:
                confidence = -1.0

            try:
                left = int(data.get('left', [0])[i])
                top = int(data.get('top', [0])[i])
                width = int(data.get('width', [0])[i])
                height = int(data.get('height', [0])[i])
            except Exception:
                continue

            tokens.append(OCRToken(
                text=text,
                left=left,
                top=top,
                width=width,
                height=height,
                confidence=confidence,
            ))

        return tokens

    def _select_x_by_zone(self, detections, y_min, y_max):
        """Return the most confident X detection inside vertical bounds."""

        candidates = [
            d for d in detections
            if y_min <= d.get('y', 0) <= y_max
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda d: d.get('confidence', 0))

    def _enqueue_special_action(self, click_type, state, point, frame_shape):
        """Queue a high-priority click derived from contextual analysis."""

        if point is None:
            return False

        height, width = frame_shape[:2]
        x = max(0, min(width - 1, int(point[0])))
        y = max(0, min(height - 1, int(point[1])))

        action_hash = f"{state.value}:{click_type}:{x // 5}:{y // 5}"
        now = time.time()

        if (self.last_special_hash == action_hash and
                now - self.last_special_time < 1.5):
            return False

        queue_payload = {'x': x, 'y': y, 'type': click_type}

        if click_type == 'x':
            detection_hash = hashlib.md5(f"{x}{y}".encode()).hexdigest()
            if detection_hash == self.last_click_hash:
                return False
            self.last_click_hash = detection_hash
        else:
            detection_hash = None

        self.click_queue.put(queue_payload)

        self.last_special_hash = action_hash
        self.last_special_time = now
        self.last_special_state = state

        if click_type == 'x':
            logging.info(
                "[X:%s] Acción guiada en (%s, %s)", state.value, x, y)
        elif click_type == 'continue':
            logging.info("[➡️] Continuar detectado en (%s, %s)", x, y)
        elif click_type == 'money':
            logging.info("[💰] Botón de recompensa detectado en (%s, %s)", x, y)

        return True

    def apply_anti_detection(self):
        """
        Aplica medidas anti-detección para parecer más humano
        """
        if not self.config['enable_anti_detection']:
            return

        # Movimiento aleatorio del mouse
        current_x, current_y = pyautogui.position()
        offset_x = np.random.randint(-5, 6)
        offset_y = np.random.randint(-5, 6)

        # Movimiento con curva bezier para parecer natural
        pyautogui.moveTo(current_x + offset_x, current_y + offset_y,
                         duration=np.random.uniform(0.1, 0.3),
                         tween=pyautogui.easeInOutQuad)

        # Delay variable
        time.sleep(np.random.uniform(0.05, 0.15))

    def click_target(self, x, y, target_type='x'):
        """
        Hace clic en las coordenadas especificadas con comportamiento humano
        """
        try:
            # Convertir coordenadas relativas a absolutas si hay región
            if self.config['region']:
                target_x = self.config['region'][0] + \
                    x + self.config['click_offset']['x']
                target_y = self.config['region'][1] + \
                    y + self.config['click_offset']['y']
            else:
                target_x = x + self.config['click_offset']['x']
                target_y = y + self.config['click_offset']['y']

            # Anti-detección: movimiento natural
            if self.config['enable_anti_detection']:
                # Movimiento en múltiples pasos con curva
                steps = np.random.randint(2, 5)
                current_x, current_y = pyautogui.position()

                for i in range(steps):
                    progress = (i + 1) / steps
                    inter_x = current_x + (target_x - current_x) * progress
                    inter_y = current_y + (target_y - current_y) * progress

                    # Añadir pequeña variación
                    if i < steps - 1:
                        inter_x += np.random.randint(-3, 4)
                        inter_y += np.random.randint(-3, 4)

                    pyautogui.moveTo(inter_x, inter_y,
                                     duration=np.random.uniform(0.05, 0.15))
            else:
                pyautogui.moveTo(target_x, target_y, duration=0.2)

            # Clic con duración variable
            pyautogui.click(duration=np.random.uniform(0.05, 0.15))

            if target_type == 'x':
                self.click_count += 1
                self.last_x_click_time = time.time()
                logging.info(
                    f"[X] Clic en X realizado en ({target_x}, {target_y}) - Total X: {self.click_count}")
            elif target_type == 'money':
                self.ad_count += 1
                logging.info(
                    f"[💰] Clic en botón de monedas en ({target_x}, {target_y}) - Total ads: {self.ad_count}")
            elif target_type == 'continue':
                logging.info(
                    f"[➡️] Clic en botón Continuar en ({target_x}, {target_y})")

            # Delay post-clic
            time.sleep(self.config['click_delay'])

            return True

        except Exception as e:
            logging.error(f"Error al hacer clic: {e}")
            return False

    def click_money_button_fallback(self):
        """
        Hace clic en la zona inferior como fallback si no se detecta el botón
        """
        if not self.config['region']:
            return False

        try:
            # Calcular posición en el centro-inferior de la región
            region_x, region_y, region_width, region_height = self.config['region']

            # Clic en el centro horizontal, 85% vertical
            target_x = region_x + region_width // 2
            target_y = region_y + int(region_height * 0.85)

            logging.info(f"[💰] Intento fallback de clic en zona inferior")

            # Movimiento natural
            pyautogui.moveTo(target_x, target_y, duration=0.3,
                             tween=pyautogui.easeInOutQuad)
            pyautogui.click(duration=0.1)

            self.ad_count += 1
            return True

        except Exception as e:
            logging.error(f"Error en clic fallback: {e}")
            return False

    def process_clicks(self):
        """
        Procesa la cola de clics con límite de velocidad
        """
        clicks_in_minute = []

        while self.running:
            try:
                # Limpiar clics antiguos (más de 1 minuto)
                current_time = time.time()
                clicks_in_minute = [t for t in clicks_in_minute
                                    if current_time - t < 60]

                # Verificar límite de clics
                if len(clicks_in_minute) >= self.config['max_clicks_per_minute']:
                    time.sleep(1)
                    continue

                # Obtener siguiente clic de la cola
                if not self.click_queue.empty():
                    target = self.click_queue.get(timeout=0.5)

                    if target['type'] == 'x':
                        if self.click_target(target['x'], target['y'], 'x'):
                            clicks_in_minute.append(current_time)
                            self.apply_anti_detection()

                            # Esperar y buscar botón de monedas
                            if self.config['auto_click_money']:
                                time.sleep(self.config['wait_after_x'])

                                # Tomar nuevo screenshot para buscar botón
                                screenshot = self.take_screenshot()
                                if screenshot is not None:
                                    money_detections = self.detect_money_button(
                                        screenshot)

                                    if money_detections:
                                        # Clic en el botón de monedas más confiable
                                        best_money = max(
                                            money_detections, key=lambda d: d['confidence'])
                                        self.click_target(
                                            best_money['x'], best_money['y'], 'money')
                                    else:
                                        # Intento fallback
                                        logging.info(
                                            "No se detectó botón de monedas, usando fallback")
                                        self.click_money_button_fallback()

                    elif target['type'] == 'money':
                        if self.click_target(target['x'], target['y'], 'money'):
                            clicks_in_minute.append(current_time)
                            self.apply_anti_detection()
                    elif target['type'] == 'continue':
                        if self.click_target(target['x'], target['y'], 'continue'):
                            clicks_in_minute.append(current_time)
                            self.apply_anti_detection()
                else:
                    time.sleep(0.1)

            except Exception as e:
                logging.error(f"Error procesando clics: {e}")

    def detection_loop(self):
        """
        Loop principal de detección
        """
        while self.running:
            try:
                start_time = time.time()

                # Tomar screenshot
                screenshot = self.take_screenshot()
                if screenshot is None:
                    time.sleep(self.config['screenshot_interval'])
                    continue

                ocr_tokens = self._extract_ocr_tokens(screenshot)
                state, state_point = analyze_tokens(ocr_tokens, screenshot.shape)

                if state == ScreenState.COUNTDOWN:
                    if self.last_special_state != ScreenState.COUNTDOWN:
                        logging.info(
                            "⏳ Contador de recompensa detectado. Esperando finalización...")
                    self.last_special_state = ScreenState.COUNTDOWN
                    elapsed = time.time() - start_time
                    time.sleep(max(
                        0, self.config['screenshot_interval'] - elapsed))
                    continue

                all_detections = []

                if 'template' in self.config['detection_methods'] and self.x_templates:
                    all_detections.extend(
                        self.detect_x_template_matching(screenshot))

                if 'contour' in self.config['detection_methods']:
                    all_detections.extend(self.detect_x_contours(screenshot))

                if 'ocr' in self.config['detection_methods']:
                    all_detections.extend(self.detect_x_ocr(screenshot))

                if 'color' in self.config['detection_methods']:
                    all_detections.extend(self.detect_x_color_based(screenshot))

                merged_detections = self.merge_detections(all_detections)
                x_detections = [
                    d for d in merged_detections if d.get('type') == 'x']

                handled_special = False
                height = screenshot.shape[0]

                if state == ScreenState.CONTINUE_ARROW and state_point:
                    handled_special = True
                    self._enqueue_special_action(
                        'continue', ScreenState.CONTINUE_ARROW, state_point, screenshot.shape)
                elif state == ScreenState.MAIN_REWARD_BUTTON and state_point:
                    handled_special = True
                    self._enqueue_special_action(
                        'money', ScreenState.MAIN_REWARD_BUTTON, state_point, screenshot.shape)
                elif state == ScreenState.REWARD_COMPLETE:
                    handled_special = True
                    reward_x = self._select_x_by_zone(
                        x_detections, 0, int(height * 0.4))
                    if reward_x:
                        self._enqueue_special_action(
                            'x', ScreenState.REWARD_COMPLETE,
                            (reward_x['x'], reward_x['y']), screenshot.shape)
                else:
                    bottom_x = self._select_x_by_zone(
                        x_detections, int(height * 0.55), height)
                    if bottom_x:
                        handled_special = self._enqueue_special_action(
                            'x', ScreenState.AD_DISMISS,
                            (bottom_x['x'], bottom_x['y']), screenshot.shape)

                if handled_special:
                    elapsed = time.time() - start_time
                    time.sleep(max(
                        0, self.config['screenshot_interval'] - elapsed))
                    continue

                self.last_special_state = ScreenState.UNKNOWN

                for detection in x_detections:
                    if detection['confidence'] >= self.config['confidence_threshold']:
                        detection_hash = hashlib.md5(
                            f"{detection['x']}{detection['y']}".encode()
                        ).hexdigest()

                        if detection_hash != self.last_click_hash:
                            self.click_queue.put({
                                'x': detection['x'],
                                'y': detection['y'],
                                'type': 'x'
                            })
                            self.last_click_hash = detection_hash

                            logging.info(
                                "[X] Detectada en (%s, %s) con confianza %.2f usando métodos: %s",
                                detection['x'], detection['y'],
                                detection['confidence'], detection.get('methods'))

                            break

                # Mantener intervalo de screenshots
                elapsed = time.time() - start_time
                sleep_time = max(
                    0, self.config['screenshot_interval'] - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                logging.error(f"Error en loop de detección: {e}")
                time.sleep(1)

    def start(self):
        """
        Inicia el bot
        """
        if self.running:
            logging.warning("El bot ya está en ejecución")
            return

        self.running = True

        # Iniciar thread de procesamiento de clics
        click_thread = Thread(target=self.process_clicks, daemon=True)
        click_thread.start()

        # Iniciar thread de detección
        detection_thread = Thread(target=self.detection_loop, daemon=True)
        detection_thread.start()

        logging.info("Bot iniciado correctamente")
        logging.info(f"Buscando X y botones de monedas...")

        try:
            while self.running:
                time.sleep(1)
                # Mostrar estadísticas cada 30 segundos
                if int(time.time()) % 30 == 0:
                    logging.info(
                        f"📊 Estadísticas - X cerradas: {self.click_count}, Ads vistos: {self.ad_count}")
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """
        Detiene el bot
        """
        logging.info("Deteniendo bot...")
        self.running = False
        time.sleep(2)
        logging.info(
            f"Bot detenido. X cerradas: {self.click_count}, Ads completados: {self.ad_count}")

    def _parse_reference_input(self, raw: str) -> list[str]:
        parts = []

        try:
            tokens = shlex.split(raw)
        except ValueError as exc:
            raise ValueError(f"no se pudieron interpretar las rutas: {exc}") from exc

        for token in tokens:
            for candidate in token.split(','):
                path = candidate.strip()
                if path:
                    parts.append(path)

        if not parts:
            raise ValueError("no se ingresaron rutas válidas")

        return parts

    def calibrate_x(self):
        """Import X templates from provided reference images."""

        print("\n" + "=" * 50)
        print("    CALIBRACIÓN DE TEMPLATES DE X (MODO ARCHIVO)")
        print("=" * 50)

        folder = self.config['templates_folder']
        os.makedirs(folder, exist_ok=True)

        suggested = sorted(
            glob.glob(os.path.join(folder, "reference", "x*.png"))
            + glob.glob(os.path.join(folder, "reference", "*continuar*.png"))
        )

        if suggested:
            print("\n📂 Se detectaron imágenes de referencia potenciales:")
            for path in suggested:
                print(f"   • {path}")

        print("\n📌 Instrucciones:")
        print("   1. Usa las capturas provistas del flujo 'X/Continuar'.")
        print("   2. Ingresa la ruta del archivo (o varias separadas por coma).")
        print("   3. Los templates se copiarán a la carpeta de trabajo.")

        raw_input = input("\n👉 Ruta(s) de imágenes: ").strip()

        if not raw_input:
            print("\n⚠️ Calibración cancelada: no se ingresaron rutas.")
            return

        try:
            paths = self._parse_reference_input(raw_input)
        except ValueError as exc:
            print(f"\n❌ Error interpretando rutas: {exc}")
            return

        try:
            saved = import_reference_templates(paths, folder, "x_template")
        except FileNotFoundError as exc:
            print(f"\n❌ {exc}")
            return
        except ValueError as exc:
            print(f"\n❌ Error procesando imágenes: {exc}")
            return

        self.load_templates()

        print("\n" + "=" * 50)
        print("✅ CALIBRACIÓN COMPLETADA")
        print(f"   Templates importados: {len(saved)}")
        print("   Archivos:")
        for name in saved:
            print(f"   • {name}")
        print("=" * 50)

    def calibrate_money(self):
        """Import money button templates from reference screenshots."""

        print("\n" + "=" * 50)
        print("    CALIBRACIÓN DE BOTÓN '+200 MONEDAS' (MODO ARCHIVO)")
        print("=" * 50)

        folder = self.config['templates_folder']
        os.makedirs(folder, exist_ok=True)

        suggested = sorted(
            glob.glob(os.path.join(folder, "reference", "money*.png"))
            + glob.glob(os.path.join(folder, "reference", "*monedas*.png"))
        )

        if suggested:
            print("\n📂 Se detectaron imágenes de referencia potenciales:")
            for path in suggested:
                print(f"   • {path}")

        print("\n📌 Instrucciones:")
        print("   1. Selecciona la captura con '+200 monedas por ver un anuncio'.")
        print("   2. Ingresa la ruta del archivo (o varias separadas por coma).")
        print("   3. Los templates se copiarán listos para usarse.")

        raw_input = input("\n👉 Ruta(s) de imágenes: ").strip()

        if not raw_input:
            print("\n⚠️ Calibración cancelada: no se ingresaron rutas.")
            return

        try:
            paths = self._parse_reference_input(raw_input)
        except ValueError as exc:
            print(f"\n❌ Error interpretando rutas: {exc}")
            return

        try:
            saved = import_reference_templates(paths, folder, "money_button")
        except FileNotFoundError as exc:
            print(f"\n❌ {exc}")
            return
        except ValueError as exc:
            print(f"\n❌ Error procesando imágenes: {exc}")
            return

        self.load_templates()

        print("\n" + "=" * 50)
        print("✅ CALIBRACIÓN COMPLETADA")
        print(f"   Templates importados: {len(saved)}")
        print("   Archivos:")
        for name in saved:
            print(f"   • {name}")
        print("=" * 50)


def main_menu():
    """
    Menú principal del bot
    """
    print("\n" + "="*50)
    print("   🤖 BOT DETECTOR DE X + MONEDAS")
    print("="*50)

    bot = XDetectorBot()

    # Verificar si existe configuración previa
    if os.path.exists('bluestacks_region.json'):
        try:
            with open('bluestacks_region.json', 'r') as f:
                saved_config = json.load(f)
                bot.config['region'] = saved_config['region']
                print("\n✅ Configuración de región cargada automáticamente")
                print(f"   Región: {saved_config['region']}")
        except:
            print("\n⚠️ No se pudo cargar la configuración guardada")

    while True:
        print("\n📋 MENÚ PRINCIPAL:")
        print("1. 🎯 Configurar región de BlueStacks")
        print("2. 📸 Calibrar templates de X (click en la ventana)")
        print("3. 💰 Calibrar botón de monedas")
        print("4. ▶️  Iniciar Bot")
        print("5. 📊 Ver configuración actual")
        print("6. ❌ Salir")

        choice = input("\nSelecciona opción (1-6): ")

        if choice == '1':
            result = bot.setup_region()
            if result:
                print("✓ Región configurada correctamente")
            else:
                print("⚠️ No se pudo configurar la región")

        elif choice == '2':
            if not bot.config['region']:
                print("\n⚠️ Primero configura la región (opción 1)")
            else:
                bot.calibrate_x()

        elif choice == '3':
            if not bot.config['region']:
                print("\n⚠️ Primero configura la región (opción 1)")
            else:
                bot.calibrate_money()

        elif choice == '4':
            if not bot.config['region']:
                print("\n⚠️ Primero configura la región (opción 1)")
                print("   La región es necesaria para que el bot funcione correctamente")
            else:
                print("\n" + "="*50)
                print("🚀 INICIANDO BOT")
                print("="*50)
                print("\n📌 Instrucciones:")
                print("   - El bot buscará X y las cerrará automáticamente")
                print(
                    "   - Después esperará 1 segundo y hará clic en el botón de monedas")
                print("   - Presiona Ctrl+C para detener\n")
                print("="*50 + "\n")

                try:
                    bot.start()
                except KeyboardInterrupt:
                    bot.stop()
                    print("\n\n✅ Bot detenido correctamente")

        elif choice == '5':
            print("\n" + "="*50)
            print("📊 CONFIGURACIÓN ACTUAL:")
            print("="*50)
            if bot.config['region']:
                x, y, w, h = bot.config['region']
                print(f"✓ Región configurada: ")
                print(f"  - Posición: ({x}, {y})")
                print(f"  - Tamaño: {w}x{h} píxeles")
            else:
                print("✗ Región: No configurada")

            print(f"✓ Templates de X cargados: {len(bot.x_templates)}")
            print(
                f"✓ Templates de monedas cargados: {len(bot.money_templates)}")
            print(
                f"✓ Intervalo de captura: {bot.config['screenshot_interval']}s")
            print(
                f"✓ Métodos de detección: {', '.join(bot.config['detection_methods'])}")
            print(
                f"✓ Auto-clic en monedas: {'Sí' if bot.config['auto_click_money'] else 'No'}")
            print(
                f"✓ Tiempo de espera después de X: {bot.config['wait_after_x']}s")
            print("="*50)

        elif choice == '6':
            print("\n👋 ¡Hasta luego!")
            print("   Gracias por usar el Bot Detector")
            break

        else:
            print("\n❌ Opción no válida. Por favor selecciona 1-6")


if __name__ == "__main__":
    import sys

    print("\n" + "🤖"*20)
    print("    BOT DETECTOR X + MONEDAS v2.0")
    print("🤖"*20)

    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            # Inicio rápido con configuración guardada
            if os.path.exists('bluestacks_region.json'):
                try:
                    with open('bluestacks_region.json', 'r') as f:
                        config = json.load(f)

                    bot = XDetectorBot()
                    bot.config['region'] = config['region']

                    print("\n✅ Configuración cargada exitosamente")
                    print(f"   Región: {config['region']}")
                    print("\n🚀 INICIANDO BOT EN MODO RÁPIDO")
                    print("   Presiona Ctrl+C para detener\n")
                    print("="*50 + "\n")

                    bot.start()

                except KeyboardInterrupt:
                    print("\n\n✅ Bot detenido por el usuario")
                except Exception as e:
                    print(f"\n❌ Error al cargar configuración: {e}")
                    print("   Ejecuta sin argumentos para configurar")
            else:
                print("\n⚠️ No hay configuración guardada")
                print("   Ejecuta primero sin argumentos para configurar la región")
                print("\n   Uso: python bot_detector.py")
        else:
            print("\n📖 USO:")
            print("   python bot_detector.py        # Menú interactivo")
            print("   python bot_detector.py quick  # Inicio rápido con config guardada")
    else:
        # Menú interactivo
        try:
            main_menu()
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print("   Por favor, reporta este error.")
