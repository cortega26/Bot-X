import cv2
import numpy as np
import pyautogui
import time
import logging
from datetime import datetime
import os
import json
from threading import Thread
from queue import Queue, Empty
import pytesseract
from PIL import Image
import hashlib
import re

# ============================
# Configuración de logging
# ============================
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

        # Mostrar preview
        self.show_preview(x, y, width, height)

        return (x, y, width, height)

    def show_preview(self, x, y, width, height):
        """
        Muestra un preview de la región seleccionada
        """
        try:
            print("\n📸 Capturando preview de la región...")
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

            # Dibujar marco
            cv2.rectangle(screenshot_cv, (5, 5),
                          (width-5, height-5), (0, 255, 0), 2)
            cv2.putText(screenshot_cv, "REGION CAPTURADA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Marcar zona inferior donde estará el botón de monedas
            button_y_start = int(height * 0.7)
            cv2.rectangle(screenshot_cv, (10, button_y_start),
                          (width-10, height-10), (255, 0, 0), 2)
            cv2.putText(screenshot_cv, "Zona boton monedas", (15, button_y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            preview_path = "region_preview.png"
            cv2.imwrite(preview_path, screenshot_cv)
            print(f"✓ Preview guardado en: {preview_path}")

            window_name = "Region BlueStacks - Presiona cualquier tecla para continuar"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(800, width), min(600, height))
            cv2.imshow(window_name, screenshot_cv)
            key = cv2.waitKey(10000)  # timeout 10s
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

        # Configuración por defecto
        self.config = {
            "screenshot_interval": 0.5,
            "confidence_threshold": 0.75,
            "click_delay": 0.5,
            "safe_mode": True,
            "detection_methods": ["template", "contour", "ocr", "color"],
            "templates_folder": "templates",
            "click_offset": {"x": 0, "y": 0},
            "region": None,
            "max_clicks_per_minute": 60,
            "enable_anti_detection": True,
            "money_button_zone": 0.7,
            "wait_after_x": 1.0,
            "auto_click_money": True
        }

        # Cargar configuración desde archivo si existe
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    self.config.update(json.load(f))
            except Exception as e:
                logging.warning(f"No se pudo leer '{config_file}': {e}")

        # Crear carpeta de templates si no existe
        os.makedirs(self.config['templates_folder'], exist_ok=True)

        # Cargar templates de X y botón de monedas
        self.load_templates()

        # Configurar PyAutoGUI
        pyautogui.FAILSAFE = self.config['safe_mode']
        pyautogui.PAUSE = 0.1

        logging.info("Bot inicializado con configuración: %s", self.config)

    # ----------------------
    # Carga de templates
    # ----------------------
    def load_templates(self):
        self.x_templates = []
        self.money_templates = []

        folder = self.config['templates_folder']
        if not os.path.exists(folder):
            return

        template_files = [f for f in os.listdir(folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for template_file in template_files:
            path = os.path.join(folder, template_file)
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue

            name_lower = template_file.lower()
            if any(k in name_lower for k in ['money', 'coin', 'moneda']):
                self.money_templates.append(
                    {'name': template_file, 'image': template, 'shape': template.shape})
                logging.info(f"Template de monedas cargado: {template_file}")
            else:
                self.x_templates.append(
                    {'name': template_file, 'image': template, 'shape': template.shape})
                logging.info(f"Template de X cargado: {template_file}")

        if not self.x_templates:
            logging.warning(
                "No se encontraron templates de X. Usando detección por contornos/OCR/color.")

    # ----------------------
    # Captura de pantalla
    # ----------------------
    def take_screenshot(self):
        try:
            if self.config['region']:
                screenshot = pyautogui.screenshot(region=self.config['region'])
            else:
                screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            return screenshot_cv
        except Exception as e:
            logging.error(f"Error tomando screenshot: {e}")
            return None

    # ----------------------
    # Detecciones (X)
    # ----------------------
    def detect_x_template_matching(self, screenshot):
        detections = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        for template_data in self.x_templates:
            template = template_data['image']
            scales = np.linspace(0.5, 1.5, 20)
            for scale in scales:
                width = int(template.shape[1] * scale)
                height = int(template.shape[0] * scale)
                if width <= 0 or height <= 0:
                    continue
                resized = cv2.resize(template, (width, height))
                result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                locations = np.where(
                    result >= self.config['confidence_threshold'])
                for pt in zip(*locations[::-1]):
                    detections.append({
                        'x': pt[0] + width // 2,
                        'y': pt[1] + height // 2,
                        'confidence': float(result[pt[1], pt[0]]),
                        'method': 'template',
                        'type': 'x',
                        'size': (width, height)
                    })
        return detections

    def detect_x_contours(self, screenshot):
        detections = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > 5000:
                continue
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if 4 <= len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.7 <= aspect_ratio <= 1.3:
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
        if roi.size == 0:
            return False
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=20)
        if lines is None:
            return False
        diagonal_count = 0
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            if (35 <= angle <= 55) or (125 <= angle <= 145):
                diagonal_count += 1
        return diagonal_count >= 2

    def detect_x_ocr(self, screenshot):
        detections = []
        pil_image = Image.fromarray(
            cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
        try:
            data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                config='--psm 11 -c tessedit_char_whitelist=Xx×✕╳'
            )
            n = len(data['text'])
            for i in range(n):
                text = (data['text'][i] or '').strip().upper()
                conf_str = data['conf'][i] if i < len(data['conf']) else '-1'
                try:
                    conf = float(conf_str)
                except Exception:
                    conf = -1.0
                if text in ['X', '×', '✕', '╳'] and conf > 60:
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    detections.append({
                        'x': int(x),
                        'y': int(y),
                        'confidence': min(conf / 100.0, 0.99),
                        'method': 'ocr',
                        'type': 'x',
                        'size': (data['width'][i], data['height'][i])
                    })
        except Exception as e:
            logging.debug(f"Error en OCR: {e}")
        return detections

    def detect_x_color_based(self, screenshot):
        detections = []
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        color_ranges = [
            {'lower': np.array([0, 50, 50]),
             'upper': np.array([10, 255, 255])},
            {'lower': np.array([170, 50, 50]),
             'upper': np.array([180, 255, 255])},
            {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 30, 100])},
            {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])}
        ]
        for cr in color_ranges:
            mask = cv2.inRange(hsv, cr['lower'], cr['upper'])
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 <= area <= 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.7 <= aspect_ratio <= 1.3:
                        detections.append({
                            'x': x + w // 2,
                            'y': y + h // 2,
                            'confidence': 0.5,
                            'method': 'color',
                            'type': 'x',
                            'size': (w, h)
                        })
        return detections

    # ----------------------
    # Detección de botón de monedas
    # ----------------------
    def detect_money_button(self, screenshot):
        detections = []
        height, width = screenshot.shape[:2]
        search_zone_y = int(height * self.config['money_button_zone'])
        search_zone = screenshot[search_zone_y:height, :]
        search_zone_gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)

        # 1) Template matching (si existen)
        for template_data in self.money_templates:
            template = template_data['image']
            result = cv2.matchTemplate(
                search_zone_gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.7)
            for pt in zip(*locations[::-1]):
                detections.append({
                    'x': pt[0] + template.shape[1] // 2,
                    'y': search_zone_y + pt[1] + template.shape[0] // 2,
                    'confidence': float(result[pt[1], pt[0]]),
                    'method': 'template',
                    'type': 'money'
                })

        # 2) OCR
        try:
            pil_image = Image.fromarray(
                cv2.cvtColor(search_zone, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT,
                                             config='--psm 11', lang='spa+eng')
            money_keywords = ['moneda', 'coin', '200', 'reward', 'premio', 'ver', 'anuncio',
                              'watch', 'ad', 'gratis', 'free', 'ganar', 'earn', '+']
            n = len(data['text'])
            for i in range(n):
                text = (data['text'][i] or '').strip().lower()
                conf_str = data['conf'][i] if i < len(data['conf']) else '-1'
                try:
                    conf = float(conf_str)
                except Exception:
                    conf = -1.0
                if conf > 50 and any(k in text for k in money_keywords):
                    x = data['left'][i] + data['width'][i] // 2
                    y = search_zone_y + data['top'][i] + data['height'][i] // 2
                    detections.append({
                        'x': int(x),
                        'y': int(y),
                        'confidence': min(conf / 100.0, 0.95),
                        'method': 'ocr',
                        'type': 'money',
                        'text': text
                    })
        except Exception as e:
            logging.debug(f"Error en OCR para botón de monedas: {e}")

        # 3) Contornos (botones grandes)
        edges = cv2.Canny(search_zone_gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 2.0 <= aspect_ratio <= 6.0:
                    detections.append({
                        'x': x + w // 2,
                        'y': search_zone_y + y + h // 2,
                        'confidence': 0.5,
                        'method': 'contour',
                        'type': 'money'
                    })
        return detections

    # ----------------------
    # Fusión de detecciones
    # ----------------------
    def merge_detections(self, all_detections):
        if not all_detections:
            return []
        x_detections = [d for d in all_detections if d.get('type') == 'x']
        money_detections = [
            d for d in all_detections if d.get('type') == 'money']
        merged = []

        if x_detections:
            used = [False] * len(x_detections)
            for i, det1 in enumerate(x_detections):
                if used[i]:
                    continue
                cluster = [det1]
                used[i] = True
                for j, det2 in enumerate(x_detections):
                    if used[j]:
                        continue
                    dist = np.hypot(det1['x'] - det2['x'],
                                    det1['y'] - det2['y'])
                    if dist < 30:
                        cluster.append(det2)
                        used[j] = True
                avg_x = int(sum(d['x'] for d in cluster) / len(cluster))
                avg_y = int(sum(d['y'] for d in cluster) / len(cluster))
                avg_conf = sum(float(d['confidence'])
                               for d in cluster) / len(cluster)
                methods = set(d.get('method', 'unknown') for d in cluster)
                confidence_boost = len(methods) * 0.1
                merged.append({
                    'x': avg_x,
                    'y': avg_y,
                    'confidence': min(avg_conf + confidence_boost, 1.0),
                    'methods': list(methods),
                    'type': 'x',
                    'cluster_size': len(cluster)
                })

        merged.extend(money_detections)
        merged.sort(key=lambda x: x['confidence'], reverse=True)
        return merged

    # ----------------------
    # Anti-detección y click
    # ----------------------
    def apply_anti_detection(self):
        if not self.config['enable_anti_detection']:
            return
        current_x, current_y = pyautogui.position()
        offset_x = np.random.randint(-5, 6)
        offset_y = np.random.randint(-5, 6)
        pyautogui.moveTo(current_x + offset_x, current_y + offset_y,
                         duration=np.random.uniform(0.1, 0.3),
                         tween=pyautogui.easeInOutQuad)
        time.sleep(np.random.uniform(0.05, 0.15))

    def click_target(self, x, y, target_type='x'):
        try:
            if self.config['region']:
                target_x = self.config['region'][0] + \
                    x + self.config['click_offset']['x']
                target_y = self.config['region'][1] + \
                    y + self.config['click_offset']['y']
            else:
                target_x = x + self.config['click_offset']['x']
                target_y = y + self.config['click_offset']['y']

            if self.config['enable_anti_detection']:
                steps = np.random.randint(2, 5)
                current_x, current_y = pyautogui.position()
                for i in range(steps):
                    progress = (i + 1) / steps
                    inter_x = current_x + (target_x - current_x) * progress
                    inter_y = current_y + (target_y - current_y) * progress
                    if i < steps - 1:
                        inter_x += np.random.randint(-3, 4)
                        inter_y += np.random.randint(-3, 4)
                    pyautogui.moveTo(inter_x, inter_y,
                                     duration=np.random.uniform(0.05, 0.15))
            else:
                pyautogui.moveTo(target_x, target_y, duration=0.2)

            pyautogui.click(duration=np.random.uniform(0.05, 0.15))

            if target_type == 'x':
                self.click_count += 1
                self.last_x_click_time = time.time()
                logging.info(
                    f"[X] Clic en X en ({target_x}, {target_y}) - Total X: {self.click_count}")
            elif target_type == 'money':
                self.ad_count += 1
                logging.info(
                    f"[💰] Clic en botón de monedas en ({target_x}, {target_y}) - Total ads: {self.ad_count}")

            time.sleep(self.config['click_delay'])
            return True
        except Exception as e:
            logging.error(f"Error al hacer clic: {e}")
            return False

    def click_money_button_fallback(self):
        if not self.config['region']:
            return False
        try:
            region_x, region_y, region_width, region_height = self.config['region']
            target_x = region_x + region_width // 2
            target_y = region_y + int(region_height * 0.85)
            logging.info(f"[💰] Intento fallback de clic en zona inferior")
            pyautogui.moveTo(target_x, target_y, duration=0.3,
                             tween=pyautogui.easeInOutQuad)
            pyautogui.click(duration=0.1)
            self.ad_count += 1
            return True
        except Exception as e:
            logging.error(f"Error en clic fallback: {e}")
            return False

    # ----------------------
    # Cola de clics (thread)
    # ----------------------
    def process_clicks(self):
        clicks_in_minute = []
        while self.running:
            try:
                current_time = time.time()
                clicks_in_minute = [
                    t for t in clicks_in_minute if current_time - t < 60]

                if len(clicks_in_minute) >= self.config['max_clicks_per_minute']:
                    time.sleep(0.25)
                    continue

                try:
                    target = self.click_queue.get(timeout=0.2)
                except Empty:
                    time.sleep(0.05)
                    continue

                if target['type'] == 'x':
                    if self.click_target(target['x'], target['y'], 'x'):
                        clicks_in_minute.append(time.time())
                        self.apply_anti_detection()
                        if self.config['auto_click_money']:
                            time.sleep(self.config['wait_after_x'])
                            screenshot = self.take_screenshot()
                            if screenshot is not None:
                                money_detections = self.detect_money_button(
                                    screenshot)
                                if money_detections:
                                    best_money = max(
                                        money_detections, key=lambda d: d['confidence'])
                                    self.click_target(
                                        best_money['x'], best_money['y'], 'money')
                                else:
                                    logging.info(
                                        "No se detectó botón de monedas, usando fallback")
                                    self.click_money_button_fallback()

                elif target['type'] == 'money':
                    if self.click_target(target['x'], target['y'], 'money'):
                        clicks_in_minute.append(time.time())
                        self.apply_anti_detection()

            except Exception as e:
                logging.error(f"Error procesando clics: {e}")

    # ----------------------
    # Loop de detección (thread)
    # ----------------------
    def detection_loop(self):
        while self.running:
            try:
                start_time = time.time()
                screenshot = self.take_screenshot()
                if screenshot is None:
                    time.sleep(self.config['screenshot_interval'])
                    continue

                all_detections = []

                time_since_last_x = time.time() - self.last_x_click_time
                if time_since_last_x > 3:
                    if 'template' in self.config['detection_methods'] and self.x_templates:
                        all_detections.extend(
                            self.detect_x_template_matching(screenshot))
                    if 'contour' in self.config['detection_methods']:
                        all_detections.extend(
                            self.detect_x_contours(screenshot))
                    if 'ocr' in self.config['detection_methods']:
                        all_detections.extend(self.detect_x_ocr(screenshot))
                    if 'color' in self.config['detection_methods']:
                        all_detections.extend(
                            self.detect_x_color_based(screenshot))

                merged_detections = self.merge_detections(all_detections)
                x_detections = [
                    d for d in merged_detections if d.get('type') == 'x']

                for detection in x_detections:
                    if float(detection['confidence']) >= float(self.config['confidence_threshold']):
                        detection_hash = hashlib.md5(
                            f"{detection['x']}{detection['y']}".encode()).hexdigest()
                        if detection_hash != self.last_click_hash:
                            self.click_queue.put(
                                {'x': detection['x'], 'y': detection['y'], 'type': 'x'})
                            self.last_click_hash = detection_hash
                            logging.info(
                                f"[X] Detectada en ({detection['x']}, {detection['y']}) "
                                f"conf {detection['confidence']:.2f} métodos: {detection.get('methods')}")
                            break  # Solo la primera de alta confianza

                elapsed = time.time() - start_time
                sleep_time = max(
                    0, self.config['screenshot_interval'] - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Error en loop de detección: {e}")
                time.sleep(1)

    # ----------------------
    # Control de ejecución
    # ----------------------
    def start(self):
        if self.running:
            logging.warning("El bot ya está en ejecución")
            return
        self.running = True
        click_thread = Thread(target=self.process_clicks, daemon=True)
        click_thread.start()
        detection_thread = Thread(target=self.detection_loop, daemon=True)
        detection_thread.start()
        logging.info("Bot iniciado correctamente")
        logging.info("Buscando X y botones de monedas...")
        try:
            while self.running:
                time.sleep(1)
                if int(time.time()) % 30 == 0:
                    logging.info(
                        f"📊 Estadísticas - X cerradas: {self.click_count}, Ads vistos: {self.ad_count}")
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logging.info("Deteniendo bot...")
        self.running = False
        time.sleep(2)
        logging.info(
            f"Bot detenido. X cerradas: {self.click_count}, Ads completados: {self.ad_count}")

    # ----------------------
    # NUEVO: Calibración con CLICK (sin depender de teclas en la ventana)
    # ----------------------
    def _compute_scale(self, w, h, max_w=1200, max_h=900):
        scale = min(1.0, min(max_w / float(w) if w else 1.0,
                    max_h / float(h) if h else 1.0))
        return scale

    def calibrate_x(self):
        """
        Modo de calibración para capturar templates de X.
        AHORA con interacción por CLICK dentro de la ventana de previsualización:
          - Click IZQUIERDO sobre la X en la ventana: guarda un template 50x50.
          - Tecla Q o ESC: salir.
        """
        print("\n" + "="*50)
        print("    CALIBRACIÓN DE TEMPLATES DE X (MODO CLICK)")
        print("="*50)
        if not self.config['region']:
            print("\n⚠️ Primero configura la región (opción 1)")
            return

        folder = self.config['templates_folder']
        os.makedirs(folder, exist_ok=True)

        existing = [f for f in os.listdir(
            folder) if f.lower().startswith('x_template')]
        template_count = len(existing)
        captures = 0

        window_name = "Calibración de X — Click para capturar | Q/Esc para salir"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Escalado para pantallas pequeñas
        _, _, W, H = self.config['region']
        scale = self._compute_scale(W, H)

        last_frame = [None]  # referencia mutable para el callback

        def draw_hud(img):
            cv2.putText(img, "CLICK: capturar 50x50  |  Q/Esc: salir", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Capturas: {captures}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        def on_mouse(event, x, y, flags, param):
            nonlocal captures, template_count
            if event == cv2.EVENT_LBUTTONDOWN and last_frame[0] is not None:
                # Mapear coordenadas de ventana -> imagen original
                x_img = int(x / scale)
                y_img = int(y / scale)
                size = 50
                x1 = max(0, x_img - size // 2)
                y1 = max(0, y_img - size // 2)
                x2 = min(last_frame[0].shape[1], x_img + size // 2)
                y2 = min(last_frame[0].shape[0], y_img + size // 2)
                if x2 > x1 and y2 > y1:
                    roi = last_frame[0][y1:y2, x1:x2]
                    if roi.shape[0] >= 24 and roi.shape[1] >= 24:
                        template_count += 1
                        captures += 1
                        filename = f"x_template_{template_count}.png"
                        filepath = os.path.join(folder, filename)
                        cv2.imwrite(filepath, roi)
                        logging.info(
                            f"✅ Template #{captures} guardado: {filename}")
                    else:
                        logging.info(
                            "Región demasiado pequeña; ignora captura")

        cv2.setMouseCallback(window_name, on_mouse)

        print("\n🎯 Nuevo flujo:")
        print("   1) Mira la ventana de calibración (se actualiza en tiempo real)")
        print("   2) Haz CLICK IZQUIERDO SOBRE LA X dentro de la ventana para guardar un template")
        print("   3) Presiona Q o Esc para terminar")
        print("\n💡 Consejo: Si no ves nada, mueve la ventana para que no tape BlueStacks.")

        try:
            while True:
                frame = self.take_screenshot()
                if frame is None:
                    # evitar bucle apretado si falla la captura
                    time.sleep(0.05)
                    continue
                last_frame[0] = frame
                display = frame.copy()
                draw_hud(display)
                if scale != 1.0:
                    disp = cv2.resize(
                        display, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
                else:
                    disp = display
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
            cv2.destroyAllWindows()
            self.load_templates()
            print("\n" + "="*50)
            print(f"✅ CALIBRACIÓN COMPLETADA")
            print(f"   Templates capturados ahora: {captures}")
            print(
                f"   Total de templates de X cargados: {len(self.x_templates)}")
            print("="*50)
        except Exception as e:
            cv2.destroyAllWindows()
            print(f"\n❌ Error durante calibración: {e}")

    def calibrate_money(self):
        """
        Modo de calibración para capturar templates del botón de monedas.
        CLICK IZQUIERDO sobre el botón en la ventana para guardar una región 150x80.
        Q/Esc para salir.
        """
        print("\n=== CALIBRACIÓN DE BOTÓN DE MONEDAS (MODO CLICK) ===")
        if not self.config['region']:
            print("\n⚠️ Primero configura la región (opción 1)")
            return

        folder = self.config['templates_folder']
        os.makedirs(folder, exist_ok=True)
        template_count = len(
            [f for f in os.listdir(folder) if 'money' in f.lower()])

        window_name = "Calibración Monedas — Click para capturar | Q/Esc para salir"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        _, _, W, H = self.config['region']
        scale = self._compute_scale(W, H)
        last_frame = [None]

        def on_mouse(event, x, y, flags, param):
            nonlocal template_count
            if event == cv2.EVENT_LBUTTONDOWN and last_frame[0] is not None:
                x_img = int(x / scale)
                y_img = int(y / scale)
                width_size = 150
                height_size = 80
                x1 = max(0, x_img - width_size // 2)
                y1 = max(0, y_img - height_size // 2)
                x2 = min(last_frame[0].shape[1], x_img + width_size // 2)
                y2 = min(last_frame[0].shape[0], y_img + height_size // 2)
                roi = last_frame[0][y1:y2, x1:x2]
                template_count += 1
                filename = f"money_button_{template_count}.png"
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, roi)
                print(f"✓ Template de botón de monedas guardado: {filename}")

        cv2.setMouseCallback(window_name, on_mouse)

        try:
            while True:
                frame = self.take_screenshot()
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_frame[0] = frame
                height = frame.shape[0]
                zone_y = int(height * self.config['money_button_zone'])
                display = frame.copy()
                cv2.line(display, (0, zone_y),
                         (display.shape[1], zone_y), (0, 255, 0), 2)
                cv2.putText(display, "Zona de búsqueda de botón", (10, zone_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if scale != 1.0:
                    disp = cv2.resize(
                        display, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
                else:
                    disp = display
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
            cv2.destroyAllWindows()
            self.load_templates()
            print("Calibración de botón completada")
        except Exception as e:
            cv2.destroyAllWindows()
            print(f"\n❌ Error durante calibración de monedas: {e}")


# ============================
# Menú principal
# ============================

def main_menu():
    print("\n" + "="*50)
    print("   🤖 BOT DETECTOR DE X + MONEDAS")
    print("="*50)

    bot = XDetectorBot()

    if os.path.exists('bluestacks_region.json'):
        try:
            with open('bluestacks_region.json', 'r') as f:
                saved_config = json.load(f)
                bot.config['region'] = tuple(saved_config['region'])
                print("\n✅ Configuración de región cargada automáticamente")
                print(f"   Región: {saved_config['region']}")
        except Exception as e:
            print(f"\n⚠️ No se pudo cargar la configuración guardada: {e}")

    while True:
        print("\n📋 MENÚ PRINCIPAL:")
        print("1. 🎯 Configurar región de BlueStacks")
        print("2. 📸 Calibrar templates de X (nuevo: CLICK en la ventana)")
        print("3. 💰 Calibrar botón de monedas (nuevo: CLICK en la ventana)")
        print("4. ▶️  Iniciar Bot")
        print("5. 📊 Ver configuración actual")
        print("6. ❌ Salir")

        choice = input("\nSelecciona opción (1-6): ")

        if choice == '1':
            result = bot.setup_region()
            if result:
                print("✓ Región configurada correctamente")
                # Guardar en archivo
                try:
                    with open('bluestacks_region.json', 'w') as f:
                        json.dump({'region': list(
                            result), 'money_button_zone': bot.config['money_button_zone'], 'timestamp': datetime.now().isoformat()}, f, indent=4)
                    print("   (Guardado en bluestacks_region.json)")
                except Exception as e:
                    print(f"   ⚠️ No se pudo guardar archivo de región: {e}")
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
    print("    BOT DETECTOR X + MONEDAS v2.1 (calibración por CLICK)")
    print("🤖"*20)

    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        if os.path.exists('bluestacks_region.json'):
            try:
                with open('bluestacks_region.json', 'r') as f:
                    config = json.load(f)
                bot = XDetectorBot()
                bot.config['region'] = tuple(config['region'])
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
        try:
            main_menu()
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print("   Por favor, reporta este error.")
