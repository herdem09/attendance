import sys
import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QComboBox, QTextEdit, QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

class Translations:
    TR = {
        'title': 'Yüz Tanıma Yoklama Sistemi',
        'language': 'Dil:',
        'add_student': 'Öğrenci Ekle',
        'take_attendance': 'Yoklama Al',
        'student_name': 'Öğrenci Adı:',
        'student_number': 'Öğrenci Numarası:',
        'class_name': 'Sınıf:',
        'start_capture': 'Fotoğraf Çekmeye Başla',
        'capture_photo': 'Fotoğraf Çek ({}/10)',
        'save_student': 'Öğrenciyi Kaydet',
        'start_attendance': 'Yoklama Başlat',
        'stop_attendance': 'Yoklama Durdur',
        'save_attendance': 'Yoklamayı Kaydet',
        'detected_students': 'Tespit Edilen Öğrenciler:',
        'enter_all_fields': 'Lütfen tüm alanları doldurun!',
        'photos_captured': '{} fotoğraf çekildi!',
        'student_saved': 'Öğrenci başarıyla kaydedildi!',
        'attendance_saved': 'Yoklama kaydedildi!',
        'cancel': 'İptal',
        'no_face': 'Yüz tespit edilemedi!',
        'no_students': 'Hiç öğrenci tespit edilmedi!'
    }
    
    EN = {
        'title': 'Face Recognition Attendance System',
        'language': 'Language:',
        'add_student': 'Add Student',
        'take_attendance': 'Take Attendance',
        'student_name': 'Student Name:',
        'student_number': 'Student Number:',
        'class_name': 'Class:',
        'start_capture': 'Start Capturing Photos',
        'capture_photo': 'Capture Photo ({}/10)',
        'save_student': 'Save Student',
        'start_attendance': 'Start Attendance',
        'stop_attendance': 'Stop Attendance',
        'save_attendance': 'Save Attendance',
        'detected_students': 'Detected Students:',
        'enter_all_fields': 'Please fill all fields!',
        'photos_captured': '{} photos captured!',
        'student_saved': 'Student saved successfully!',
        'attendance_saved': 'Attendance saved!',
        'cancel': 'Cancel',
        'no_face': 'Face not detected!',
        'no_students': 'No students detected!'
    }
    
    IT = {
        'title': 'Sistema di Presenza con Riconoscimento Facciale',
        'language': 'Lingua:',
        'add_student': 'Aggiungi Studente',
        'take_attendance': 'Prendi Presenza',
        'student_name': 'Nome Studente:',
        'student_number': 'Numero Studente:',
        'class_name': 'Classe:',
        'start_capture': 'Inizia a Scattare Foto',
        'capture_photo': 'Scatta Foto ({}/10)',
        'save_student': 'Salva Studente',
        'start_attendance': 'Inizia Presenza',
        'stop_attendance': 'Ferma Presenza',
        'save_attendance': 'Salva Presenza',
        'detected_students': 'Studenti Rilevati:',
        'enter_all_fields': 'Per favore compila tutti i campi!',
        'photos_captured': '{} foto scattate!',
        'student_saved': 'Studente salvato con successo!',
        'attendance_saved': 'Presenza salvata!',
        'cancel': 'Annulla',
        'no_face': 'Viso non rilevato!',
        'no_students': 'Nessuno studente rilevato!'
    }

class SimpleFaceRecognizer:
    """Basit histogram karşılaştırma tabanlı yüz tanıma"""
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        return faces, gray
    
    def extract_features(self, face_img):
        """Yüz özelliklerini çıkar (histogram tabanlı)"""
        # Yüzü normalize et
        face_img = cv2.resize(face_img, (100, 100))
        face_img = cv2.equalizeHist(face_img)
        
        # Histogram özelliklerini çıkar
        hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Ek özellikler: gradyan histogram
        sobelx = cv2.Sobel(face_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(face_img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        grad_hist = cv2.calcHist([gradient_magnitude.astype(np.uint8)], [0], None, [128], [0, 256])
        grad_hist = cv2.normalize(grad_hist, grad_hist).flatten()
        
        # Tüm özellikleri birleştir
        features = np.concatenate([hist, grad_hist, face_img.flatten() / 255.0])
        return features
    
    def compare_faces(self, features1, features2):
        """İki yüz özelliğini karşılaştır (0-100 arası skor, düşük = benzer)"""
        # Kosinüs benzerliği
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        # 0-100 skalasına çevir (düşük = daha iyi eşleşme)
        score = (1 - similarity) * 100
        return score

class AttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = 'TR'
        self.translations = Translations.TR
        
        # Veri dosyaları
        self.data_file = 'students_data.pkl'
        self.students_db = self.load_students()
        
        # Yüz tanıma
        self.face_recognizer = SimpleFaceRecognizer()
        
        # Kamera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Öğrenci ekleme değişkenleri
        self.capturing = False
        self.captured_features = []
        self.max_photos = 10
        
        # Yoklama değişkenleri
        self.attendance_mode = False
        self.detected_students = {}
        self.detection_counts = {}  # Her öğrenci için tespit sayısı
        
        self.init_ui()
        
    def load_students(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_students(self):
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.students_db, f)
    
    def init_ui(self):
        self.setWindowTitle(self.translations['title'])
        self.setGeometry(100, 100, 1200, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Üst bar - Dil seçimi
        top_bar = QHBoxLayout()
        lang_label = QLabel(self.translations['language'])
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['Türkçe', 'English', 'Italiano'])
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        top_bar.addWidget(lang_label)
        top_bar.addWidget(self.lang_combo)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        # Ana butonlar
        btn_layout = QHBoxLayout()
        self.btn_add_student = QPushButton(self.translations['add_student'])
        self.btn_take_attendance = QPushButton(self.translations['take_attendance'])
        self.btn_add_student.clicked.connect(self.show_add_student)
        self.btn_take_attendance.clicked.connect(self.show_take_attendance)
        
        # Buton stilleri
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """
        self.btn_add_student.setStyleSheet(button_style)
        self.btn_take_attendance.setStyleSheet(button_style.replace('#4CAF50', '#2196F3').replace('#45a049', '#0b7dda'))
        
        btn_layout.addWidget(self.btn_add_student)
        btn_layout.addWidget(self.btn_take_attendance)
        main_layout.addLayout(btn_layout)
        
        # Video gösterimi
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 3px solid #333; background-color: #000; border-radius: 10px;")
        main_layout.addWidget(self.video_label)
        
        # Öğrenci ekleme paneli
        self.add_student_widget = QWidget()
        add_layout = QVBoxLayout(self.add_student_widget)
        
        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel(self.translations['student_name']))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Ali Yılmaz")
        form_layout.addWidget(self.name_input)
        
        form_layout.addWidget(QLabel(self.translations['student_number']))
        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("20240001")
        form_layout.addWidget(self.number_input)
        
        form_layout.addWidget(QLabel(self.translations['class_name']))
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("10-A")
        form_layout.addWidget(self.class_input)
        add_layout.addLayout(form_layout)
        
        capture_btn_layout = QHBoxLayout()
        self.btn_start_capture = QPushButton(self.translations['start_capture'])
        self.btn_start_capture.clicked.connect(self.start_capturing)
        self.btn_save_student = QPushButton(self.translations['save_student'])
        self.btn_save_student.clicked.connect(self.save_student)
        self.btn_save_student.setEnabled(False)
        self.btn_start_capture.setStyleSheet(button_style.replace('#4CAF50', '#FF9800').replace('#45a049', '#e68900'))
        self.btn_save_student.setStyleSheet(button_style)
        capture_btn_layout.addWidget(self.btn_start_capture)
        capture_btn_layout.addWidget(self.btn_save_student)
        add_layout.addLayout(capture_btn_layout)
        
        self.add_student_widget.hide()
        main_layout.addWidget(self.add_student_widget)
        
        # Yoklama paneli
        self.attendance_widget = QWidget()
        attendance_layout = QVBoxLayout(self.attendance_widget)
        
        attendance_btn_layout = QHBoxLayout()
        self.btn_start_attendance = QPushButton(self.translations['start_attendance'])
        self.btn_start_attendance.clicked.connect(self.start_attendance)
        self.btn_save_attendance = QPushButton(self.translations['save_attendance'])
        self.btn_save_attendance.clicked.connect(self.save_attendance)
        self.btn_save_attendance.setEnabled(False)
        self.btn_start_attendance.setStyleSheet(button_style.replace('#4CAF50', '#FF9800').replace('#45a049', '#e68900'))
        self.btn_save_attendance.setStyleSheet(button_style)
        attendance_btn_layout.addWidget(self.btn_start_attendance)
        attendance_btn_layout.addWidget(self.btn_save_attendance)
        attendance_layout.addLayout(attendance_btn_layout)
        
        self.detected_label = QLabel(self.translations['detected_students'])
        self.detected_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        self.detected_text = QTextEdit()
        self.detected_text.setReadOnly(True)
        self.detected_text.setMaximumHeight(150)
        self.detected_text.setStyleSheet("""
            background-color: #f5f5f5; 
            border: 2px solid #ddd; 
            border-radius: 5px;
            padding: 10px;
            font-size: 13px;
        """)
        attendance_layout.addWidget(self.detected_label)
        attendance_layout.addWidget(self.detected_text)
        
        self.attendance_widget.hide()
        main_layout.addWidget(self.attendance_widget)
        
    def change_language(self, index):
        lang_map = {0: 'TR', 1: 'EN', 2: 'IT'}
        self.current_lang = lang_map[index]
        self.translations = getattr(Translations, self.current_lang)
        self.update_ui_texts()
        
    def update_ui_texts(self):
        self.setWindowTitle(self.translations['title'])
        self.btn_add_student.setText(self.translations['add_student'])
        self.btn_take_attendance.setText(self.translations['take_attendance'])
        if not self.capturing:
            self.btn_start_capture.setText(self.translations['start_capture'])
        self.btn_save_student.setText(self.translations['save_student'])
        if self.attendance_mode:
            self.btn_start_attendance.setText(self.translations['stop_attendance'])
        else:
            self.btn_start_attendance.setText(self.translations['start_attendance'])
        self.btn_save_attendance.setText(self.translations['save_attendance'])
        self.detected_label.setText(self.translations['detected_students'])
        
    def show_add_student(self):
        self.attendance_mode = False
        self.attendance_widget.hide()
        self.add_student_widget.show()
        self.start_camera()
        
    def show_take_attendance(self):
        self.add_student_widget.hide()
        self.attendance_widget.show()
        self.detected_students.clear()
        self.detection_counts.clear()
        self.detected_text.clear()
        self.start_camera()
        
    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Kamera açılamadı / Cannot open camera / Impossibile aprire la fotocamera")
                return
            self.timer.start(30)
            
    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            if self.attendance_mode:
                frame = self.process_attendance(frame)
            else:
                # Yüz tespiti göster
                faces, gray = self.face_recognizer.detect_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            
    def start_capturing(self):
        if not self.name_input.text() or not self.number_input.text() or not self.class_input.text():
            QMessageBox.warning(self, self.translations['cancel'], 
                              self.translations['enter_all_fields'])
            return
            
        self.capturing = True
        self.captured_features = []
        self.btn_start_capture.setText(self.translations['capture_photo'].format(0, self.max_photos))
        self.btn_start_capture.clicked.disconnect()
        self.btn_start_capture.clicked.connect(self.capture_photo)
        
    def capture_photo(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                faces, gray = self.face_recognizer.detect_faces(frame)
                
                if len(faces) > 0:
                    # İlk yüzü al
                    (x, y, w, h) = faces[0]
                    face_img = gray[y:y+h, x:x+w]
                    
                    # Özellikleri çıkar
                    features = self.face_recognizer.extract_features(face_img)
                    self.captured_features.append(features)
                    
                    self.btn_start_capture.setText(
                        self.translations['capture_photo'].format(len(self.captured_features), self.max_photos)
                    )
                    
                    if len(self.captured_features) >= self.max_photos:
                        self.capturing = False
                        self.btn_save_student.setEnabled(True)
                        QMessageBox.information(self, self.translations['cancel'], 
                                              self.translations['photos_captured'].format(self.max_photos))
                        self.btn_start_capture.clicked.disconnect()
                        self.btn_start_capture.clicked.connect(self.start_capturing)
                        self.btn_start_capture.setText(self.translations['start_capture'])
                else:
                    QMessageBox.warning(self, self.translations['cancel'], 
                                      self.translations['no_face'])
                    
    def save_student(self):
        if len(self.captured_features) < self.max_photos:
            return
            
        student_id = self.number_input.text()
        
        student_data = {
            'name': self.name_input.text(),
            'number': student_id,
            'class': self.class_input.text(),
            'features': self.captured_features
        }
        
        self.students_db[student_id] = student_data
        self.save_students()
        
        QMessageBox.information(self, self.translations['cancel'], 
                              self.translations['student_saved'])
        
        self.name_input.clear()
        self.number_input.clear()
        self.class_input.clear()
        self.captured_features = []
        self.btn_save_student.setEnabled(False)
        
    def start_attendance(self):
        if not self.attendance_mode:
            self.attendance_mode = True
            self.detected_students.clear()
            self.detection_counts.clear()
            self.detected_text.clear()
            self.btn_start_attendance.setText(self.translations['stop_attendance'])
            self.btn_save_attendance.setEnabled(True)
        else:
            self.attendance_mode = False
            self.btn_start_attendance.setText(self.translations['start_attendance'])
            
    def process_attendance(self, frame):
        faces, gray = self.face_recognizer.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            current_features = self.face_recognizer.extract_features(face_img)
            
            best_match = None
            best_score = float('inf')
            
            # Tüm kayıtlı öğrencilerle karşılaştır
            for student_id, student_data in self.students_db.items():
                for stored_features in student_data['features']:
                    score = self.face_recognizer.compare_faces(current_features, stored_features)
                    if score < best_score:
                        best_score = score
                        best_match = student_id
            
            name = "Unknown"
            # Eşik değeri: 35'in altındaki skorlar eşleşme olarak kabul edilir
            if best_match and best_score < 35:
                student_data = self.students_db[best_match]
                name = student_data['name']
                
                # Tespit sayısını artır (en az 5 kez tespit edilen öğrenciler kaydedilir)
                if best_match not in self.detection_counts:
                    self.detection_counts[best_match] = 0
                self.detection_counts[best_match] += 1
                
                # 5 veya daha fazla tespit edilirse listeye ekle
                if self.detection_counts[best_match] >= 5:
                    self.detected_students[best_match] = student_data
                    
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Yüz çerçevesi
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # İsim etiketi arka planı
            label_text = f"{name}"
            score_text = f"Score: {int(best_score)}"
            
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-40), (x+text_width+10, y), color, -1)
            cv2.putText(frame, label_text, (x+5, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, score_text, (x+5, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        self.update_detected_list()
        return frame
        
    def update_detected_list(self):
        text = ""
        for student_id, student in sorted(self.detected_students.items(), 
                                         key=lambda x: x[1]['name']):
            text += f"✓ {student['name']:20} | {student['number']:10} | {student['class']}\n"
        self.detected_text.setText(text)
        
    def save_attendance(self):
        if len(self.detected_students) == 0:
            QMessageBox.warning(self, self.translations['cancel'], 
                              self.translations['no_students'])
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"   YOKLAMA RAPORU / ATTENDANCE REPORT / RAPPORTO DI PRESENZA\n")
            f.write("="*80 + "\n")
            f.write(f"Tarih / Date / Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Toplam / Total / Totale: {len(self.detected_students)} öğrenci/students/studenti\n")
            f.write("="*80 + "\n\n")
            
            for i, (student_id, student) in enumerate(sorted(self.detected_students.items(), 
                                                             key=lambda x: x[1]['name']), 1):
                f.write(f"{i:3}. ✓ {student['name']:25} | {student['number']:12} | {student['class']}\n")
            
            f.write("\n" + "="*80 + "\n")
                
        QMessageBox.information(self, self.translations['cancel'], 
                              f"{self.translations['attendance_saved']}\n\n📄 {filename}")
        
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AttendanceSystem()
    window.show()
    sys.exit(app.exec_())
