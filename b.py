import pyttsx3
import serial
import math
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

# Khởi tạo camera
cap = cv2.VideoCapture(1)
w, h = 640, 360
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 30)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

engine = pyttsx3.init()


try :
    ESP = serial.Serial('COM3 ', 115200, timeout=1)
    print("Kết nối với ESP32 thành công.")
    pyttsx3.speak("completed connect to ESP32")
except serial.SerialException:
    print("Không thể kết nối với ESP32. Vui lòng kiểm tra cổng COM và kết nối.")
    pyttsx3.speak("Cannot connect to ESP32. Please check the COM port and connection.")
    exit()
def send_signal_to_esp32(signal):
    if ESP.is_open:
        ESP.write(signal.encode())



# 1. BIẾN CHO QUÁ TRÌNH HIỆU CHUẨN (CALIBRATION) ---
is_calibrated = False
calibration_ear_list = []
calibration_start_time = time.time()
CALIBRATION_DURATION = 3.0  # Thời gian hiệu chuẩn (giây)
EAR_THRESHOLD = 0.38        # Giá trị tạm thời, sẽ được ghi đè sau khi hiệu chuẩn

start_close_eye = 0
CLOSE_TIME_THRESHOLD = 2.0
frame_count = 0
SKIP_FRAME = 2
cached_ear = 0.5
catched_face = None

while True:
    success, img = cap.read()
    if not success:
        break
    
    frame_count += 1
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Chạy AI (có áp dụng resize và skip frame để tối ưu tốc độ)
    if frame_count % SKIP_FRAME == 0:
        small_img = cv2.resize(img, (320, 180))
        rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        result = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small))
        if result.face_landmarks:
            catched_face = result.face_landmarks[0]
            face = catched_face
            sw, sh = 320, 180
            for landmark in face:
                print("LEFT EYE")
                for diem in LEFT_EYE:
                    landmark = face[diem]
                    print(landmark.x, landmark.y)
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                print("RIGHT EYE")
                for diem in RIGHT_EYE:
                    landmark = face[diem]
                    print(landmark.x, landmark.y)
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            # Tính toán EAR
            d_1 = math.dist((face[33].x * w, face[33].y * h), (face[133].x * w, face[133].y * h))
            d_2 = math.dist((face[160].x * w, face[160].y * h), (face[153].x * w, face[153].y * h))
            d_3 = math.dist((face[158].x * w, face[158].y * h), (face[144].x * w, face[144].y * h))
            d_4 = math.dist((face[362].x * w, face[362].y * h), (face[263].x * w, face[263].y * h))
            d_5 = math.dist((face[385].x * w, face[385].y * h), (face[373].x * w, face[373].y * h))
            d_6 = math.dist((face[387].x * w, face[387].y * h), (face[380].x * w, face[380].y * h))
    
            EAR_L = (d_2 + d_3) / (2 * d_1)
            EAR_R = (d_5 + d_6) / (2 * d_4)
            cached_ear = (EAR_L + EAR_R) / 2
        else:
            catched_face = None
    

    # 1. GIAI ĐOẠN HIỆU CHUẨN (CALIBRATION PHASE)

    if not is_calibrated:
        calib_time = time.time() - calibration_start_time    
        if calib_time < CALIBRATION_DURATION:
            if cached_ear > 0.2 and catched_face is not None:  # Chỉ lấy mẫu khi mắt mở bình thường
                calibration_ear_list.append(cached_ear)
            
            # Hiển thị thông báo đang hiệu chuẩn lên màn hình
            cv2.putText(img, "DANG HIEU CHUAN... MO MAT NHIN THANG", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f"Thoi gian con lai: {int(CALIBRATION_DURATION - calib_time) + 1}s", (30, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Hoàn tất hiệu chuẩn
            if len(calibration_ear_list) > 0:
                baseline_open_ear = sum(calibration_ear_list) / len(calibration_ear_list)
                EAR_THRESHOLD = baseline_open_ear * 0.86
                is_calibrated = True
                print(f"[INFO] Hieu chuan thanh cong! Baseline EAR: {baseline_open_ear:.2f}, Nguong EAR_THRESHOLD moi: {EAR_THRESHOLD:.2f}")
            else:
                # Fallback an toàn nếu không bắt được khuôn mặt trong 3s đầu
                calibration_start_time = time.time()
        cv2.putText(img, f"EAR: {cached_ear:.2f} | Nguong: {EAR_THRESHOLD:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("camera canh bao ngu gat", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        continue  # Bỏ qua các bước tiếp theo trong vòng lặp, chỉ thực hiện hiệu chuẩn

    # 2. GIAI ĐOẠN GIÁM SÁT CHÍNH THỨC (SAU KHI ĐÃ CÓ NGƯỠNG CHUẨN) 
    if cached_ear < EAR_THRESHOLD:
        if start_close_eye == 0:
            start_close_eye = time.time()
    else:
        start_close_eye = 0
    if start_close_eye != 0 and time.time() - start_close_eye > CLOSE_TIME_THRESHOLD:
                for diem in LEFT_EYE:
                    landmark = face[diem]
                    print (landmark.x, landmark.y)
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(img, (x,y), 1, (0, 0, 255), -1)
                for diem in RIGHT_EYE:
                    landmark = face[diem]
                    print (landmark.x, landmark.y)
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(img, (x,y), 1, (0, 0, 255), -1)
                cv2.putText(img, "DANG NGU GAT",(50, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 100, 255),2)
                pyttsx3.speak(" Wake Up ")
                ESP.write(b'1')  # Gửi tín hiệu '1' đến ESP32
    else:
                cv2.putText(img, "Binh thuong",(50, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
                ESP.write(b'0')  # Gửi tín hiệu '0' đến ESP32

    # Hiển thị thông số lên màn hình
    cv2.putText(img, "nhan q de thoat", (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
    cv2.putText(img, f"EAR: {cached_ear:.2f} | Nguong: {EAR_THRESHOLD:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("camera canh bao ngu gat", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ESP.close()