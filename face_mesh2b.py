import cv2
import mediapipe as mp

# 開啟畫關鍵點與face mesh網格功能
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
# 設定畫出點與線的粗細與顏色
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=1, circle_radius=1)
# 載入嘴唇的透明背景圖片
mouth_normal = cv2.imread("m6.png")
# 設定正確率
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 設定攝影機
cap = cv2.VideoCapture(0)
while (True):
    # 從攝影機擷取一禎圖片
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    # 先找出畫面的長寬大小
    h, w, d = frame.shape
    # Opencv用BGR所以先轉換顏色為RGB並傳到face mesh運算
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                # connections=mp_face_mesh.FACE_CONNECTIONS,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            # 點0與17分別是嘴唇上下的座標，取得嘴唇大小
            mouth_len = int((face_landmarks.landmark[17].y * h)-int(face_landmarks.landmark[0].y * h))
            # 將嘴唇圖案的圖片轉換成適合的大小
            mouth = cv2.resize(mouth_normal, (mouth_len * 3, mouth_len))
            # 將嘴唇圖案轉灰階
            mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            # 將嘴唇圖案去背
            _, mouth_mask = cv2.threshold(mouth_gray, 25, 255, cv2.THRESH_BINARY_INV)
            # 找出嘴唇的高度img_height 與寬度img_width
            img_height, img_width, _ = mouth.shape
            # 點13與14的中間是嘴唇的中心點，找出放圖的左上角落座標
            x, y = int(face_landmarks.landmark[13].x * w - img_width/2), \
                   int(((face_landmarks.landmark[13].y + face_landmarks.landmark[14].y)/2) * h - img_height/2)
            # 將去背圖案與真的人嘴唇合併成一矩形 mouth
            mouth_area = frame[y: y + img_height, x: x + img_width]
            mouth_area_no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=mouth_mask)
            mouth = cv2.add(mouth_area_no_mouth, mouth)
            # 在點(x, y)放上圖案mouth
            frame[y: y+img_height, x: x+img_width] = mouth
    cv2.imshow("face_mesh2b", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
