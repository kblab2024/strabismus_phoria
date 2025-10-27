#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
單張圖片最小示範（VS Code 直接執行）：
- MediaPipe Face Mesh（refine_landmarks=True 啟用 IRIS）
- 擬合每眼 4 個虹膜點為圓，畫出虹膜中心與半徑
- 由全部臉部關鍵點計算臉部外接矩形（臉框）
- **新增**：若已安裝 PyPupilEXT，於每眼虹膜 ROI 內偵測瞳孔中心/橢圓，並疊加結果

依賴：
    pip install opencv-python mediapipe numpy
（可選）
    pip install pypupilext   # 安裝後自動啟用更強的瞳孔偵測

使用：
    修改下方 IMAGE_PATH 後直接在 VS Code 執行。
"""

import cv2
import math
import numpy as np
import mediapipe as mp

# ===== 使用者設定 =====
IMAGE_PATH = r'F:\36FF2C53-501B-4458-84BA-58ED949777F1.jpeg'  # 請改成你的圖片路徑
# ====================

# 嘗試載入 PyPupilEXT
USE_PYPEX = False
try:
    import pypupilext as ppex  # type: ignore
    print("pypupilext imported")
    USE_PYPEX = True
except Exception:
    USE_PYPEX = False

RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS  = [474, 475, 476, 477]
mp_face_mesh = mp.solutions.face_mesh

def fit_circle(points_xy):
    """最小二乘擬合圓，points_xy: Nx2 ndarray"""
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(max(0.0, c[2] + cx*cx + cy*cy))
    return int(round(cx)), int(round(cy)), int(round(r))

def draw_face_bbox(img, landmarks_px):
    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]
    x0, y0 = max(0, min(xs)), max(0, min(ys))
    x1, y1 = min(img.shape[1]-1, max(xs)), min(img.shape[0]-1, max(ys))
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 150, 255), 2)

def crop_square(img, center, radius, scale=2.0):
    """以 center, radius 擷取方形 ROI；回傳 ROI 與其在原圖左上角座標 (x0,y0)。"""
    h, w = img.shape[:2]
    R = int(radius * scale)
    x0 = max(0, center[0] - R)
    y0 = max(0, center[1] - R)
    x1 = min(w, center[0] + R)
    y1 = min(h, center[1] + R)
    roi = img[y0:y1, x0:x1].copy()
    return roi, (x0, y0)

def detect_pupil_pypex(roi):
    """使用 PyPupilEXT（若可用）偵測瞳孔中心與橢圓。回傳 (center(x,y), ellipse 或 None) in ROI 座標。"""
    try:
        print("using pypex")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        result = None
        if hasattr(ppex, 'PuRe'):
            pure = ppex.PuRe()
            result = pure.detect(gray)
        elif hasattr(ppex, 'detect_pupil'):
            result = ppex.detect_pupil(gray)
        if result is None:
            return None, None
        cx, cy = int(result['center'][0]), int(result['center'][1])
        if 'axes' in result and 'angle' in result:
            a, b = int(result['axes'][0]), int(result['axes'][1])
            angle = float(result['angle'])
            return (cx, cy), ((cx, cy), (a, b), angle)
        return (cx, cy), None
    except Exception:
        return None, None

def detect_pupil_basic(roi):
    """不依賴外部庫的基本瞳孔偵測，回傳 (center(x,y), ellipse 或 None) in ROI 座標。"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    thr = cv2.medianBlur(thr, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    def roundness(cnt):
        area = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, True)
        if per == 0: return 0
        return 4 * math.pi * area / (per*per)
    candidates = []
    area_min = 20
    area_max = int(0.6 * roi.shape[0] * roi.shape[1])
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_min or a > area_max:
            continue
        candidates.append((roundness(c), a, c))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_cnt = candidates[0][2]
    if len(best_cnt) < 5:
        M = cv2.moments(best_cnt)
        if M['m00'] == 0:
            return None, None
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy), None
    else:
        ellipse = cv2.fitEllipse(best_cnt)
        (cx, cy), axes, angle = ellipse
        return (int(cx), int(cy)), ((int(cx), int(cy)), (int(axes[0]), int(axes[1])), float(angle))

def process_frame(frame, fm):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        print('未偵測到人臉/虹膜')
        return frame

    lms = res.multi_face_landmarks[0].landmark
    pts_px = [(int(l.x * w), int(l.y * h)) for l in lms]
    vis = frame.copy()
    draw_face_bbox(vis, pts_px)

    def get_xy(indices):
        return np.array([[pts_px[i][0], pts_px[i][1]] for i in indices], dtype=np.float32)

    for iris_idx, color in ((LEFT_IRIS, (0,255,0)), (RIGHT_IRIS, (0,255,0))):
        iris_xy = get_xy(iris_idx)
        cx, cy, r = fit_circle(iris_xy)
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.circle(vis, (cx, cy), 2, color, -1)

        # 以虹膜為中心建立 ROI，於其中偵測瞳孔
        roi, (x0, y0) = crop_square(frame, (cx, cy), r, scale=1.1)
        pupil_c, pupil_e = (None, None)
        if USE_PYPEX:
            pupil_c, pupil_e = detect_pupil_pypex(roi)
        if pupil_c is None:
            pupil_c, pupil_e = detect_pupil_basic(roi)
        if pupil_c is not None:
            pcx, pcy = x0 + pupil_c[0], y0 + pupil_c[1]
            cv2.circle(vis, (pcx, pcy), 2, (255, 0, 0), -1)
        if pupil_e is not None:
            (ecx, ecy), (ax, by), ang = pupil_e
            cv2.ellipse(vis, ((x0+int(ecx), y0+int(ecy)), (int(ax), int(by)), float(ang)), (255, 0, 0), 2)

    cv2.putText(vis, f'Face bbox + Iris + Pupil({"PyPupilEXT" if USE_PYPEX else "basic"})', (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2, cv2.LINE_AA)
    return vis

if __name__ == '__main__':
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise SystemExit(f'讀不到圖片：{IMAGE_PATH}')

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as fm:
        out = process_frame(img, fm)

    cv2.imshow('facemesh_iris_pupil_single', out)
    cv2.imwrite(r"D:\output_filename.png", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()