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
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 使用者設定 =====
IMAGE_PATH = r'/Volumes/SanDisk/587552351348064685.jpg'  # 請改成你的圖片路徑
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


def detect_largest_circle_at_center(img_vis, roi, roi_coords, target_center):
    """
    在 ROI 中尋找距離 target_center (cx, cy) 最近且面積最大的橢圓。
    Returns: img_vis, detected_ellipse_params (or None if not found)
    """
    x0, y0, x1, y1 = roi_coords
    tx, ty = target_center
    
    # 將全域中心點轉換為 ROI 局部座標
    local_center = (int(tx - x0), int(ty - y0))
    
    # 檢查中心點是否在 ROI 範圍內
    h, w = roi.shape[:2]
    if not (0 <= local_center[0] < w and 0 <= local_center[1] < h):
        return img_vis, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 1. 前處理
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 3.5)
    thr = cv2.medianBlur(thr, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 2. 輪廓檢測
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_candidates = []
    roi_area = roi.shape[0] * roi.shape[1]
    
    for c in cnts:
        area = cv2.contourArea(c)
        
        # 基礎過濾
        if area < 100 or area > 0.9 * roi_area:
            continue
        if len(c) < 5:
            continue

        # 擬合橢圓
        ellipse = cv2.fitEllipse(c)
        (cx, cy), (ax, by), ang = ellipse
        
        # 計算距離
        dist = ((cx - local_center[0])**2 + (cy - local_center[1])**2)**0.5
            
        # 儲存候選者：(面積, 距離, 橢圓參數)
        valid_candidates.append((area, dist, ellipse))

    # 3. 選取最大面積的候選者
    if valid_candidates:
        # 選取面積最大的單一候選者
        largest_area, dist, target_ellipse = max(valid_candidates, key=lambda x: x[0])
        
        # 4. 繪圖
        (cx_roi, cy_roi), axes, ang = target_ellipse
        global_center_ellipse = (int(cx_roi + x0), int(cy_roi + y0))
        
        # 繪製橢圓 (綠色)
        cv2.ellipse(img_vis, global_center_ellipse, (int(axes[0]/2), int(axes[1]/2)), ang, 0, 360, (0, 255, 0), 2)
        
        # 繪製目標中心 (紅色)
        cv2.circle(img_vis, (int(tx), int(ty)), 3, (0, 0, 255), -1)
        
        # 繪製連線 (顯示偏移量)
        cv2.line(img_vis, (int(tx), int(ty)), global_center_ellipse, (0, 255, 255), 1)
        
        # 標示文字
        cv2.putText(img_vis, f"Dist:{dist:.1f}", (global_center_ellipse[0]-20, global_center_ellipse[1]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return img_vis, target_ellipse

    return img_vis, None


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

    for nummer, (iris_idx, color) in enumerate(((LEFT_IRIS, (0,255,0)), (RIGHT_IRIS, (0,255,0)))):
        iris_xy = get_xy(iris_idx)
        cx, cy, r = fit_circle(iris_xy)
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.circle(vis, (cx, cy), 2, color, -1)
        roi, (x0, y0) = crop_square(frame, (cx, cy), r, scale=3.0)
        h, w = roi.shape[:2]
        roi_coords = (x0, y0, x0 + w, y0 + h)   
        vis, detected_ellipse = detect_largest_circle_at_center(vis, roi, roi_coords, (cx, cy))
        
        # Save the detected circle to a figure
        if 'IMAGE_PATH' in globals() and IMAGE_PATH and detected_ellipse is not None:
            outpath = Path(IMAGE_PATH).parent/'output'/f'{Path(IMAGE_PATH).stem}'
            outpath.mkdir(parents=True, exist_ok=True)
            
            #存最大的圓
            (ecx_roi, ecy_roi), (eax, eby), eang = detected_ellipse
            
            # Save the cropped ROI with circular mask applied
            circle_roi = vis[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]].copy()
            
            # Create elliptical mask for the detected circle
            mask = np.zeros(circle_roi.shape[:2], dtype=np.uint8)
            # Calculate center in ROI coordinates (already in ROI coords from detection)
            center_in_roi = (int(ecx_roi), int(ecy_roi))
            # Draw ellipse on mask
            cv2.ellipse(mask, center_in_roi, (int(eax/2), int(eby/2)), eang, 0, 360, 255, -1)
            
            # Apply mask to keep only circular region
            circle_roi = cv2.bitwise_and(circle_roi, circle_roi, mask=mask)
            
            cv2.imwrite(str(outpath/f"largest_circle_{nummer+1}.png"), circle_roi)
            circle_roi = cv2.bitwise_and(circle_roi, circle_roi, mask=mask)
            
            cv2.imwrite(str(outpath/f"largest_circle_{nummer+1}.png"), circle_roi)
            #存虹膜區域
            # Create a circular mask and extract only the circle region
            # 1. Create a square ROI with some padding
            padding = int(r * 1.2)
            x_min, x_max = int(cx - padding), int(cx + padding)
            y_min, y_max = int(cy - padding), int(cy + padding)
            
            # 2. Ensure coordinates are within image bounds
            h, w = frame.shape[:2]
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            # 3. Crop the square region
            iris_roi = frame[y_min:y_max, x_min:x_max].copy()
            
            # 4. Create circular mask
            mask = np.zeros(iris_roi.shape[:2], dtype=np.uint8)
            # Calculate center in ROI coordinates
            center_in_roi = (cx - x_min, cy - y_min)
            cv2.circle(mask, center_in_roi, r, 255, -1)
            
            # 5. Apply mask to keep only circular region
            iris_circle = cv2.bitwise_and(iris_roi, iris_roi, mask=mask)
            
            cv2.imwrite(str(outpath/f"{nummer+1}.png"), iris_circle)
        """
        # 以虹膜為中心建立 ROI，於其中偵測瞳孔
        
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
        """
    cv2.putText(vis, f'Face bbox + Iris + Pupil({"PyPupilEXT" if USE_PYPEX else "basic"})', (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2, cv2.LINE_AA)
    return vis

def detect_bright_spots_in_output():
    """
    Detects bright spots in the output images generated by the face mesh processing.
    """
    pic_base = Path(IMAGE_PATH).parent
    outpicpath = pic_base / 'output' / Path(IMAGE_PATH).stem
    
    if not outpicpath.exists():
        print(f"Output folder not found: {outpicpath}")
        return
    
    for file in outpicpath.rglob('*.png'):
        if '_processed' in file.name:  # Skip already processed files
            continue
            
        print(f'Extracting bright spot from {file.name}')
        
        img = cv2.imread(str(file))
        if img is None:
            continue
        
        # Convert to RGB for processing and plotting
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_result = img_rgb.copy()
        
        threshold = 190
        foundspot = []
        
        # Detect bright spots
        for i in range(img_result.shape[0]):
            for j in range(img_result.shape[1]):
                if (img_result[i][j][0] > threshold and 
                    img_result[i][j][1] > threshold and 
                    img_result[i][j][2] > threshold):
                    img_result[i][j] = [255, 0, 0]  # Mark in red
                    foundspot.append((i, j))
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Pic')
        axes[0].axis('off')
        
        axes[1].imshow(img_result)
        axes[1].set_title('Result (Red Spots)')
        axes[1].axis('off')
        
        fig.suptitle(
            f'File: {file.name}\nSize: {img.shape[:2]}\n'
            f'Spots found: {len(foundspot)}\nThreshold: RGB > {threshold}',
            fontsize=12, fontweight='bold'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = file.parent / f"{file.stem}_processed.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path.name}")


if __name__ == '__main__':
    #process_image_for_iris_output(IMAGE_PATH)    
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
    # now detect bright spots in the output image
    detect_bright_spots_in_output()
    output_path = Path(IMAGE_PATH).parent / 'output' / f'{Path(IMAGE_PATH).stem}_result.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    print(f"Saved result to: {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    