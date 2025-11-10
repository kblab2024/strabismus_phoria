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
    將資料夾(裡面裝所有原始圖片)的路徑換到picpath這個參數下
    
輸入輸出：
#input(a folder with all of the original pics) -> output(a subfolder named 'output', with a subfolder for each one of the pictures)
#output example:
    
input_folder(picpath)
----...all of the pictures
----output
--------name of picture 1
-------------1.png(right iris)
-------------1_processed.png(right iris processed)
-------------2.png(left iris)
-------------2_processed.png(left iris processed)
-------------original_picture_with_ellipse_processed.png
--------name of picture 2
..........................
"""

import cv2
import math
from math import sin, cos, radians
import numpy as np
import mediapipe as mp
from pprint import pprint as pp
from pathlib import Path
import matplotlib.pyplot as plt

def mainfunc(IMAGE_PATH):
    # ===== 使用者設定 =====
    #IMAGE_PATH = r'C:\Users\Tristan\Downloads\pics\3.JPEG'
    IMAGE_PATH = r"C:\Users\TBYTh\OneDrive\Desktop\Program\lab\eye\pic\S__159744056_0.jpg"  # 請改成你的圖片路徑
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
    #======================新增======================
    def get_upper_half_bbox_roi(img, landmarks_px):
        """
        從臉部關鍵點計算臉部 BBox，並返回其上半 1/2 的 ROI 和座標 (x0, y0, x1, y1)。
        """
        if not landmarks_px:
            return None, None

        # 1. 計算臉部外接矩形 (BBox)
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        x_min, y_min = max(0, min(xs)), max(0, min(ys))
        x_max, y_max = min(img.shape[1]-1, max(xs)), min(img.shape[0]-1, max(ys))
        
        # 計算臉部 BBox 的高度
        bbox_h = y_max - y_min
        
        # 2. 定義 ROI (BBox 的上方 1/2)
        x0 = x_min
        y0 = y_min
        x1 = x_max
        y1 = y_min + bbox_h // 2
        
        # 確保 y1 不超過圖片邊界
        y1 = min(img.shape[0], y1)

        # 擷取 ROI
        roi = img[y0:y1, x0:x1].copy()
        
        return roi, (x0, y0, x1, y1)
    
    def detect_largest_ellipses_in_roi(img_vis, roi, roi_coords):
        """
        在給定的 ROI 中使用輪廓偵測與橢圓擬合，尋找前兩大且最接近正圓的橢圓。
        """
        x0, y0, x1, y1 = roi_coords
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. 前處理：使用自適應閾值處理以分離前景/背景
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 3.5)
        thr = cv2.medianBlur(thr, 5) # 中值濾波降噪
        #形態學閉運算，用於連接破碎的輪廓
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        # 2. 輪廓檢測
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        roi_area = roi.shape[0] * roi.shape[1]
        
        for c in cnts:
            area = cv2.contourArea(c)
            
            # 1) 篩選面積：排除極小或極大的雜訊 (面積篩選標準不變)
            if area < 800 or area > 0.7 * roi_area: 
                continue

            # 2) 點數檢查：只有足夠的點 (>= 5) 才能進行橢圓擬合
            if len(c) < 5:
                continue
            
            # 3) 擬合橢圓
            ellipse = cv2.fitEllipse(c)
            (cx, cy), (ax, by), ang = ellipse
            
            # 4) 計算長短軸比例 (Aspect Ratio)
            minor_axis = min(ax, by)
            major_axis = max(ax, by)
            
            # 確保軸長非零
            if minor_axis <= 0:
                continue
                
            ratio = major_axis / minor_axis # 比例 >= 1
            
            # 5) 初步篩選：排除過於狹長的橢圓 (例如長軸是短軸的 1.5 倍以上)
            if ratio > 1.5:  
                continue

            # 6) 儲存候選者：儲存 (長短軸比例距離1的絕對值, 長軸大小, 橢圓參數)
            # 比例距離1越小，越接近正圓
            ratio_diff = abs(ratio - 1.0) 
            candidates.append((ratio_diff, major_axis, ellipse)) 

        # 3. 排序與選取：
        # 排序依據：
        #   主排序：ratio_diff 升序 (越接近 0 越好，即越接近圓)
        #   次排序：major_axis 降序 (在比例相近時，選更大的)
        candidates.sort(key=lambda x: (x[0], -x[1])) 
        
        # 取得最符合條件的前兩大橢圓
        detected_ellipses = candidates[:2] 

        print(f"Found {len(candidates)} candidate contours after filtering. Displaying top {len(detected_ellipses)} (Roundest & Largest).")

        # 4. 繪圖 (此部分與上次相同，保持不變)
        for ratio_diff, major_axis, ellipse in detected_ellipses:
            (cx_roi, cy_roi), axes, ang = ellipse
            
            global_center = (int(cx_roi + x0), int(cy_roi + y0))
            # 獲取軸長度（直徑）
            ax, by = axes
            major_len = max(ax, by)
            minor_len = min(ax, by)
            # 在原圖上繪製橢圓 (使用紫色 (255, 0, 255))
            cv2.ellipse(img_vis, global_center, (int(axes[0]/2), int(axes[1]/2)), ang, 0, 360, (255, 0, 255), 2)
            cv2.circle(img_vis, global_center, 2, (255, 0, 255), -1)
            
            # 將數值格式化為顯示兩位小數
            text = f"Maj:{major_len:.2f}, Min:{minor_len:.2f}"
            
            # 計算文字位置：放在橢圓頂部略偏外側 (使用 minor_len 的一半作為參考距離，避免太遠)
            half_minor = minor_len / 2
            text_x = global_center[0] - int(half_minor * 1.5) 
            text_y = global_center[1] - int(half_minor) - 5 # 離橢圓上邊緣約 5 像素
            
            # 避免文字超出圖像頂部 (如果太高，改放到下方)
            if text_y < 15:
                text_y = global_center[1] + int(half_minor) + 15
            
            cv2.putText(img_vis, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

        # 繪製 ROI 邊框 (橙色)
        cv2.rectangle(img_vis, (x0, y0), (x1, y1), (0, 150, 255), 2)
        cv2.putText(img_vis, 'Target Round Ellipses (Top 2)', (x0, y0 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1, cv2.LINE_AA)
        
        return img_vis
    #============================================
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

        # ===============================================
        #臉部上半部 ROI 圓圈偵測
        
        # 取得臉部上半部 ROI
        upper_roi, roi_coords = get_upper_half_bbox_roi(frame, pts_px)
        
        # 在 ROI 內偵測最大與次大圓，並繪製結果到 vis 圖片上
        if upper_roi is not None:
            vis = detect_largest_ellipses_in_roi(vis, upper_roi, roi_coords)
            
        # ===============================================

        def get_xy(indices):
            return np.array([[pts_px[i][0], pts_px[i][1]] for i in indices], dtype=np.float32)

        for nummer, (iris_idx, color) in enumerate(((LEFT_IRIS, (0,255,0)), (RIGHT_IRIS, (0,255,0)))):
            iris_xy = get_xy(iris_idx)
            cx, cy, r = fit_circle(iris_xy)
            #cv2.circle(vis, (cx, cy), r, color, 2)
            #cv2.circle(vis, (cx, cy), 2, color, -1)

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
                pp(pupil_e)
                (ecx, ecy), (ax, by), ang = pupil_e
                ecx = int(x0+int(ecx))
                ecy = int(y0+int(ecy))
                ang_deg = float(ang)
                ang = radians(float(ang))
                cv2.ellipse(vis, ((ecx, ecy), (int(ax), int(by)), ang_deg), (255, 0, 0), 2)
                
                ###以下為20251030新添加的，進行橢圓的遮罩###
                imgtemp = np.array(frame)
                gray = cv2.cvtColor(imgtemp, cv2.COLOR_BGR2GRAY)
                Y, X= np.ogrid[:gray.shape[0], :gray.shape[1]]
                #橢圓判斷式：
                #[ (x-xc) * cos(a) + (y-yc) * sin(a) ]^2 / ra^2 + [ -(x-xc) * sin(a) + (y-yc) * cos(a) ]^2 / rb^2 ≤ 1
                diff_formula = ((((X-ecx)*cos(ang) + (Y-ecy)*sin(ang))**2) / int(ax/2)**2) + (((-(X-ecx)*sin(ang))+((Y-ecy)*cos(ang)))**2/int(by/2)**2)
                #pr(distance_squared)
                mask = diff_formula <= 1
                imgtemp[~mask] = [255,0,0]
                
                #找橢圓極值：寫出橢圓的x(theta), y(theta)兩個函數後，微分一階 = 0
                # 最左
                xmin = int(ecx - math.sqrt(((ax/2) * cos(ang))**2 + ((by/2) * sin(ang))**2))
                # 最右
                xmax = int(ecx + math.sqrt(((ax/2) * cos(ang))**2 + ((by/2) * sin(ang))**2))
                # 最上
                ymax = int(ecy + math.sqrt(((ax/2) * sin(ang))**2 + ((by/2) * cos(ang))**2))
                # 最下
                ymin = int(ecy - math.sqrt(((ax/2) * sin(ang))**2 + ((by/2) * cos(ang))**2))
                
                imgtemp = imgtemp[ ymin:(ymax+1), xmin:(xmax+1), :]
                
                
                outpath = Path(IMAGE_PATH).parent/'output'/f'{Path(IMAGE_PATH).stem}'
                outpath.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(outpath/f"{nummer+1}.png", imgtemp)
                ###END###


        cv2.putText(imgtemp, f'Face bbox + Iris + Pupil({"PyPupilEXT" if USE_PYPEX else "basic"})', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2, cv2.LINE_AA)
        return imgtemp, vis

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise SystemExit(f'讀不到圖片：{IMAGE_PATH}')

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as fm:
        out, out2 = process_frame(img, fm)
    #cv2.imshow('facemesh_iris_pupil_single', out)
    outpath = Path(IMAGE_PATH).parent/'output'/f'{Path(IMAGE_PATH).stem}'
    outpath.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(outpath/'original_picture_with_ellipse_processed.png', out2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#input(a folder with all of the original pics) -> output(a subfolder named 'output', with a subfolder for each one of the pictures)
#output example:
    
#input_folder(picpath)
#----...all of the pictures
#----output
#-------------1.png(right iris)
#-------------1_processed.png(right iris processed)
#-------------2.png(left iris)
#-------------2_processed.png(left iris processed)
#-------------original_picture_with_ellipse_processed.png
#..........................

#picpath = Path(r"C:\Users\Tristan\Downloads\picswithlens") #裝有原圖的資料夾
picpath = Path(r"C:\Users\TBYTh\OneDrive\Desktop\Program\lab\eye\pic")
for file in picpath.iterdir():
    if file.suffix.lower() in ['.jpg', '.jpeg','.JPEG', '.JPG','.png']:
        out=mainfunc(file)  #input image path
    
###以下是將切下來的橢圓拿來找亮點###

outpicpath = picpath/'output' #the 'output' folder generated from face_mesh_iris_pupil.py

for file in outpicpath.rglob('*png'):
    print(f'extracting bright spot from {file}')
    if 'processed' not in str(file): 
        img = cv2.imread(file)
        img2 = np.array(img)
        foundspot = []
    
        #ser_bound_x = round(img.shape[0]/4)
        #ser_bound_y = round(img.shape[1]/4)
    
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img2[i][j][0] >190 and img2[i][j][1] >190 and img2[i][j][2] >190:
                    img2[i][j] = [0,0,255]
                    foundspot.append((i,j))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        images = [img, img2]
        imagename = ['origianl pic', 'result']
    
    
    
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))   # 建立 2x2 子圖
        for i, ax in enumerate(axes.flat):
            if i ==0:
                ax.imshow(images[i], cmap = 'gray')
            else: 
                ax.imshow(images[i])
            ax.set_title(imagename[i])
            
        fig.suptitle(f'file name:{file}\n pic size: {img.shape[:2]}\n spot found: {foundspot}\n threshold: [190,190,190]', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # 預留 top 1% 給 suptitle，不會壓到子圖
        plt.savefig(f'{file.parent/file.stem}_processed.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        #plt.show()