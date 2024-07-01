import cv2
import numpy as np
import sys

CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
CAMERA_WIDTH_S, CAMERA_HEIGHT_S = CAMERA_WIDTH // 8, CAMERA_HEIGHT // 8

ROAD_VANISH_Y = 0

ROAD_MARGIN_X = 600
ROAD_MARGIN_Y = 150

def get_persptrans_matrix():
    vanish_y = ROAD_VANISH_Y
    margin_x = ROAD_MARGIN_X

    dx = CAMERA_WIDTH / 2 + margin_x
    dy = CAMERA_HEIGHT - vanish_y
    
    margin_y = ROAD_MARGIN_Y
    
    xr = (dx / dy) * (margin_y - vanish_y) + (CAMERA_WIDTH / 2)
    xl = CAMERA_WIDTH - xr
    
    # 해당 src는 주행 도로 폭이 너무 좁았음 한번 이거로도 돌려보길 추천
    # src = np.array([(xl, margin_y),
    #                 (xr, margin_y),
    #                 (-margin_x, CAMERA_HEIGHT-250),
    #                 (CAMERA_WIDTH + margin_x, CAMERA_HEIGHT-250)], dtype=np.float32)
    
    # 그래서 그냥 중심 기준으로 해서 자르도록 사다리꼴 만들었음
    src = np.array([(CAMERA_WIDTH/2 - 120, margin_y),
                    (CAMERA_WIDTH/2 + 120, margin_y),
                    (CAMERA_WIDTH/2 - 1300, CAMERA_HEIGHT),
                    (CAMERA_WIDTH/2 + 1300, CAMERA_HEIGHT)], dtype=np.float32)
    dst = np.array([(0, 0),
                    (CAMERA_WIDTH, 0),
                    (0, CAMERA_HEIGHT),
                    (CAMERA_WIDTH, CAMERA_HEIGHT)], dtype=np.float32)
    
    return cv2.getPerspectiveTransform(src, dst)

persptrans_matrix = get_persptrans_matrix()

BASE_LINE_RATIO = 0.8
BASE_LINE_POSITION = int(CAMERA_HEIGHT_S * BASE_LINE_RATIO)
WHITE_SENSITIVITY = 30
YELLOW_SENSITIVITY = 30  # 노란색 감도 설정

def detect_lane(image):
    img = cv2.warpPerspective(image, persptrans_matrix, (CAMERA_WIDTH, CAMERA_HEIGHT))
    img_small = cv2.resize(img, dsize=(CAMERA_WIDTH_S, CAMERA_HEIGHT_S), interpolation=cv2.INTER_NEAREST)
    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    # 흰색 검출
    lower_white = np.array([0, 0, 255 - WHITE_SENSITIVITY])
    upper_white = np.array([255, WHITE_SENSITIVITY, 255])
    img_white = cv2.inRange(img_hsv, lower_white, upper_white)
    
    # 노란색 검출
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    img_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    
    # 흰색과 노란색 합치기
    img_lane = cv2.bitwise_or(img_white, img_yellow)
    
    base_line = img_lane[BASE_LINE_POSITION, :]
    
    return base_line, img_lane

def handle_signal(signum, frame):
    sys.exit(0)

if __name__ == '__main__':
    video_path = r"./Lane-lines-detection-using-Python-and-OpenCV/videoAWS_scaled.mp4"  # 비디오 파일 경로를 입력하세요
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video!")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or end of video")
            break

        # ROI 설정
        roi = frame[ROAD_VANISH_Y:CAMERA_HEIGHT, 0:CAMERA_WIDTH]

        # 전처리: 블러, 캐니 엣지
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 차선 검출 함수 호출
        base_line, img_lane = detect_lane(roi)
        
        # 차선 검출 이미지 출력
        img_lane = cv2.cvtColor(img_lane, cv2.COLOR_GRAY2BGR)
        img_lane = cv2.line(img_lane, (0, BASE_LINE_POSITION), (CAMERA_WIDTH_S - 1, BASE_LINE_POSITION), (0, 255, 0), 1)
        
        # img_lane의 크기를 roi와 동일하게 조정
        img_lane_resized = cv2.resize(img_lane, (roi.shape[1], roi.shape[0]))
        
        # 원본 프레임에 차선 검출 결과 합성
        frame_with_lane = frame.copy()
        frame_with_lane[ROAD_VANISH_Y:CAMERA_HEIGHT, 0:CAMERA_WIDTH] = img_lane_resized
        
        # 원본 영상과 차선 검출 영상을 나란히 합치기
        combined_frame = np.hstack((frame, frame_with_lane))
        
        # 화면에 출력
        cv2.imshow("Original and Lane Detection", combined_frame)
        
        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
