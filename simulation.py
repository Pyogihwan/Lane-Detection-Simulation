#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
< Ubuntu 버전에 따른 코드 수정 사항 >
Ubuntu 18.04 이하 (ROS melodic, kinetic 등)을 사용하면
첫 번째 줄의 주석을 #! /usr/bin/env python으로,
Ubuntu 20.04 이상을 사용하면 #! /usr/bin/env python3을 사용해야 한다.
'''

import time
import sys
import os
import signal
import numpy as np
import cv2

# ----------------------------------------
# Lane Dicision
# -> Decide lane using heuristic algorithm
# ----------------------------------------

class Lane:
    
    from enum import Enum
    
    class State(Enum):
        STATE_NONE = 0
        STATE_LEFT = 1
        STATE_RIGHT = 2
    
    def __init__(self, data_width, lane_width, lane_width_min, score_lower_bound, center_range):
        """
        차선 위치 상태를 유지하고 업데이트하기 위한 클래스

        Args:
            data_width (int): 입력 데이터의 길이\n
            lane_width (int): 차선의 일반적인 폭\n
            lane_width_min (int): 차선의 폭 하한선\n
            score_lower_bound (int): 점수 임계치가 가지는 하한선\n
            center_range (int): 직진 구간을 판단하기 위한 범위 인자\n
        """
        self.__data_width = data_width
        self.__pos_l, self.__pos_r = 0, data_width - 1
        self.__pos = 0
        self.__state = self.State.STATE_NONE
        
        self.__lane_width = lane_width
        self.__lane_width_min = lane_width_min

        self.__score_lower_bound = score_lower_bound
        self.__center_range = center_range
    
    @property
    def left(self):
        """
        Returns:
            int: 왼쪽 차선의 위치
        """
        return self.__pos_l
    
    @property
    def right(self):
        """
        Returns:
            int: 오른쪽 차선의 위치
        """
        return self.__pos_r
    
    @property
    def position(self):
        """
        Returns:
            int: 차선의 position
        """
        return self.__pos
    
    @property
    def state(self):
        return self.__state
    
    @property
    def score_lower_bound(self):
        return self.__score_lower_bound
    
    @score_lower_bound.setter
    def score_lower_bound(self, value):
        if value > 0:
            raise ValueError("점수 하한선은 0 또는 음수여야 합니다.")
        
        self.__score_lower_bound = value
        
    from sys import maxsize as __inf
    
    def update(self, positions):
        """
        y=-abs(curr_x - prev_x) 휴리스틱 함수를 이용하여 검출된 차선 중
        이전에 결정된 차선에 가장 가까운 차선의 위치를 고르도록 하는 함수\n
        
        초기에는 먼 거리에 존재하는 차선도 인식시키기 위해(차량이 보고 있는 차선은 매번 다르므로)
        하한선 개념을 도입하여 차선이 검출될 때마다 하한선을 조금식 올리며 점차 그 간극을 줄여나간다. \n

        Args:
            positions (_type_): _description_
        """
        
        # score는 가능한 가장 낮은 점수의 값으로 설정
        score_l, pos_l = -self.__inf - 1, None
        score_r, pos_r = -self.__inf - 1, None
        
        # positions 리스트(또는 np.array)를 순회하여 가장 점수가 큰 위치 결정
        for pos, valid in enumerate(positions):
            if not valid: continue
            
            score_l_cand = -abs(pos - self.__pos_l)
            score_r_cand = -abs(pos - self.__pos_r)
            
            if score_l_cand > score_l:
                score_l = score_l_cand
                pos_l = pos
            if score_r_cand > score_r:
                score_r = score_r_cand
                pos_r = pos
                
        # score가 유효한 값인지 확인: 너무 작은 score는 무시한다
        detect_l = score_l > self.__score_lower_bound
        detect_r = score_r > self.__score_lower_bound
                
        # 차선 결정
        if detect_l:
            # 경우 1: 왼쪽과 오른쪽 차선 둘 다 검출됐을 때
            if detect_r:
                # 두 차선이 너무 가깝거나, 역전됐을 때
                if (pos_r - pos_l) < self.__lane_width_min:
                    if score_l > score_r:
                        pos_r = pos_l + self.__lane_width_min
                    else:
                        pos_l = pos_r - self.__lane_width_min
            # 경우 2: 왼쪽 차선만 검출됐을 때 (보통 우회전인 경우)
            else:
                # 현재 상태가 왼쪽일 때 - 유효하지 않은 값
                if self.__state == self.State.STATE_LEFT: pos_l = self.__pos_l
                pos_r = pos_l + self.__lane_width
        else:
            # 경우 3: 오른쪽 차선만 검출됐을 때 (보통 좌회전인 경우)
            if detect_r:
                # 현재 상태가 오른쪽일 때 - 유효하지 않은 값
                if self.__state == self.State.STATE_RIGHT: pos_r = self.__pos_r
                pos_l = pos_r - self.__lane_width
            # 경우 4: 두 차선 모두 다 검출되지 않았을 때
            else:
                # 이전 차선 정보를 사용한다
                pos_l = self.__pos_l
                pos_r = self.__pos_r
        
        # Update Lane Information
        self.__pos_l = pos_l
        self.__pos_r = pos_r
        self.__pos = pos_l + pos_r - self.__data_width
        
        # Update State
        if self.__pos < -self.__center_range:
            self.__state = self.State.STATE_LEFT
        elif self.__pos > self.__center_range:
            self.__state = self.State.STATE_RIGHT
        else:
            self.__state = self.State.STATE_NONE


# ----------------------------------------
# Lane Detection
# -> Detect lane using OpenCV
# ----------------------------------------

CAMERA_FPS = 30
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
CAMERA_WIDTH_S, CAMERA_HEIGHT_S = CAMERA_WIDTH // 8, CAMERA_HEIGHT // 8

ROAD_VANISH_Y = 220 # 영상에서 도로의 소실점 y 값
ROAD_MARGIN_X = 190 # 영상의 아래 끝에서 가로 길이(CAMERA_WIDTH)로부터 떨어진 차선의 거리
ROAD_MARGIN_Y = 280 # 소실점보다 아래에 있는 어느 적당한 y 값

def get_persptrans_matrix():
    """
    cv2.warpPerspective() 함수에 사용할 Perspective Transform Matrix를 반환하는 함수
    
    Returns:
        Any: Perspective Transform Matrix
    """
    
    # 영상에서 도로의 소실점 중 y 값
    vanish_y = ROAD_VANISH_Y
    
    # 영상의 가장 밑에 위치한 차선의 여백
    # 카메라 영상을 잘 관찰해보면, 영상의 가장 아래쪽에는 차선이 보이지 않는 것을 확인할 수 있다.
    # 보이지 않는 차선에 y축 방향으로 margin_x 만큼 양쪽으로 늘려 차선의 두 끝점을 지정하는데 사용한다.
    margin_x = ROAD_MARGIN_X

    # vanish_y와 margin_x가 주어졌을 때, y좌표에 따른 x좌표를 계산해보자.
    # 소실점과 오른쪽 차선의 끝점을 구하면 아래와 같다.
    # (CAMERA_WIDTH / 2, vanish_y), (CAMERA_WIDTH + margin_x, CAMERA_HEIGHT)
    # 위의 두 점을 이용하여 직선의 방정식을 구해보자.
    # dx = CAMERA_WIDTH / 2 + margin_x
    # dy = CAMERA_HEIGHT - vanish_y
    # y = (dy / dx) * (x - CAMERA_WIDTH / 2) + vanish_y
    # dx * y = (dy * x) - (dy * CAMERA_WIDTH / 2) + (vanish_y * dx)
    # (dy * x) = (dx * y) + (dy * CAMERA_WIDTH / 2) - (vanish_y * dx)
    # x = (dx / dy * y) + (CAMERA_WIDTH / 2) - (vanish_y * dx / dy)
    #   = (dx / dy) * (y - vanish_y) + (CAMERA_WIDTH / 2)
    
    dx = CAMERA_WIDTH / 2 + margin_x
    dy = CAMERA_HEIGHT - vanish_y
    
    # 실제로 영상에서는 소실점은 보이지 않고, 소실점과 가까운 두 차선의 양끝이 보인다.
    # 이러한 두 차선의 양끝을 정의한다.
    margin_y = ROAD_MARGIN_Y
    
    xr = (dx / dy) * (margin_y - vanish_y) + (CAMERA_WIDTH / 2)
    xl = CAMERA_WIDTH - xr
    
    src = np.array([(xl, margin_y),
                    (xr, margin_y),
                    (-margin_x, CAMERA_HEIGHT),
                    (CAMERA_WIDTH + margin_x, CAMERA_HEIGHT)], dtype=np.float32)
    dst = np.array([(0, 0),
                    (CAMERA_WIDTH, 0),
                    (0, CAMERA_HEIGHT),
                    (CAMERA_WIDTH, CAMERA_HEIGHT)], dtype=np.float32)
    
    return cv2.getPerspectiveTransform(src, dst)

persptrans_matrix = get_persptrans_matrix()

BASE_LINE_RATIO = 0.8
BASE_LINE_POSITION = int(CAMERA_HEIGHT_S * BASE_LINE_RATIO)
WHITE_SENSITIVITY = 30

def detect_lane(image, lane=None):
    """
    OpenCV를 이용하여 자선을 검출하고 정해진 가로선(base line)을 추출하여
    1차원 차선 데이터를 반환하는 함수\n
    왼쪽 차선과 오른쪽 차선을 구분하지는 않고, 오직 '차선의 위치'만을 검출하기에
    Lane 클래스를 이용하여 휴리스틱하게 왼쪽, 오른쪽 차선을 검출하여야 한다.

    Args:
        image (Any): 영상 데이터
        lane (Lane, optional): 차선 데이터,
                               이 값이 주어지면 Lane 인스턴스에서 구한 차선을 화면에 띄워준다.

    Returns:
        np.array: 1차원 차선 데이터
    """
    
    # perspective transform: 항공뷰 구하기 
    img = cv2.warpPerspective(image, persptrans_matrix, (CAMERA_WIDTH, CAMERA_HEIGHT))
    
    # resize: 연산량 최소화를 위해 이미지 크기 줄이기 (여기선 가로가 80으로 줄어듬)
    img_small = cv2.resize(img, dsize=(CAMERA_WIDTH_S, CAMERA_HEIGHT_S), interpolation=cv2.INTER_NEAREST)
    
    # cvtColor: BGR 색공간을 HSV 색공간으로 변환
    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    # inRange: 흰색 픽셀만 검출
    lower_white = np.array([0, 0, 255 - WHITE_SENSITIVITY])
    upper_white = np.array([255, WHITE_SENSITIVITY, 255])
    img_white = cv2.inRange(img_hsv, lower_white, upper_white)
    
    # 기준선(가로)에 대해 차선이 존재하는 위치 구하기
    base_line = img_white[BASE_LINE_POSITION, :]
    
    # 기준선에 lane 정보 붙여서 화면에 띄워주기
    if lane:
        img_lane = img_white.copy()
        img_lane = cv2.cvtColor(img_lane, cv2.COLOR_GRAY2BGR)
        img_lane = cv2.line(img_lane, (0, BASE_LINE_POSITION), (CAMERA_WIDTH_S - 1, BASE_LINE_POSITION), (0, 255, 0), 1)
        img_lane = cv2.line(img_lane, (lane.left, BASE_LINE_POSITION), (lane.left, BASE_LINE_POSITION), (0, 0, 255), 1)
        img_lane = cv2.line(img_lane, (lane.right, BASE_LINE_POSITION), (lane.right, BASE_LINE_POSITION), (255, 0, 0), 1)
        img_lane = cv2.resize(img_lane, dsize=(CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Lane Image", img_lane)
        cv2.waitKey(1)
    
    return base_line

def handle_signal():
    """
    SIGINT 시그널이 들어와 프로그램 실행을 끝낼 때
    그 처리시간을 줄이기 위한 함수
    """
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)

def main():
    # 이미지 파일 경로
    image_path = r"./Lane-lines-detection-using-Python-and-OpenCV/test_images_track/test05.jpg"
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    import os
    if not os.path.exists(image_path):
        print("경로에 파일이 존재하지 않습니다.")
        return
    elif image is None:
        print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return
    else:
        height, width, channels = image.shape
        print(f"Image Size: Width={width}, Height={height}, Channels={channels}")
    
    
    cv2.imshow("Lane Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    lane = Lane(
        data_width=CAMERA_WIDTH_S, center_range=20,
        lane_width=60, lane_width_min=24,
        score_lower_bound=-24, 
    )
    """
    Args:
    data_width (int): 입력 데이터의 길이\n
    lane_width (int): 차선의 일반적인 폭\n
    lane_width_min (int): 차선의 폭 하한선\n
    score_lower_bound (int): 점수 임계치가 가지는 하한선\n
    center_range (int): 직진 구간을 판단하기 위한 범위 인자\n
    """
    detect_lane(image, lane)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()