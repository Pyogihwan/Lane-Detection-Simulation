# Lane Detection Simulation

## 내부 파일 구조
- **simulation_ver2**: 시뮬레이션 파일
- **videoAWS_scaled**: 입력 영상
- **test_image_track**: 트랙 사진

이 프로젝트는 영상을 입력받아 차선 검출(Lane Detection)을 수행하는 코드로 구성되어 있습니다. 알고리즘의 단계는 다음과 같습니다:

1. **ROI (Region of Interest) 설정**
2. **Gaussian Blur**
3. **Canny Edge Detection**
4. **Perspective Transform**
5. **RGB to HSV 변환**
6. **White/Yellow Filtering**
7. **Position Calculating**
