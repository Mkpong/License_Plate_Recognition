import cv2

# 비디오 파일 경로
video_path = "video0.mp4"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 저장할 이미지 파일의 경로 및 이름 설정
image_path = "frames/frame_{}.jpg"  # 이미지 파일의 경로와 이름 패턴
frame_count = 0  # 프레임 번호 초기화

# 일정 시간 간격 설정 (여기서는 1초마다)
time_interval = 1  # 1초 (단위: 초)

# 비디오 프레임 가져오기
while cap.isOpened():
    # 비디오 프레임 읽기
    ret, frame = cap.read()

    if ret:
        # 일정 시간 간격마다 프레임 저장
        if frame_count % (time_interval * cap.get(cv2.CAP_PROP_FPS)) == 0:
            # 이미지 파일 경로 생성
            image_file = image_path.format(frame_count)

            # 프레임을 이미지로 저장
            cv2.imwrite(image_file, frame)
            print(f"프레임 {frame_count} 저장 완료.")

        frame_count += 1
    else:
        break

# 자원 해제
cap.release()
