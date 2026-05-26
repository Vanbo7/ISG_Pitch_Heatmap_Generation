import cv2

cap = cv2.VideoCapture('game3/game3.avi')
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 1000 == 0:
        print(f"Read frame {frame_count}")

cap.release()
print(f'Total readable frames: {frame_count}')
