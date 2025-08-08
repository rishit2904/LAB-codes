import cv2

video_url = '/home/student/Documents/220962436_CV/file_example_MP4_1920_18MG.mp4'
cap = cv2.VideoCapture(video_url)
ret,frame = cap.read()

while cap.isOpened():

    ret,frame = cap.read()
    if ret:
        cv2.imshow('MYVIDEO',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()