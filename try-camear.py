import cv2

address = "udp://127.0.0.1:11111"

cap = cv2.VideoCapture(address)

if not cap.isOpened():
    print("未接收到数据流，请确认数据流是否发送到本机！")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
    else:
        print('等待数据......')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()