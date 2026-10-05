import cv2

print("Opening camera with DirectShow...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("❌ Camera could not be opened")
    input("Press Enter...")
    exit()

print("✅ Camera opened successfully")
print("Press Q to quit")

while True:
    success, frame = camera.read()

    if not success:
        print("❌ Frame read failed")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()