import cv2
from my_pipeline import process_frame

VIDEO_PATH = r"C:\Users\davit\Downloads\video.mp4"
OUTPUT_PATH = "output.mp4"


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Cannot open video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        out.write(frame)

        cv2.imshow("AI Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video saved as output.mp4")


if __name__ == "__main__":
    main()