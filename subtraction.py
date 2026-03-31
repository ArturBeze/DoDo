import argparse

import cv2
import pandas as pd
from ultralytics import YOLO

THRESHOLD = 500

def point_in_roi(cx, cy, roi):
    rx, ry, rw, rh = roi
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh

def main():
    df = pd.DataFrame(columns=["time", "state"])

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to video file', default='source/video_2.mp4')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    model = YOLO("yolov8n.pt")

    if not cap.isOpened():
        print("Ошибка открытия видео")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f"fps: {fps}, width: {width}, height: {height}, frame_count: {frame_count}")

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=False
    )

    ret, first_frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр")
        cap.release()
        return

    roi = cv2.selectROI("Выберите ROI", first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Выберите ROI")

    x, y, w, h = map(int, roi)

    if w == 0 or h == 0:
        print("ROI не выбрана")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (int(width), int(height)))

    average_count = 0
    average_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Видео закончилось или ошибка")
            break

        output = frame.copy()
        fgmask = fgbg.apply(output)



        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # gray = cv2.medianBlur(gray, 5)
        #
        # fgmask = fgbg.apply(gray)
        # _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)



        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < THRESHOLD:
                continue

            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            cx = x1 + w1 // 2
            cy = y1 + h1 // 2

            # 🔥 рисуем ВСЕ движения
            cv2.rectangle(output, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

            # 🔥 проверяем только ROI
            if point_in_roi(cx, cy, roi):
                motion_detected = True
                cv2.rectangle(output, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000

        # print(f"Текущее время: {current_time_sec:.2f} сек")

        if df.empty:
            df.loc[len(df)] = [
                current_time_sec,
                "Подход к столу" if motion_detected else "Стол пустой"
            ]
            print(f"Подход к столу: {current_time_sec:.2f} сек")
        else:
            last_row = df.iloc[-1]

            if last_row["state"] == "Подход к столу":
                if motion_detected:
                    df.loc[len(df)] = [current_time_sec, "Стол занят"]
                    print(f"Стол занят: {current_time_sec:.2f} сек")
                else:
                    if current_time_sec - last_row["time"] > THRESHOLD:
                        df.loc[len(df)] = [current_time_sec, "Стол пустой"]
                        print(f"Стол пустой: {current_time_sec:.2f} сек")

            elif last_row["state"] == "Стол занят":
                if motion_detected:
                    # df.loc[len(df)] = [current_time_sec, "Стол занят"]
                    df.iloc[-1, df.columns.get_loc("time")] = current_time_sec
                else:
                    if current_time_sec - last_row["time"] > THRESHOLD:
                        df.loc[len(df)] = [current_time_sec, "Стол пустой"]
                        print(f"Стол пустой: {current_time_sec:.2f} сек")

            elif last_row["state"] == "Стол пустой" and motion_detected:
                df.loc[len(df)] = [current_time_sec, "Подход к столу"]
                print(f"Подход к столу: {current_time_sec:.2f} сек")

                average_count = average_count + 1
                average_time = average_time + current_time_sec - last_row["time"]

        roi_state = df.iloc[-1]["state"]
        if roi_state == "Стол занят":
            roi_color = (0, 0, 255)
        elif roi_state == "Подход к столу":
            roi_color = (0, 255, 255)
        else:
            roi_color = (0, 255, 0)

        cv2.rectangle(output, (x, y), (x + w, y + h), roi_color, 2)
        text = "Motion in ROI" if motion_detected else "No motion"
        cv2.putText(output, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2)

        out.write(output)
        cv2.imshow("Frame", output)
        cv2.imshow("Mask", fgmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()