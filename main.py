import argparse
from datetime import timedelta

import cv2
import pandas as pd
from ultralytics import YOLO

# временная задержка, определяет через какое время после полного отсутствия людей стол будет считаться не занятым
THRESHOLD = 15
# если confidence отслеживаемого объекта ниже, то мы его не учитываем
THRESHOLD_CONF = .15
# время за которое мы определяем подходящие люди к столу являются посетителями или же это погрешность
THRESHOLD_TO_TIME = 1
# плотность вероятности нахождения людей за заданный период времени
THRESHOLD_TO_BUSY = .15

# метод проверяет, находится ли точка с координатами cx cy
# в интересующих нас границах roi
def point_in_roi(cx, cy, roi):
    rx, ry, rw, rh = roi
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh

# метод, который возвращает цвет ячеек для таблицы excel
def color_row(row):
    if row["state"] == "Подход к столу":
        return ["background-color: lightgreen"] * len(row)
    elif row["state"] == "Стол занят":
        return ["background-color: lightcoral"] * len(row)
    elif row["state"] == "Стол пуст":
        return ["background-color: lightblue"] * len(row)
    else:
        return [""] * len(row)

# метод преобразует секунды в привычный формат HH MM SS
def seconds_to_time(seconds: int) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"

def main():
    # создаем пустую таблицу
    df = pd.DataFrame(columns=["time", "state"])

    # парсим аргумент при запуске main.py файла
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to video file', default='source/video_1.mp4')
    args = parser.parse_args()

    # открываем видео и скачиваем модель YOLO
    cap = cv2.VideoCapture(args.video)
    model = YOLO("yolov8n.pt")

    if not cap.isOpened():
        print("Ошибка открытия видео")
        exit()

    # базовые параметры видеофайла
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"fps: {fps}, width: {width}, height: {height}, frame_count: {frame_count}")

    # получаем первый кадр видео и просим пользователя указать стол для трекинга
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

    # перематывает видеофайл назад до первого кадра
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # создаем видеофайл для записи видео после обработки
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (int(width), int(height)))

    # создаем переменную для подсчета количества подходов к столу и общего времени пустования стола
    average_time = list()

    # переменные для статуса "Подход к столу"
    busy_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Видео закончилось или ошибка")
            break

        results = model.track(frame, classes=[0], persist=True, verbose=False)

        # создаем дубликат кадра
        output = frame.copy()

        # получаем с границы найденных объектов
        boxes = results[0].boxes

        # переменная будет хранить количество людей в области roi
        count = 0

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy() # координаты
            ids = boxes.id.cpu().numpy() # идентификатор
            confs = boxes.conf.cpu().numpy() # нормированная вероятность

            for box, track_id, conf in zip(xyxy, ids, confs):
                x1, y1, x2, y2 = map(int, box)

                # центр объекта трекинга
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # если объект находится в области roi рисуем его границы, центр и идентификатор,
                # а также увеличиваем счетчик найденных людей count
                if point_in_roi(cx, cy, roi) and conf > THRESHOLD_CONF:
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(output, (cx, cy), 4, (0, 255, 0), -1)

                    label = f"ID {int(track_id)}  {conf:.2f}"
                    cv2.putText(output, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    count += 1

        # логирование текущего времени
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000

        # если список пуст добавляем статус и время в зависимости от
        # того занят стол посетителем или нет
        if df.empty:
            df.loc[len(df)] = [
                current_time_sec,
                "Подход к столу" if count > 0 else "Стол пустой"
            ]
            print(f"{'Подход к столу' if count > 0 else 'Стол пустой'} {current_time_sec:.2f} сек")
        # если в таблице уже есть какие-либо данные смотрим предыдущий статус,
        # в зависимости от него и данных с трекинга текущего кадра принимаем решение
        # менять статус или нет.
        else:
            last_row = df.iloc[-1]

            if last_row["state"] == "Подход к столу":
                # если через THRESHOLD_TO_BUSY секунд посетитель появляется чаще чем нормированное значение THRESHOLD_TO_BUSY
                # переходим в статус "Стол занят" в противном случае в статус "Стол пустой" и добавляем к счетчику времени THRESHOLD_TO_TIME
                if current_time_sec - last_row["time"] > THRESHOLD_TO_TIME:
                    if busy_frames / total_frames >= THRESHOLD_TO_BUSY:
                        df.loc[len(df)] = [current_time_sec, "Стол занят"]
                        print(f"Смена статуса. Стол занят: {current_time_sec:.2f} сек")
                    else:
                        df.loc[len(df)] = [current_time_sec, "Стол пустой"]
                        print(f"Смена статуса. Стол пустой: {current_time_sec:.2f} сек")
                        average_time.append(THRESHOLD_TO_TIME)

                    busy_frames = 0
                    total_frames = 0
                else:
                    # если время THRESHOLD_TO_BUSY еще не прошло считаем все кадры и кадры, где есть люди
                    if count > 0:
                        busy_frames = busy_frames + 1
                    total_frames = total_frames + 1

            elif last_row["state"] == "Стол занят":
                if count > 0:
                    df.iloc[-1, df.columns.get_loc("time")] = current_time_sec
                else:
                    if current_time_sec - last_row["time"] > THRESHOLD:
                        df.loc[len(df)] = [current_time_sec, "Стол пустой"]
                        print(f"Смена статуса. Стол пустой: {current_time_sec:.2f} сек")

            elif last_row["state"] == "Стол пустой":
                # если статус меняется с "Стол пустой" на "Подход к столу"
                # или если это последний кадр видео и стол пуст
                # увеличиваем время пустования стола
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    print(f"Конец видео: {current_time_sec:.2f} сек")
                    average_time.append(current_time_sec - last_row["time"])
                else:
                    if count > 0:
                        df.loc[len(df)] = [current_time_sec, "Подход к столу"]
                        print(f"Смена статуса. Подход к столу: {current_time_sec:.2f} сек")
                        average_time.append(current_time_sec - last_row["time"])

        # определяем цвет рамки в зависимости от текущего статуса
        roi_state = df.iloc[-1]["state"]
        if roi_state == "Стол занят":
            roi_color = (0, 0, 255)
            st = "busy"
        elif roi_state == "Подход к столу":
            roi_color = (0, 255, 255)
            st = "ready"
        elif roi_state == "Стол пустой":
            roi_color = (0, 255, 0)
            st = "empty"
        else:
            roi_color = (255, 255, 255)
            st = "unknown"

        # рисуем рамку roi
        cv2.rectangle(output, (x, y), (x + w, y + h), roi_color, 2)
        text = "Motion in ROI." if count > 0 else "No motion"
        cv2.putText(output, f"{text}. State {st}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2)

        # сохраняем видео и показываем картинку
        out.write(output)
        cv2.imshow("Frame", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # закрываем считывание и запись видео. Закрываем окно
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # считаем среднее время которое стол был пуст
    if len(average_time) != 0:
        print(f"Customers: {len(average_time)}")
        print(f"Average time: {seconds_to_time(int(sum(average_time)/ len(average_time)))}")

    # сохраняем данные из таблицы pandas в файл excel
    # df.to_excel("output.xlsx", index=False)

    # если нужно чтобы строки в таблице были разных цветов в зависимости от статуса
    styled = df.style.apply(color_row, axis=1)
    styled.to_excel("output.xlsx", index=False)

if __name__ == '__main__':
    main()