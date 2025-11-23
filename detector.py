import cv2
import torch
import json
import uuid
from datetime import datetime
from typing import Union, List, Optional, Dict, Any
import pathlib
import pandas as pd
import time

from config import (
    IMG_TARGET_WIDTH,
    FRAME_SKIP,
    PERSON_INTERVAL_SEC,
    CUSTOM_CONF,
    PEOPLE_CONF,
    DWELL_SECONDS,
    MAX_DISTANCE,
    MAX_LOST_FRAMES,
    SMOOTH_ALPHA,
    MIN_TRACK_FRAMES,
    PERSON_TRAIN_COOLDOWN_SEC,
    TRAIN_BARRIERUP_COOLDOWN_SEC,
    BARRIER_STUCK_DOWN_SEC,
)

from triggers import Detection, CROSSING_AREA
from tracker import SimpleTracker


class CrossingDetector:
    def __init__(
        self,
        source: Union[int, str],
        model_path: Optional[str] = None,
        camera_id: str = "camera_1",
    ):
        self.source = source
        self.camera_id = camera_id

        if not model_path:
            raise ValueError(
                "Нужно передать путь к кастомной модели (например, best_barrier_rail_v1_hub.pt)"
            )

        # ====== 1. КАСТОМНАЯ МОДЕЛЬ ИЗ ЛОКАЛЬНОЙ ПАПКИ yolov5 ======
        repo_dir = pathlib.Path(__file__).parent / "yolov5"

        print(f"Загружаю КАСТОМНУЮ модель из {model_path}...")
        self.model_custom = torch.hub.load(
            repo_dir.as_posix(),
            "custom",
            path=model_path,  # best_barrier_rail_v1_hub.pt
            source="local",
        )
        # порог уверенности для кастомной модели
        self.model_custom.conf = CUSTOM_CONF

        # ====== 2. СТАНДАРТНАЯ yolov5s ИЗ GITHUB ДЛЯ ЛЮДЕЙ ======
        print("Загружаю СТАНДАРТНУЮ модель yolov5s (для людей)...")
        self.model_people = torch.hub.load(
            "ultralytics/yolov5",
            "yolov5s",
            pretrained=True,
        )
        self.model_people.conf = PEOPLE_CONF

        # ====== 3. ТРЕКЕР ======
        self.tracker = SimpleTracker(
            max_distance=MAX_DISTANCE,
            max_lost_frames=MAX_LOST_FRAMES,
            dwell_seconds=DWELL_SECONDS,
            smooth_alpha=SMOOTH_ALPHA,
            min_track_frames=MIN_TRACK_FRAMES,
        )

        # ====== 4. ЛОГ-ФАЙЛ ДЛЯ СОБЫТИЙ ======
        self.log_file = pathlib.Path(__file__).parent / "events_log.jsonl"

        # ====== 5. НАСТРОЙКИ ИНТЕРВАЛОВ И РАЗМЕРА КАДРА ======
        self.person_interval_sec: float = PERSON_INTERVAL_SEC
        self.person_interval_frames: Optional[int] = None
        self.last_df_people: Optional[pd.DataFrame] = None

        # размер, до которого будем ужимать картинку для нейросети
        self.target_width: int = IMG_TARGET_WIDTH

        # пропуск кадров: 1 = каждый, 2 = каждый второй
        self.frame_skip: int = FRAME_SKIP

        # ------- состояние для доп. триггеров -------
        self.last_person_train_event_frame: Optional[int] = None
        self.last_train_barrierup_event_frame: Optional[int] = None
        self.barrier_down_no_train_frames: int = 0

    def _open_capture(self) -> cv2.VideoCapture:
        """Открываем камеру или RTSP-поток."""
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError("Не могу открыть источник видео")

        return cap

    def _detections_from_df(self, df, frame_width: int, frame_height: int):
        """Переводим pandas DataFrame от YOLO в список Detection (нормированные координаты)."""
        detections = []
        for _, row in df.iterrows():
            x_center = (row.xmin + row.xmax) / 2 / frame_width
            y_center = (row.ymin + row.ymax) / 2 / frame_height

            detections.append(Detection(
                cls=row["name"],
                x_center=float(x_center),
                y_center=float(y_center),
                conf=float(row["confidence"]),
            ))
        return detections

    def _get_crossing_area_from_rail_zone(self, df_custom, frame_width: int, frame_height: int):
        """
        Ищем в детекциях КАСТОМНОЙ модели объект rail_zone и
        превращаем его bbox в нормированные координаты (0..1).
        Чуть расширяем зону, чтобы надёжнее ловить машины.
        Если не нашли — возвращаем None.
        """
        rz = df_custom[df_custom["name"] == "rail_zone"]
        if len(rz) == 0:
            return None

        row = rz.sort_values(by="confidence", ascending=False).iloc[0]

        xmin = row.xmin / frame_width
        ymin = row.ymin / frame_height
        xmax = row.xmax / frame_width
        ymax = row.ymax / frame_height

        # чуть расширим зону (10% по краям), чтобы центр объекта легче попадал
        margin = 0.1
        xmin = max(0.0, xmin - margin)
        ymin = max(0.0, ymin - margin)
        xmax = min(1.0, xmax + margin)
        ymax = min(1.0, ymax + margin)

        return (float(xmin), float(ymin), float(xmax), float(ymax))

    def _build_event_json(self, ev: Dict[str, Any]) -> str:
        """Формируем JSON-событие для оператора."""
        event_type = ev.get("event", "unknown_event")
        cls = ev.get("cls")
        dwell_time = ev.get("dwell_time", 0.0)

        # для событий "долго в зоне" уточняем тип
        if event_type == "object_dwell_in_crossing":
            if cls == "car":
                event_type = "car_dwell_on_tracks"
            elif cls == "obstacles":
                event_type = "obstacle_on_tracks"

        # базовое описание
        if event_type == "car_dwell_on_tracks":
            description = (
                f"Автомобиль находится в зоне ж/д путей более {dwell_time:.1f} секунд"
            )
        elif event_type == "obstacle_on_tracks":
            description = (
                f"Посторонний объект находится в зоне ж/д путей более {dwell_time:.1f} секунд"
            )
        elif event_type == "person_on_tracks_with_train":
            description = "Человек(и) на ж/д путях при наличии поезда в зоне переезда"
        elif event_type == "train_with_barrier_up":
            description = "Поезд на переезде при поднятых шлагбаумах (возможная неисправность оборудования)"
        elif event_type == "barrier_stuck_down_without_train":
            description = (
                f"Шлагбаум опущен без поезда более {dwell_time:.1f} секунд "
                f"(возможная неисправность оборудования)"
            )
        else:
            description = (
                f"Объект класса '{cls}' находится в зоне ж/д путей "
                f"более {dwell_time:.1f} секунд"
            )

        event_dict = {
            "event_id": str(uuid.uuid4()),
            "type": event_type,
            "source": "crossing_ai",
            "camera_id": self.camera_id,
            "timestamp": datetime.utcnow().isoformat(),
            "track_id": ev.get("track_id"),
            "object_class": cls,
            "dwell_time_sec": round(dwell_time, 2),
            "plate": None,  # сюда потом можно подставить распознанный номер
            "description": description,
        }
        return json.dumps(event_dict, ensure_ascii=False)

    def _log_event(self, json_str: str) -> None:
        """
        Дописываем событие в локальный лог-файл в формате JSONL
        (по одному JSON-объекту в строке).
        """
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(json_str)
                f.write("\n")
        except Exception as e:
            print("⚠ Не удалось записать событие в лог:", e)

    def _extra_triggers(
        self,
        df_all: pd.DataFrame,
        crossing_area,
        frame_idx: int,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> list[dict]:
        """
        Дополнительные триггеры, которые не завязаны на трекер:
        - человек в зоне путей + поезд в кадре
        - поезд при поднятых шлагбаумах
        - шлагбаум долго опущен без поезда
        """
        events: list[dict] = []

        # если зону не удалось определить по rail_zone — берём запасную
        if crossing_area is None:
            crossing_area = CROSSING_AREA
        if crossing_area is None:
            return events

        xmin, ymin, xmax, ymax = crossing_area

        train_present = False
        barrier_down_present = False
        barrier_up_present = False
        persons_in_crossing = 0

        for _, row in df_all.iterrows():
            name = row["name"]

            if name == "train":
                train_present = True
            elif name == "barrier_down":
                barrier_down_present = True
            elif name == "barrier_up":
                barrier_up_present = True
            elif name == "person":
                # центр объекта в нормированных координатах
                cx = (row.xmin + row.xmax) / 2 / frame_width
                cy = (row.ymin + row.ymax) / 2 / frame_height
                if xmin <= cx <= xmax and ymin <= cy <= ymax:
                    persons_in_crossing += 1

        # --- 1) Человек(и) на путях + поезд ---
        if train_present and persons_in_crossing > 0:
            cooldown_frames = int(fps * PERSON_TRAIN_COOLDOWN_SEC)
            if (
                self.last_person_train_event_frame is None
                or frame_idx - self.last_person_train_event_frame >= cooldown_frames
            ):
                self.last_person_train_event_frame = frame_idx
                events.append({
                    "track_id": None,
                    "cls": "person",
                    "event": "person_on_tracks_with_train",
                    "dwell_time": 0.0,
                })

        # --- 2) Поезд при поднятых шлагбаумах (нет barrier_down) ---
        if train_present and barrier_up_present and not barrier_down_present:
            cooldown_frames = int(fps * TRAIN_BARRIERUP_COOLDOWN_SEC)
            if (
                self.last_train_barrierup_event_frame is None
                or frame_idx - self.last_train_barrierup_event_frame >= cooldown_frames
            ):
                self.last_train_barrierup_event_frame = frame_idx
                events.append({
                    "track_id": None,
                    "cls": "train",
                    "event": "train_with_barrier_up",
                    "dwell_time": 0.0,
                })

        # --- 3) Шлагбаум долго опущен без поезда ---
        if barrier_down_present and not train_present:
            self.barrier_down_no_train_frames += 1
        else:
            self.barrier_down_no_train_frames = 0

        if fps > 0 and self.barrier_down_no_train_frames >= fps * BARRIER_STUCK_DOWN_SEC:
            dwell_time = self.barrier_down_no_train_frames / fps
            events.append({
                "track_id": None,
                "cls": "barrier_down",
                "event": "barrier_stuck_down_without_train",
                "dwell_time": dwell_time,
            })
            # сбрасываем счётчик, чтобы не спамить
            self.barrier_down_no_train_frames = 0

        return events

    # ---------- основной цикл ----------

    def run(self):
        """Главный цикл: читаем кадры, делаем детекцию, трекинг, выдаём события и картинку."""
        try:
            cap = self._open_capture()
        except RuntimeError as e:
            print("❌", e)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25.0
        print(f"Используем FPS = {fps}")

        # считаем, через сколько кадров запускать модель для людей
        self.person_interval_frames = max(1, int(fps * self.person_interval_sec))
        print(f"Людей считаем каждые {self.person_interval_frames} кадров (~{self.person_interval_sec} с)")

        print("✅ Видеопоток открыт. Нажми ESC, чтобы выйти.")

        frame_idx = 0
        stats_frame_count = 0
        stats_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Не удалось прочитать кадр.")
                break

            frame_idx += 1

            # если включен пропуск кадров — обрабатываем не каждый
            if getattr(self, "frame_skip", 1) > 1 and (frame_idx % self.frame_skip != 0):
                cv2.imshow("Crossing detector (custom + people 1.5s)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # уменьшаем картинку для нейросети
            h, w, _ = frame.shape
            scale = self.target_width / w
            new_h = int(h * scale)
            frame_resized = cv2.resize(frame, (self.target_width, new_h))

            h_r, w_r, _ = frame_resized.shape

            # === ИНФЕРЕНС МОДЕЛЕЙ ===
            with torch.no_grad():
                # 1) кастомная модель — КАЖДЫЙ кадр
                results_custom = self.model_custom(frame_resized)
                df_custom = results_custom.pandas().xyxy[0]

                # 2) люди — РАЗ В N КАДРОВ
                if frame_idx % self.person_interval_frames == 0:
                    results_people = self.model_people(frame_resized)
                    df_people = results_people.pandas().xyxy[0]
                    df_people = df_people[df_people["name"] == "person"]
                    self.last_df_people = df_people
                else:
                    if frame_idx % self.person_interval_frames == 0:
                        results_people = self.model_people(frame_resized)
                        df_people = results_people.pandas().xyxy[0]
                        df_people = df_people[df_people["name"] == "person"]
                        self.last_df_people = df_people
                    else:
                        df_people = self.last_df_people
                        if df_people is None:
                            # один раз создаём и используем дальше
                            df_people = pd.DataFrame(
                                columns=["xmin", "ymin", "xmax", "ymax",
                                         "confidence", "class", "name"]
                            )
                            self.last_df_people = df_people

            # === ОБЪЕДИНЯЕМ ДЕТЕКЦИИ ===
            if len(df_custom) == 0 and len(df_people) == 0:
                df_all = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax",
                                               "confidence", "class", "name"])
            else:
                df_all = pd.concat([df_custom, df_people], ignore_index=True)

            if len(df_all) > 0:
                print("Детекции:", df_all["name"].value_counts())

            detections = self._detections_from_df(df_all, w_r, h_r)

            # rail_zone берём только из кастомной модели
            crossing_area = self._get_crossing_area_from_rail_zone(df_custom, w_r, h_r)

            # --- события от трекера (машина/объект долго в зоне) ---
            events = self.tracker.update(
                detections,
                frame_idx,
                fps,
                crossing_area=crossing_area
            )

            # --- дополнительные события по кадру ---
            extra_events = self._extra_triggers(
                df_all=df_all,
                crossing_area=crossing_area,
                frame_idx=frame_idx,
                fps=fps,
                frame_width=w_r,
                frame_height=h_r,
            )
            events.extend(extra_events)

            # --- отправляем / логируем все события ---
            for ev in events:
                json_str = self._build_event_json(ev)
                print("⚠ JSON-событие для оператора:")
                print(json_str)
                self._log_event(json_str)

            # === ОТРИСОВКА ===
            annotated_frame = results_custom.render()[0].copy()

            # дорисуем людей зелёными рамками
            for _, row in df_people.iterrows():
                x1, y1 = int(row.xmin), int(row.ymin)
                x2, y2 = int(row.xmax), int(row.ymax)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame, "person",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1
                )

            # обновляем статистику FPS
            stats_frame_count += 1
            if stats_frame_count >= 30:  # каждые 30 обработанных кадров
                now = time.time()
                dt = now - stats_start_time
                if dt > 0:
                    fps_real = stats_frame_count / dt
                    print(f"[STATS] Реальный FPS ~ {fps_real:.1f}")
                stats_frame_count = 0
                stats_start_time = now

            cv2.imshow("Crossing detector (custom + people 1.5s)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()
        print("▶ Детектор остановлен.")
