from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import math

from triggers import Detection, in_area, CROSSING_AREA
from config import (
    HYSTERESIS_IN_FRAMES,
    HYSTERESIS_OUT_FRAMES,
)


@dataclass
class TrackedObject:
    track_id: int
    cls: str
    x_center: float      # "сырые" координаты центра (0..1)
    y_center: float
    smooth_x: float      # сглаженные координаты центра (0..1)
    smooth_y: float
    first_frame: int     # на каком кадре впервые увидели
    last_frame: int      # на каком кадре обновляли последний раз
    entered_frame: Optional[int] = None  # на каком кадре вошёл в зону
    in_crossing: bool = False            # сейчас в зоне?
    reported: bool = False               # уже отправляли триггер?

    # новые поля для гистерезиса
    inside_frames: int = 0               # сколько кадров подряд внутри зоны
    outside_frames: int = 0              # сколько кадров подряд снаружи


class SimpleTracker:
    """
    Простой трекер:
    - связывает детекции между кадрами по расстоянию,
    - сглаживает траекторию (чтобы объекты не "скакали"),
    - отслеживает, когда объект зашёл/вышел из области путей с учётом гистерезиса,
    - считает время нахождения в зоне и генерирует события.
    """

    def __init__(
        self,
        max_distance: float = 0.05,
        max_lost_frames: int = 30,
        dwell_seconds: float = 1.0,
        smooth_alpha: float = 0.5,
        min_track_frames: int = 3,
    ):
        """
        max_distance    — максимум расстояния (в нормированных координатах), чтобы считать детекцию тем же объектом
        max_lost_frames — через сколько кадров без обновления трек удаляется
        dwell_seconds   — сколько секунд объект должен находиться в зоне, чтобы сработал триггер
        smooth_alpha    — коэффициент сглаживания (0..1), ближе к 1 — меньше сглаживание
        min_track_frames— минимальное количество кадров жизни трека, чтобы учитывать его в триггерной логике
        """
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.dwell_seconds = dwell_seconds
        self.smooth_alpha = smooth_alpha
        self.min_track_frames = min_track_frames

        self.next_id: int = 1
        self.tracks: Dict[int, TrackedObject] = {}

    def _match_detection_to_track(self, det: Detection) -> Optional[int]:
        """
        Находит ID ближайшего трека того же класса, если он не дальше max_distance.
        """
        best_id: Optional[int] = None
        best_dist = self.max_distance

        for tid, track in self.tracks.items():
            if track.cls != det.cls:
                continue

            dist = math.hypot(
                det.x_center - track.x_center,
                det.y_center - track.y_center,
            )

            if dist < best_dist:
                best_dist = dist
                best_id = tid

        return best_id

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        fps: float,
        crossing_area=None,
    ) -> List[Dict[str, Any]]:
        """
        Обновляем треки по текущим детекциям.
        crossing_area — зона путей из rail_zone (если есть),
        иначе используем запасную CROSSING_AREA.
        Возвращает список событий (триггеров).
        """
        events: List[Dict[str, Any]] = []

        if crossing_area is None:
            crossing_area = CROSSING_AREA

        matched_track_ids = set()

        # === 1. Обновляем / создаём треки по детекциям ===
        for det in detections:
            # трекаем только те объекты, которые нас интересуют
            if det.cls not in ("car", "obstacles"):
                continue

            track_id = self._match_detection_to_track(det)

            if track_id is None:
                # создаём новый трек
                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = TrackedObject(
                    track_id=track_id,
                    cls=det.cls,
                    x_center=det.x_center,
                    y_center=det.y_center,
                    smooth_x=det.x_center,
                    smooth_y=det.y_center,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                )
            else:
                # обновляем существующий трек
                tr = self.tracks[track_id]
                tr.x_center = det.x_center
                tr.y_center = det.y_center
                # экспоненциальное сглаживание координат
                tr.smooth_x = (
                    self.smooth_alpha * det.x_center
                    + (1.0 - self.smooth_alpha) * tr.smooth_x
                )
                tr.smooth_y = (
                    self.smooth_alpha * det.y_center
                    + (1.0 - self.smooth_alpha) * tr.smooth_y
                )
                tr.last_frame = frame_idx

            matched_track_ids.add(track_id)

        # === 2. Удаляем треки, которые давно не обновлялись ===
        to_delete = [
            tid
            for tid, tr in self.tracks.items()
            if frame_idx - tr.last_frame > self.max_lost_frames
        ]
        for tid in to_delete:
            del self.tracks[tid]

        # === 3. Обрабатываем треки: в зоне / не в зоне, считаем время, генерируем события ===
        for tid, tr in self.tracks.items():
            # игнорируем совсем короткие треки (шум)
            track_len_frames = frame_idx - tr.first_frame + 1
            if track_len_frames < self.min_track_frames:
                continue

            # используем сглаженные координаты (in_area сам возьмёт smooth_x/smooth_y)
            now_in_crossing = in_area(tr, crossing_area)

            # обновляем счётчики для гистерезиса
            if now_in_crossing:
                tr.inside_frames += 1
                tr.outside_frames = 0
            else:
                tr.outside_frames += 1
                tr.inside_frames = 0

            # вошёл в зону (с учётом гистерезиса)
            if (
                now_in_crossing
                and not tr.in_crossing
                and tr.inside_frames >= HYSTERESIS_IN_FRAMES
            ):
                tr.in_crossing = True
                tr.entered_frame = frame_idx
                tr.reported = False
                print(
                    f"[TRACK] id={tid}, cls={tr.cls} ВОШЁЛ в зону "
                    f"на кадре {frame_idx} (inside_frames={tr.inside_frames})"
                )

            # вышел из зоны (с учётом гистерезиса)
            if (
                not now_in_crossing
                and tr.in_crossing
                and tr.outside_frames >= HYSTERESIS_OUT_FRAMES
            ):
                print(
                    f"[TRACK] id={tid}, cls={tr.cls} ВЫШЕЛ из зоны "
                    f"на кадре {frame_idx} (outside_frames={tr.outside_frames})"
                )
                tr.in_crossing = False
                tr.entered_frame = None
                tr.reported = False

            # если объект в зоне и момент входа известен — считаем время
            if tr.in_crossing and tr.entered_frame is not None and not tr.reported:
                frames_inside = frame_idx - tr.entered_frame
                dwell_time = frames_inside / fps if fps > 0 else 0.0

                print(
                    f"[TRACK] id={tid}, cls={tr.cls}, "
                    f"frames_inside={frames_inside}, t={dwell_time:.2f}s"
                )

                if dwell_time >= self.dwell_seconds:
                    tr.reported = True
                    print(
                        f"[EVENT] СРАБОТАЛ триггер для id={tid}, "
                        f"cls={tr.cls}, t={dwell_time:.2f}s"
                    )

                    events.append({
                        "track_id": tid,
                        "cls": tr.cls,
                        "event": "object_dwell_in_crossing",
                        "dwell_time": dwell_time,
                    })

        return events
