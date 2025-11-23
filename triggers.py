from dataclasses import dataclass


@dataclass
class Detection:
    cls: str
    x_center: float  # 0..1
    y_center: float  # 0..1
    conf: float


# Запасная зона путей (если rail_zone временно не детектится)
# xmin, ymin, xmax, ymax в нормированных координатах [0..1]
CROSSING_AREA = (0.3, 0.4, 0.7, 0.8)


def in_area(obj, area) -> bool:
    """
    Проверяем, находится ли объект в области.
    Если у объекта есть сглаженные координаты smooth_x/smooth_y — используем их.
    Иначе берём x_center/y_center.
    """
    xmin, ymin, xmax, ymax = area

    x = getattr(obj, "smooth_x", getattr(obj, "x_center"))
    y = getattr(obj, "smooth_y", getattr(obj, "y_center"))

    return xmin <= x <= xmax and ymin <= y <= ymax

