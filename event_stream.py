"""
event_stream.py
---------------
Утилита для "подписки" на события, которые пишет detector.py в events_log.jsonl.
Каждая строка файла — отдельный JSON-объект.
"""

import json
import time
import os
from pathlib import Path
from typing import Iterator, Dict, Any

# путь к лог-файлу с событиями
LOG_PATH = Path(__file__).parent / "events_log.jsonl"


def follow_events(poll_interval: float = 0.2) -> Iterator[Dict[str, Any]]:
    """
    "Подписка" на новые события из файла events_log.jsonl.

    Работает как tail -f:
    - доходим до конца файла,
    - ждём новые строки,
    - как только появилась строка — отдаём dict.

    Использование:
        for event in follow_events():
            print(event)
    """
    LOG_PATH.touch(exist_ok=True)

    with LOG_PATH.open("r", encoding="utf-8") as f:
        # перематываемся в конец файла — нас интересуют только новые события
        f.seek(0, os.SEEK_END)

        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_interval)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # кривую строку просто пропускаем
                continue

            yield event


# небольшой тест: можно запустить python event_stream.py и смотреть события в консоли
if __name__ == "__main__":
    print("Слушаю события из events_log.jsonl ... Ctrl+C чтобы выйти")
    try:
        for ev in follow_events():
            print(
                f"[EVENT] {ev.get('timestamp')} | {ev.get('camera_id')} | "
                f"{ev.get('type')} | {ev.get('description')}"
            )
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
