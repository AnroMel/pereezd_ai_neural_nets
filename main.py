from detector import CrossingDetector
from config import CAMERA_SOURCE, MODEL_PATH


def main():
    detector = CrossingDetector(
        source=CAMERA_SOURCE,
        model_path=MODEL_PATH,
        camera_id="camera_1",
    )
    detector.run()


if __name__ == "__main__":
    main()
