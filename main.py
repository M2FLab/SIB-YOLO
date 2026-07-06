from ultralytics import YOLO

if __name__ == '__main__':
    import os
    config_path = os.path.join(os.path.dirname(__file__), 'SIB-YOLO.yaml')
    model = YOLO(config_path)
    model.train(
        data=r"Dronevehicle.yaml",
        epochs=200,
        batch=16,
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        imgsz=640,
        workers=8,
        patience=100,
        close_mosaic=10,
        amp=True,
        verbose=True
    )



