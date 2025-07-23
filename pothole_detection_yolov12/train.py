from ultralytics import YOLO

if __name__ == '__main__':

    # Load a COCO-pretrained YOLO12n model
    model = YOLO("yolo12n.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data=r"D:\users\tam231\pothole_detection\pothole_dataset\data.yaml", epochs=1000, project= r'D:\users\tam231\pothole_detection\runs', patience = 100)