from ultralytics import YOLO
import os

if __name__ == '__main__':
    if not os.path.exists('validations'):
        os.makedirs('validations')

    # Load a model
    model = YOLO(r"D:\users\tam231\pothole_detection\runs\train2\weights\best.pt") # load a custom model

    # Validate the model
    metrics = model.val(data= r'D:\users\tam231\pothole_detection\pothole_dataset\data.yaml', project= r'validations')  # no arguments needed, dataset and settings remembered
    print('map', metrics.box.map)  # map50-95
    print('map50', metrics.box.map50)  # map50
    print('map75', metrics.box.map75)  # map75
    # print(metrics.box.maps)  # a list contains map50-95 of each category
