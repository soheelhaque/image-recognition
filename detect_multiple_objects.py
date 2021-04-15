from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "data/resnet50_coco_best_v2.1.0.h5"))
print("Loading model...")
detector.loadModel()

input_files = [f for f in os.listdir(os.path.join(execution_path, "images")) if f.endswith((".jpg", ".jpeg"))]
for input_file in input_files:
    print("Processing " + os.path.basename(input_file) + "...")
    input_file_path = os.path.join(execution_path, "images/" + os.path.basename(input_file))
    output_file_path = os.path.join(execution_path, "images/output/" + os.path.basename(input_file))
    detections = detector.detectObjectsFromImage(input_image=input_file_path, output_image_path=output_file_path)
    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"])
