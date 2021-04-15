from imageai.Detection import ObjectDetection
import os


def get_detector():
    object_detector = ObjectDetection()
    object_detector.setModelTypeAsRetinaNet()
    object_detector.setModelPath(os.path.join(execution_path, "data/resnet50_coco_best_v2.1.0.h5"))
    object_detector.loadModel()
    return object_detector


def detect_objects(detector, input_file, output_file):
    detections = detector.detectObjectsFromImage(input_image=input_file, output_image_path=output_file)
    return detections


if __name__=="__main__":
    execution_path = os.getcwd()

    print("Loading model...")
    retina_net_detector = get_detector()

    input_image_files = [f for f in os.listdir(os.path.join(execution_path, "images")) if f.endswith((".jpg", ".jpeg"))]
    for input_image_file in input_image_files:
        print("Processing " + os.path.basename(input_image_file) + "...")
        input_file_path = os.path.join(execution_path, "images/" + os.path.basename(input_image_file))
        output_file_path = os.path.join(execution_path, "images/output/" + os.path.basename(input_image_file))

        detected_objects = detect_objects(retina_net_detector, input_file_path, output_file_path)

        for detected_object in detected_objects:
            print(detected_object["name"], " : ", detected_object["percentage_probability"])

        print("Finished. Output sent to " + output_file_path)
