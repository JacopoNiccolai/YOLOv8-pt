from coco2yolo.general_json2yolo import convert_coco_json

input_path = "data/coco2017/annotations"
output_path = "data/coco2017/converted"

convert_coco_json(json_dir=input_path, labels_dir=output_path, cls91to80=True)
	