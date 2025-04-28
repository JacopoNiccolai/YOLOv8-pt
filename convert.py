from coco2yolo.coco2yolo import convert_coco_json
import subprocess

coco = True

if coco:
	# Convert COCO JSON to YOLO format
	input_path = "data/coco2017/annotations"
	output_path = "data/coco2017/converted"

	convert_coco_json(json_dir=input_path, labels_dir=output_path, cls91to80=True)

else:
	voc2yolo_script = "voc2yolo/voc2yolo.py"
	command = ["python", voc2yolo_script]
	try:
		subprocess.run(command, check=True)
	except subprocess.CalledProcessError as e:
		print(f"Error running {voc2yolo_script}: {e}")
	
	
	
 