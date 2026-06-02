put original (uncropped) images dataset here along with labels folder with annotation.csv inside it.


Folder structure expected:
crop_img_from_json
	images
		images
			train_00000.jpeg
			train_00001.jpeg
			train_00002.jpeg
			...
	labels
		annotation.csv
	crop_detections.py
	generate_crop_annotation.py
	test_rf-detr_predictions-detection-only.json
	test_yolo_predictions-detection-only

run crop_detections.py then generate_crop_annotation.py to crop the dataset according to json bounding boxes, then to generate cropped_annotations.csv (both datasets rf-detr and yolo is generated)