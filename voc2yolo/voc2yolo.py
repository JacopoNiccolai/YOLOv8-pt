import os
import xml.etree.ElementTree


def voc2yolo(xml_file, label_dir, convert_dir, names):
    in_file = open(f'{label_dir}/{xml_file}')

	# if not exists create convert_dir
    if not os.path.exists(convert_dir):
        os.makedirs(convert_dir)

    root = xml.etree.ElementTree.parse(in_file).getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    has_class = False
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name in names:
            has_class = True
    if has_class:
        out_file = open(f'{convert_dir}/{xml_file[:-4]}.txt', 'w')
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name in names:
                xml_box = obj.find('bndbox')
                x_min = float(xml_box.find('xmin').text)
                y_min = float(xml_box.find('ymin').text)
                x_max = float(xml_box.find('xmax').text)
                y_max = float(xml_box.find('ymax').text)

                box_x = (x_min + x_max) / 2.0 - 1
                box_y = (y_min + y_max) / 2.0 - 1
                box_w = x_max - x_min
                box_h = y_max - y_min
                box_x = box_x * 1. / w
                box_w = box_w * 1. / w
                box_y = box_y * 1. / h
                box_h = box_h * 1. / h

                b = [box_x, box_y, box_w, box_h]
                cls_id = names.index(obj.find('name').text)
                out_file.write(str(cls_id) + " " + " ".join([str(f'{a:.6f}') for a in b]) + '\n')


if __name__ == '__main__':
    
    label_dir = 'data/voc2012/VOC2012_test/Annotations'
    convert_dir = 'data/voc2012/VOC2012_test/converted'
    image_dir = 'data/voc2012/VOC2012_test/JPEGImages'
    names = [
		"aeroplane",
		"bicycle",
		"bird",
		"boat",
		"bottle",
		"bus",
		"car",
		"cat",
		"chair",
		"cow",
		"diningtable",
		"dog",
		"horse",
		"motorbike",
		"person",
		"pottedplant",
		"sheep",
		"sofa",
		"train",
		"tvmonitor",
	]
    
    print('VOC to YOLO')
    xml_files = [name for name in os.listdir(label_dir) if name.endswith('.xml')]
    
    for xml_file in xml_files:
        voc2yolo(xml_file, label_dir, convert_dir, names)
    
    