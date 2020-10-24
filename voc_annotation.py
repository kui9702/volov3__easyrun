import xml.etree.ElementTree as ET
from os import getcwd

'''
    解析xml文件
'''


sets=['train','val','test']

classes = ["Car"]


def convert_annotation(image_id, list_file):
    in_file = open('./data/Annotations/%s.xml'%(image_id),'rb')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):         #解析xml文件的节点
        difficult = obj.find('difficult').text      #difficult指目标是否难以辨别
        cls = obj.find('name').text         #标记对象
        if cls not in classes or int(difficult)==1:     #标记对象是否在我们所需要的类里面，或者标记目标是否难以辨别
            continue    
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')             #标记的框坐标
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))    #将坐标b以","连接起来

wd = getcwd()

for image_set in sets:
    image_ids = open('./data/ImageSets/%s.txt'%(image_set)).read().strip().split()      #将读取的文件变成数组
    list_file = open(r'./yolov3_config/%s.txt'%(image_set), 'w')
    for image_id in image_ids:                              #将文件名 以及该文件的标注目标的坐标，类型写入文件
        list_file.write('./data/images/%s.jpg'%(image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
print("-------voc_annotation.py-finish------------------")

