import os
import random

trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = r'./data/Annotations'         #Labelimg 标注的xml标签位置
txtsavepath = r'./data/ImageSets'           #trainval、test、train、val所需要的文件名 存储的位置
total_xml = os.listdir(xmlfilepath)         #xml文件的list

num = len(total_xml)                #将Labelimg 标注的image图片 按比例分成trainval、test、train、val所需要的数据集
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(r'./data/ImageSets/trainval.txt', 'w')
ftest = open(r'./data/ImageSets/test.txt', 'w')
ftrain = open(r'./data/ImageSets/train.txt', 'w')
fval = open(r'./data/ImageSets/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
print("------------voc_generate.py-finish------------------")