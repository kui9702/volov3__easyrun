# volov3__easyrun
environment：
    python=3.6
    tensorflow=1.13.1 or tensorflow-gpu=1.13.1
    keras=2.2.4
    matplotlib
    pillow
    and so on

voc_generate.py：
    将Labelimg 标注的image图片 按比例分成trainval、test、train、val所需要的数据集，获得分配完的trainval、test、train、val所需要的文件名

voc_annotation.py：
    解析xml文件，将数据转化为txt

kmeans.py：
     输入上面得到的txt文件，通过聚类得到数据最佳anchors

train.py：
     进行yolov3训练的文件

yolo.py：
     构建以yolov3为底层构件的yolo检测模型，因为上面的yolov3还是分开的单个函数，功能并没有融合在一起，即使在训练的时候所有的yolov3组件还是分开的功能，并没有统一接口，供在模型训练完成之后，直接使用。通过yolo.py融合所有的组件。

yolov3.cfg：
     构建yolov3检测模型的整个超参文件

