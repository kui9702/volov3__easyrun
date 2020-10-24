import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import keras

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import os
import tensorflow as tf
import config

def _main():
    annotation_path = r'./yolov3_config/train.txt'
    model_path = config.model_save_path
    config.dir_exist(model_path)
    classes_path = 'yolov3_config/need_classes.txt'     #需要检测的类别
    anchors_path = r'yolov3_config\anchors.txt'         #保存的锚点信息
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)                 #9个anchor box
    input_shape = config.img_size                           #图片的尺寸 416*416
    model = create_model(input_shape, anchors, len(class_names) )
    train(model, annotation_path, input_shape, anchors, len(class_names), model_path=model_path)

#回调函数，自动停止训练，保存最好的模型
class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # on_epoch_end: logs include `acc` and `loss`, and
        #    logs[]包括'acc'和'loss',还有'val_loss'、'val_acc'
        print("----train logs epoch accuracy----",
              logs['loss'], '-------------------------------', sep='\n')
        if logs['loss'] < 1000 and logs['val_loss'] <1000:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print('------train end------')

def train(model, annotation_path, input_shape, anchors, num_classes, model_path='logs/'):
    model.compile(optimizer='adam', loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})
    # logging = TensorBoard(model_path=model_path)
    checkpoint = ModelCheckpoint(model_path + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    batch_size = config.batch_size
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))


    each_epoch_callback = CustomCallback()

    model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=1000,
            initial_epoch=0,
            callbacks=[each_epoch_callback])
    model.save_weights(model_path + 'trained_weights.h5')

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


#input_shape:图片尺寸 ;anchors:9个anchor box ;num_classes: 类别数; load_pretrained: 是否加载预训练模型 
#   freeze_body：冻结模式，false是冻结DarkNet53的层，true是冻结全部，只保留最后3层（？？？）； weights_path：预训练模型位置
def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=False,
            weights_path='model_data/yolo_weights.h5'):
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            # Do not freeze 3 output layers.
            num = len(model_body.layers)-7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model
    
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()    