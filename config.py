import os

def dir_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

batch_size = 4
need_modify = '416_416_2'
img_size = (416,416)
model_save_path = r'./model_save/' + need_modify
output_path = r'./result/' + need_modify
model_path = os.path.join(model_save_path,'trained_weights.h5')



