#为efficientnet训练分类的数据进行预处理（训练集切分+补边）
import os
import glob
import cv2
import random
from pathlib import Path


#补边,这一步主要是为了将图片填充为正方形，防止直接resize导致图片变形
def expend_img(img):
    '''
    :param img: 图片数据
    :return:
    '''
    fill_pix=[122,122,122] #填充色素，可自己设定
    h,w=img.shape[:2]
    if h>=w: #左右填充
        padd_width=int(h-w)//2
        padd_top,padd_bottom,padd_left,padd_right=0,0,padd_width,padd_width #各个方向的填充像素
    elif h<w: #上下填充
        padd_high=int(w-h)//2
        padd_top,padd_bottom,padd_left,padd_right=padd_high,padd_high,0,0 #各个方向的填充像素
    new_img = cv2.copyMakeBorder(img,padd_top,padd_bottom,padd_left,padd_right,cv2.BORDER_CONSTANT, value=fill_pix)
    return new_img


#切分训练集、验证集、测试集，并进行补边处理
def split_train_val_test(img_dir,save_dir,train_rate,val_rate):
    '''
    :param img_dir: 原始图片路径，注意是所有类别所在文件夹的上一级目录
    :param save_dir: 保存图片路径
    :param train_val_num: 切分比例
    :return:
    '''
    img_dir_list=glob.glob(img_dir+os.sep+"*")#os.sep是Python中os模块提供的一个属性，表示当前操作系统的路径分隔符。获取img_dir下面每一个文件的路径（此时一个类别对应一个路径）
    for class_dir in img_dir_list:
        class_name=class_dir.split(os.sep)[-1] #根据路径分隔符分割路径，获取最后一级，即当前类别
        img_list=glob.glob(class_dir+os.sep+"*") #获取每个类别文件夹下的所有图片
        all_num=len(img_list) #获取总个数

        train_list=random.sample(img_list,int(all_num*train_rate)) #保存的是训练集图片的！！！路径！！！
        val_test_list = list(set(img_list) - set(train_list))
        val_list=random.sample(val_test_list,int(all_num*val_rate)) #保存的是验证集图片的！！！路径！！！
        test_list = list(set(val_test_list) - set(val_list))

        save_train=save_dir+os.sep+"train"+os.sep+class_name
        save_val=save_dir+os.sep+"val"+os.sep+class_name
        save_test=save_dir+os.sep+"test"+os.sep+class_name

        os.makedirs(save_train,exist_ok=True)
        os.makedirs(save_val,exist_ok=True) 
        os.makedirs(save_test,exist_ok=True) #建立对应的文件夹

        print('\n')
        print(class_name+" all num",len(img_list))
        print(class_name+" trian num",len(train_list))
        print(class_name+" val num",len(val_list))
        print(class_name+" test num",all_num - len(train_list) - len(val_list))

        #保存切分好的数据集
        for imgpath in img_list:
            imgname=Path(imgpath).name #获取文件名
            if imgpath in train_list:
                img=cv2.imread(imgpath)
                new_img=expend_img(img)
                cv2.imwrite(save_train+os.sep+imgname,new_img)
            elif imgpath in val_list: #将除了训练集意外的数据均视为验证集
                img = cv2.imread(imgpath)
                new_img = expend_img(img)
                cv2.imwrite(save_val + os.sep + imgname, new_img)
            else:
                img = cv2.imread(imgpath)
                new_img = expend_img(img)
                cv2.imwrite(save_test + os.sep + imgname, new_img)


    print("\nsplit train and val and test finished !")


# 测试代码
if __name__ == "__main__":
    split_train_val_test('/mnt/nas-new/home/zhanggefan/dataset/prediction/train_val_1101', '/mnt/nas-new/home/zhanggefan/dataset/prediction/aaaa', 0.8, 0.1)

 