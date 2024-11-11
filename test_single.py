# 只推理单一个模型权重

import torch
import os
import torchvision
import glob
from PIL import Image
import cv2
import argparse
import matplotlib.pyplot as plt
import shutil

device="cuda" if torch.cuda.is_available() else "cpu"

#测试图片
class Test_model():
    def __init__(self,opt):
        self.imgsz=opt.imgsz #测试图片尺寸
        self.img_dir=opt.test_dir #测试图片路径
        self.model=(torch.load(opt.weights_dir)).to(device) #加载模型
        self.model.eval()
        self.class_name={'自主学习严重图库': 0, '自主学习轻微图库': 1, '自主学习OK图库': 1, 'ng': 0, 'ok': 1}  #类别信息
        
        
    def __call__(self):
        #图像转换
        data_transorform=torchvision.transforms.Compose([
                # torchvision.transforms.Grayscale(num_output_channels=3), # 转换channel为3
                # torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
                torchvision.transforms.Resize((self.imgsz,self.imgsz)),
                torchvision.transforms.CenterCrop((self.imgsz,self.imgsz)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        
        for filename in os.listdir(self.img_dir):  
            # if filename=='自主学习OK图库':
            #     continue
            cls_dir = os.path.join(self.img_dir, filename)

            # img_list=glob.glob(cls_dir+os.sep+"*.[pj][np]g")
            img_list = glob.glob(cls_dir + os.sep + "*.[jp][pn]g") + glob.glob(cls_dir + os.sep + "*.bmp")

            result_0 = []
            result_1 = []
            result_diff = []

            correct_num = 0
            for imgpath in img_list:
                img=cv2.imread(imgpath)
                new_img=self.expend_img(img) #补边
                img=Image.fromarray(new_img)
                img=data_transorform(img) #转换
                img=torch.reshape(img,(-1,3,self.imgsz,self.imgsz)).to(device) #维度转换[B,C,H,W]
                              
                pred=self.model(img)
                result_0.append(pred[0][0].item())
                result_1.append(pred[0][1].item())
                result_diff.append( pred[0][0].item() - pred[0][1].item() )

                _,pred=torch.max(pred,1)

                if self.class_name[filename] == pred.item():
                    correct_num = correct_num+1
                # else:
                #     if filename=='自主学习严重图库':
                #         # 如果预测错误，保存图像副本
                #         save_path = os.path.join("/mnt/nas-new/home/zhanggefan/zw/efficientnet/严重图库中判别错误", os.path.basename(imgpath))
                #         shutil.copy(imgpath, save_path)  # 复制图像到目标路径
                #         # print(f"保存预测错误的图片：{save_path}")



            # # 设置直方图的颜色和标签
            # plt.hist(result_0, bins=50, color='blue', alpha=0.6, label='ng')  # 使用蓝色绘制 0 类
            # plt.hist(result_1, bins=50, color='red', alpha=0.6, label='ok')   # 使用红色绘制 1 类
            # # 添加图例和标题
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # plt.title("result of ng/ok")
            # plt.legend()
            # # 指定保存路径和文件名
            # output_path = "/mnt/nas-new/home/zhanggefan/zw/efficientnet/plot/"+filename+".png"
            # plt.savefig(output_path)  # 保存图像到指定位置
            # plt.close()
                    

            # # 设置直方图的颜色和标签
            # plt.hist(result_diff, bins=50, color='blue', alpha=0.6, label='0-1')  
            # # 添加图例和标题
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # plt.title("result of ng-ok")
            # plt.legend()
            # # 指定保存路径和文件名
            # output_path = "/mnt/nas-new/home/zhanggefan/zw/efficientnet/plot/"+filename+"_0-1.png"
            # plt.savefig(output_path)  # 保存图像到指定位置
            # plt.close()

            print( "class:", filename, "all_nums:",len(img_list),"correct_nums:",correct_num,"acc:",correct_num/len(img_list) )

    #补边为正方形
    def expend_img(self,img,fill_pix=122):
        '''
        :param img: 图片数据
        :param fill_pix: 填充像素，默认为灰色，自行更改
        :return:
        '''
        
        h,w=img.shape[:2] #获取图像的宽高
        if h>=w: #左右填充
            padd_width=int(h-w)//2
            padd_h,padd_b,padd_l,padd_r=0,0,padd_width,padd_width #获取上下左右四个方向需要填充的像素

        elif h<w: #上下填充
            padd_high=int(w-h)//2
            padd_h,padd_b,padd_l,padd_r=padd_high,padd_high,0,0

        new_img = cv2.copyMakeBorder(img, padd_h, padd_b, padd_l, padd_r, borderType=cv2.BORDER_CONSTANT,
                                     value=[fill_pix,fill_pix,fill_pix])
        return new_img
    
# #参数设置
# def parser_opt():
#     parser=argparse.ArgumentParser()   
#     parser.add_argument("--test-dir",type=str,default="/mnt/nas-new/home/zhanggefan/dataset/prediction/train_val_test_1103/test")
#     parser.add_argument("--weights",type=str,default="/mnt/nas-new/home/zhanggefan/zw/efficientnet/weights/1103/best_1103_train_val_epo10.pth",help="model path")
#     parser.add_argument("--imgsz",type=int,default=512,help="test image size")
#     opt=parser.parse_known_args()[0]
#     return opt


# if __name__ == '__main__':
#     opt=parser_opt()
#     test_img=Test_model(opt)
#     test_img()


# 参数设置
def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="/mnt/nas-new/home/zhanggefan/dataset/prediction/train_val_test_1109/test_full", help="Directory containing test images")
    parser.add_argument("--weights-dir", type=str, default="/mnt/nas-new/home/zhanggefan/zw/efficientnet/weights/1102/best_before.pth", help="Directory containing model weights")
    parser.add_argument("--imgsz", type=int, default=512, help="Test image size")
    opt = parser.parse_known_args()[0]
    return opt

if __name__ == '__main__':
    opt = parser_opt()

    # 初始化测试模型并加载权重
    test_img = Test_model(opt)
    test_img()  # 运行测试