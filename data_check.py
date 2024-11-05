# 这个文件是为了检查数据的类型
# 得到的结果是按照（图片大小 图片通道 图片后缀）的数量统计以及总数统计


import os
from PIL import Image
from collections import defaultdict

# 指定要检查的文件夹路径
root_path = '/mnt/nas-new/home/zhanggefan/dataset/prediction/train_val_test_1104_now/train'

# 用于存储类别统计
total_cls_num = 0
for clsname in os.listdir(root_path):  
    print('\n'+clsname)
    cls_path = os.path.join(root_path, clsname)
    
    # print('\n路径: ',cls_path)
    category_counts = defaultdict(int)  
    # 遍历文件夹中的所有文件
    for imgname in os.listdir(cls_path):  
        # 获取文件的完整路径
        img_path = os.path.join(cls_path, imgname)

        # 确保是文件且是图片格式
        if os.path.isfile(img_path) and imgname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')):
            with Image.open(img_path) as img:
                # 获取图片的尺寸和通道
                width, height = img.size
                mode = img.mode  # 通道模式
                # 确定通道数
                if mode == 'RGB':
                    channels = 3
                elif mode == 'RGBA':
                    channels = 4
                elif mode == 'L':
                    channels = 1
                else:
                    channels = len(mode)  # 其他模式的通道数

                # 创建类别标识
                category = (f"{width}*{height}", channels, os.path.splitext(img_path)[-1].lower())
                # 统计类别
                category_counts[category] += 1

    # 输出统计结果
    nums = 0
    for category, count in category_counts.items():
        nums = nums+count
        print(f"类别: {category}, 数量: {count}")
        
    print("该文件夹类型是: ", clsname, "该文件夹下面的数据总量是: ", nums)
    total_cls_num = total_cls_num+nums
    
print("\ntotal_cls_num: ", total_cls_num)