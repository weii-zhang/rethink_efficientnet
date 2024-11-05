from torchvision import datasets,transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import os
import time
from collections import Counter
import argparse

device="cuda" if torch.cuda.is_available() else "cpu"

class Efficientnet_train():
    def __init__(self,opt):  # 创建实例的时候会使用
        self.epochs=opt.epochs #训练周期
        self.batch_size=opt.batch_size #batch_size
        self.class_num=opt.class_num #类别数
        self.imgsz=opt.imgsz #图片尺寸
        self.img_dir=opt.img_dir #图片路径
        self.weights=opt.weights #模型路径
        self.save_dir=opt.save_dir #保存模型路径
        self.lr=opt.lr #初始化学习率
        self.moment=opt.m #动量
        base_model = EfficientNet.from_name('efficientnet-b5') #记载模型，使用b几的就改为b几
        state_dict = torch.load(self.weights)
        base_model.load_state_dict(state_dict)
        # 修改全连接层
        num_ftrs = base_model._fc.in_features
        base_model._fc = nn.Linear(num_ftrs, self.class_num)
        self.model = base_model.to(device)
        # 交叉熵损失函数
        self.cross = nn.CrossEntropyLoss()
        # 优化器
        self.optimzer = optim.SGD((self.model.parameters()), lr=self.lr, momentum=self.moment, weight_decay=0.0004)

        #获取处理后的数据集和类别映射表
        self.trainx,self.valx,self.b=self.process()
        print('\n初始化结束: ')
        print(self.b)

    def __call__(self):  # 对象后面加()时候自动调用
        best_acc = 0
        self.model.train(True)
        for ech in range(self.epochs):
            optimzer1 = self.lrfn(ech, self.optimzer)

            print("\n--------------------Epoch %d / %d--------------------" % (ech + 1, self.epochs))
            # 开始训练
            run_loss = 0.0  # 损失
            run_correct = 0.0  # 准确率
            count = 0.0  # 分类正确的个数

            for i, data in enumerate(self.trainx):

                inputs, label = data
                inputs, label = inputs.to(device), label.to(device)

                # 训练
                optimzer1.zero_grad()
                output = self.model(inputs)  # 形状是(channels, cls)

                # print(output.shape)
                # print(label.shape)

                loss = self.cross(output, label)
                loss.backward()
                optimzer1.step()

                run_loss += loss.item()  # 损失累加
                _, pred = torch.max(output.data, 1)
                count += label.size(0)  # 求总共的训练个数
                run_correct += pred.eq(label.data).cpu().sum()  # 截止当前预测正确的个数
                #每隔100个batch打印一次信息，这里打印的ACC是当前预测正确的个数/当前训练过的的个数
                if (i+1)%300==0:
                    print('iter:{}/{}, Acc:{}'.format(i+1,len(self.trainx), run_correct/count))

            train_acc = run_correct / count
            # 每次训完一批打印一次信息
            print('Loss:{}, Acc:{}'.format(run_loss / len(self.trainx), train_acc))

            # 训完一批次后进行验证
            with torch.no_grad():               
                # pred-label的统计
                num_0_0 = 0  # 把0判正确
                num_1_1 = 0  # 把1判正确

                correct = 0.  # 预测正确的个数
                total = 0.  # 总个数
                for i_val, data_val in enumerate(self.valx):
                    inputs, label = data_val
                    inputs, label = inputs.to(device), label.to(device)
                    output = self.model(inputs)  # 形状是(channels, cls)

                    _, pred = torch.max(output.data, 1)
                    total += label.size(0)  # 求总共的训练个数
                    correct += pred.eq(label.data).cpu().sum()  # 截止当前预测正确的个数

                    num_0_0 = num_0_0 + ((pred==0) & (label==0)).sum().item()
                    num_1_1 = num_1_1 + ((pred==1) & (label==1)).sum().item()

                print('ng正确个数:',num_0_0,' ok正确个数:',num_1_1)

                test_acc = correct.item() / total
                print('Acc:{}, correct:{}, total:{}'.format(test_acc, correct, total))

            # if best_acc < test_acc:
            #     best_acc = test_acc
            #     start_time=(time.strftime("%m%d",time.localtime()))
            #     save_weight=self.save_dir+os.sep+start_time #保存路径
            #     os.makedirs(save_weight,exist_ok=True)
            #     torch.save(self.model, save_weight + os.sep + "best_1101_train_val.pth")
            
            os.makedirs(self.save_dir,exist_ok=True)
            torch.save(self.model, self.save_dir + os.sep + "epo{}.pth".format(ech+1))


    #数据处理
    def process(self):
        # 数据增强
        data_transforms = {
            'train': transforms.Compose([
                # transforms.Grayscale(num_output_channels=3), # 转换channel为3
                transforms.Resize((self.imgsz, self.imgsz)),  # resize
                transforms.CenterCrop((self.imgsz, self.imgsz)),  # 中心裁剪
                transforms.RandomRotation(10),  # 随机旋转，旋转范围为【-10,10】
                transforms.RandomHorizontalFlip(p=0.2),  # 水平镜像
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
            ]),
            "val": transforms.Compose([
                # transforms.Grayscale(num_output_channels=3), # 转换channel为3
                transforms.Resize((self.imgsz, self.imgsz)),  # resize
                transforms.CenterCrop((self.imgsz, self.imgsz)),  # 中心裁剪
                transforms.ToTensor(),  # 张量转换
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        # 定义图像生成器
        # image_datasets = {'train': <ImageFolder对象, 包含训练集图像和标签>,'val': <ImageFolder对象, 包含验证集图像和标签>}
        # train对应的值由两种图片组成。val对应的值由两种图片组成
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.img_dir, x), data_transforms[x]) for x in
                          ['train', 'val']}
        
        # print("\n训练样本的个数: ", len(image_datasets['train']))
        # print("验证样本的个数: ", len(image_datasets['val']))

        # 统计每个类别的样本数量
        for phase in ['train', 'val']:
            class_counts = Counter([image_datasets[phase].targets[i] for i in range(len(image_datasets[phase]))])
            print(f"\n{phase.capitalize()}集样本数量:", len(image_datasets[phase]))
            for class_name, class_idx in image_datasets[phase].class_to_idx.items():
                print(f"类别 '{class_name}': {class_counts[class_idx]} 个样本")


        
        # 得到训练集和验证集
        trainx = DataLoader(image_datasets["train"], batch_size=self.batch_size, shuffle=True, drop_last=False)
        # print('trainx的类型：')
        # for i, data in enumerate(trainx):  # i是索引，由enumerate提供
        #      inputs, label = data  
        #      print(inputs.shape)  # input是(batch_size, channels, height, width)
        #      print(label.shape)  # label是(batch_size)
        #      break


        # valx = DataLoader(image_datasets["val"], batch_size=len(image_datasets['val']), shuffle=True, drop_last=True)
        valx = DataLoader(image_datasets["val"], batch_size=self.batch_size, shuffle=True, drop_last=False)

        b = image_datasets["train"].class_to_idx  # id和类别对应

        return trainx,valx,b


    # 学习率慢热加下降
    def lrfn(self,num_epoch, optimzer):
        lr_start = 0.00001  # 初始值
        max_lr = 0.0004  # 最大值
        lr_up_epoch = 3  # 之前是10 # 学习率上升10个epoch
        lr_sustain_epoch = 5  # 学习率保持不变
        lr_exp = .8  # 衰减因子
        if num_epoch < lr_up_epoch:  # 0-10个epoch学习率线性增加
            lr = (max_lr - lr_start) / lr_up_epoch * num_epoch + lr_start
        elif num_epoch < lr_up_epoch + lr_sustain_epoch:  # 学习率保持不变
            lr = max_lr
        else:  # 指数下降
            lr = (max_lr - lr_start) * lr_exp ** (num_epoch - lr_up_epoch - lr_sustain_epoch) + lr_start
        for param_group in optimzer.param_groups:
            param_group['lr'] = lr
        return optimzer
#参数设置
def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights",type=str,default="/mnt/nas-new/home/zhanggefan/zw/efficientnet/models/efficientnet-b5-b6417697.pth",help='initial weights path')#预训练模型路径
    parser.add_argument("--img-dir",type=str,default="/mnt/nas-new/home/zhanggefan/dataset/prediction/train_val_test_1104_pre",help="train image path") #数据集的路径
    parser.add_argument("--imgsz",type=int,default=512,help="image size") #图像尺寸
    parser.add_argument("--epochs",type=int,default=12,help="train epochs")#训练批次
    parser.add_argument("--batch-size",type=int,default=4,help="train batch-size") #batch-size
    parser.add_argument("--class_num",type=int,default=2,help="class num") #类别数
    parser.add_argument("--lr",type=float,default=0.0001,help="Init lr") #学习率初始值
    parser.add_argument("--m",type=float,default=0.9,help="optimer momentum") #动量
    parser.add_argument("--save-dir",type=str,default="/mnt/nas-new/home/zhanggefan/zw/efficientnet/weights/1104_pre",help="save models dir")#保存模型路径
    opt=parser.parse_known_args()[0]  # 用于解析已知的命令行参数，同时忽略未知参数
    return opt

if __name__ == '__main__':
    opt=parse_opt()
    models=Efficientnet_train(opt)
    models()
