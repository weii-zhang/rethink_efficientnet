import matplotlib.pyplot as plt

# 示例数据：两个一维数组，分别代表 0 类和 1 类
class_0 = [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7, 8]  # 0 类数据
class_1 = [5, 6, 6, 7, 8, 8, 9, 10, 10, 10, 11, 12]  # 1 类数据

# 设置直方图的颜色和标签
plt.hist(class_0, bins=10, color='blue', alpha=0.6, label='Class 0')  # 使用蓝色绘制 0 类
plt.hist(class_1, bins=10, color='red', alpha=0.6, label='Class 1')   # 使用红色绘制 1 类

# 添加图例和标题
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Class 0 and Class 1")
plt.legend()

# 指定保存路径和文件名
output_path = "/mnt/nas-new/home/zhanggefan/zw/efficientnet/plot/histogram.png"
plt.savefig(output_path)  # 保存图像到指定位置

# 显示图形
plt.show()

print(f"Histogram saved to {output_path}")
