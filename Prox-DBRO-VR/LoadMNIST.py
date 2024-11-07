import numpy as np
import struct
# -*- coding=utf-8 -*-

def readfile(path_image, path_label):   # 定义读取二进制文件的函数
    with open(path_image, 'rb') as f1: # 读取二进制文件, 比如图片、视频等等
        buf1 = f1.read()
    with open(path_label, 'rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


def get_image(buf1, number):  # 读取图像数据
    """
    Read the images from the binary data

    :param buf1: the binary data of images
    :param number: the number of images
    """
    image_index = 0
    image_index += struct.calcsize('>IIII')  # .calcsize()函数计算给定的格式(fmt)占用多少字节的内存， IIII = 16字节整型
    im = []
    for i in range(number):
        temp = struct.unpack_from('>784B', buf1, image_index)
        # 按照给定的格式(fmt)解析以offset开始的缓冲区,并返回解析结果
        temp = np.array(temp)
        im.append(temp / 255)
        image_index += struct.calcsize('>784B')  # 784 整型
    return im


def get_label(buf2, number):  # 读取标签
    """
    Read the labels from the binary data

    :param buf2: the binary data of labels
    :param number: the number of labels
    """
    label_index = 0
    label_index += struct.calcsize('>II')
    label = struct.unpack_from('>%dB' % number, buf2, label_index)
    return label


def get_number(buf1):
    num = struct.unpack_from('>4B',buf1, struct.calcsize('>I'))
    number = num[2] * 256 + num[3]
    return number


def getData(imagePath, labelPath):
    """
    Get the images and labels from mnist dataset

    :param imagePath: 图像样本的路径
    :param labelPath: 标签对应的路径
    """
    image_data, label_data = readfile(imagePath, labelPath)
    number = get_number(label_data)
    image = np.array( get_image(image_data, number) )
    label = np.array( get_label(label_data, number) )

    return image, label


def data_redistribute(image, label):  # 按照标签顺序生成  “non-i.i.d." 数据
    """
    Rearrange the samples in the order of label

    :param image: image, shape(10, 784)
    :param label: label, scalar
    """
    number_sample = len(label)  # 返回对象中项目的数量
    im = [[] for _ in range(10)]  # 构建10个list类型空变量, 用于存放图像数据
    la = [[] for _ in range(10)]  # 构建10个list类型空变量, 用于存放标签 0~9
    for i in range(number_sample):
        im[label[i]].append(image[i])  # 样本i放入标签i对应的List里面
        la[label[i]].append(label[i])  # 标签i对应的样本标签i
    data_image = []
    data_label = []
    for i in range(10):
        for j in range(len(la[i])):  # len(la[i])表示第个标签对应的样本个数
            data_image.append(im[i][j])  # 标签为i的第j个样本
            data_label.append(la[i][j])  # 标签为i的第j个标签

    return data_image, data_label


