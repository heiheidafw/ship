# ---------------------------------------------------------------------------------------------------------
# 导入一些必要的第三方包
# ---------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from PIL import Image
import os
import cv2

__global_times = 0
boats = ['buoy', 'cruise ship', 'ferry boat', 'freight boat', 'gondola', 'inflatable boat', 'kayak', 'paper boat', 'sailboat']

# ---------------------------------------------------------------------------------------------------------
# 搭建神经网络
# tf.placeholder --- 设置一个容器，用于接下来存放数据
# keep_prob --- dropout的概率，也就是在训练的时候有多少比例的神经元之间的联系断开
# images --- 喂入神经网络的图片， labels --- 喂入神经网络图片的标签
# is_training --- 一个标志位 --- 用于判断是否训练神经网络，训练的时候设置为True，识别的时候设置为False
# with tf.device('/cpu:0'): --- 表示使用CPU进行训练
# slim.conv2d --- 卷积层 --- (images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
# 第一个参数表示输入的训练图片，第二个参数表示滤波器的个数，原来的数据是宽64高64维度1，处理后的数据是维度64，
# 第三个参数是滤波器的大小，宽3高3的矩形，第四个参数是表示输入的维度，第五个参数padding表示加边的方式，第六个参数表示层的名称
# slim.max_pool2d --- 表示池化层 --- 池化就是减小卷积神经网络提取的特征，将比较明显的特征提取了，不明显的特征就略掉
# slim.flatten(max_pool_4) --- 表示将数据压缩
# slim.fully_connected --- 全连接层 --- 也就是一个个的神经元
# slim.dropout(flatten, keep_prob) --- dropout层，在训练的时候随机的断掉神经元之间的连接，keep_prob就是断掉的比例
# tf.reduce_mean --- 得到平均值
# tf.nn.sparse_softmax_cross_entropy_with_logits --- 求得交叉熵
# tf.argmax(logits, 1) --- 得到较大的值
# tf.equal() --- 两个数据相等为True，不等为False --- 用于得到预测准确的个数
# tf.cast() --- 将True和False转为1和0，
# global_step --- 训练的步数 --- initializer=tf.constant_initializer(0.0) --- 步数初始化
# tf.train.AdamOptimizer(learning_rate=0.1) --- 优化器的选择，这个训练使用的Adam优化器
# learning_rate=0.1 --- 学习率 --- 训练的过程也就是神经网络学习的过程
# tf.nn.softmax(logits) --- 得到可能性
#  tf.summary.scalar / merged_summary_op --- 用于显示训练过程的数据
# predicted_val_top_k --- 喂入图片得到的可能性，也就是识别得到是哪一个汉字的可能性，top_k表示可能性最大的K个数据
# predicted_index_top_k --- 这个表示识别最大K个可能性汉字的索引 --- 也就是汉字对应的数字
# return 表示这个函数的返回值
# ---------------------------------------------------------------------------------------------------------

def build_graph(top_k):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
    conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
    max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(max_pool_3)

    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 512, activation_fn=tf.nn.tanh, scope='fc1')
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob), 9, activation_fn=None, scope='fc2')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(logits)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    #为了找到输入的张量的最后的一个维度的最大的k个值和它的下标
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


# ---------------------------------------------------------------------------------------------------------
# 加载训练保存的模型
# tf.Session() --- 新建一个会话 --- 也就是相当于开启一个任务
# build_graph(top_k=3) --- 搭建一个神经网络 --- top_k=3表示得到三个最大值
# tf.train.Saver() --- tf.train.latest_checkpoint(__checkpoint_dir) --- 得到最新的模型 --- 若存在返回True
# saver.restore(sess, ckpt) --- 恢复保存的模型 --- 也就是将神经网络所有的参数恢复训练时候得到的数据
# return graph, sess --- 返回恢复好参数的神经网络和会话
# ---------------------------------------------------------------------------------------------------------

def predictPrepare():
    sess = tf.Session()
    graph = build_graph(top_k=1)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('./model')
    if ckpt:
        saver.restore(sess, ckpt)
    return graph, sess


# ---------------------------------------------------------------------------------------------------------
# imagePrepare(image_path): --- 图片的预处理
# Image.open(image_path).convert('L') --- 打开图像，并且将图像转化为灰度图像，
# temp_image.resize((64, 64), Image.ANTIALIAS) --- 改变原来图片的大小， Image.ANTIALIAS --- 抗锯齿
# np.asarray(temp_image) / 255.0 --- 归一化 --- 将原来0-255的数据转化为0-1的数据
# temp_image.reshape([-1, 64, 64, 1]) --- 改变矩阵的形状
# return temp_image --- 返回预处理之后的的图像
# ---------------------------------------------------------------------------------------------------------

def imagePrepare(image_path):
    temp_image = Image.open(image_path)
    temp_image = temp_image.resize((128, 128), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 128, 128, 3])
    return temp_image


# ---------------------------------------------------------------------------------------------------------
# chineseRecognize(__test_image_file): --- 识别函数
# global __global_times --- 转化为全局变量
# if (__global_times == 0) --- 也就是当第一次识别的时候
#  __graph1, __sess1 = predictPrepare() --- 加载模型，准备好预测
# imagePrepare(__test_image_file) --- 调用上面的imagePrepare()函数 --- 对图片进行预处理
# predict_val, predict_index = __sess1.run() --- 向神经元喂入数据，得到预测的可能性以及预测汉字的索引值
# __global_times = 1 --- 将识别的次数置1
# return predict_val, predict_index --- 返回预测的可能性以及预测汉字的索引值 --- 三个数据 + 三个数据
# else: --- 下面表示不再是第一次进行数据的预测了，这样就没有了网络的加载和模型的加载过程，节省了时间
# 还必须有的过程是图片的预处理和测试数据喂入神经网络的过程
# ---------------------------------------------------------------------------------------------------------

def BoatRecognize(__test_image_file):

    global __global_times
    if __global_times == 0:
        global __graph1, __sess1
        __graph1, __sess1 = predictPrepare()
        temp_image = imagePrepare(__test_image_file)
        predict_val, predict_index = __sess1.run([__graph1['predicted_val_top_k'], __graph1['predicted_index_top_k']],
                                                 feed_dict={__graph1['images']: temp_image,
                                                            __graph1['keep_prob']: 1.0})
        __global_times = 1
        return predict_val, predict_index

    else:
        temp_image = imagePrepare(__test_image_file)
        predict_val, predict_index = __sess1.run([__graph1['predicted_val_top_k'], __graph1['predicted_index_top_k']],
                                                 feed_dict={__graph1['images']: temp_image, __graph1['keep_prob']: 1.0})

        return predict_val, predict_index


# count_1 = 0
# count_2 = 0
# l1 = os.listdir('./data/test/')
# l1.sort()
# length1 = len(l1)
#
# for i in range(length1):
#     path = './data/test/' + l1[i] + '/'
#     l2 = os.listdir(path)
#     l2.sort()
#     length2 = len(l2)
#
#     for j in range(length2):
#         image_path = path + l2[j]
#         count_2 += 1
#         a, b = BoatRecognize(image_path)
#         predict = b[0]
#         if int(l1[i]) in predict:
#             count_1 += 1
#         print(count_1, count_2)
#
        image = cv2.imread(image_path)
        image = cv2.putText(image, str(boats[int(l1[i])]), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        cv2.imshow('', image)
        cv2.waitKey(0)
#
# print(count_1, count_2, count_1 / count_2)

