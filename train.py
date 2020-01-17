# -------------------------------------------------------------------------------------------------
# 导入一些第三方包
# -------------------------------------------------------------------------------------------------

import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os


x = []
y_1 = []
y_2 = []


class DataIterator:
    def __init__(self, data_dir):
        truncate_path = data_dir + ('%03d' % 9)
        print(truncate_path)
        self.image_names = []

        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names = self.image_names+[os.path.join(root, file_path) for file_path in file_list]
                print(self.image_names)
        random.shuffle(self.image_names)

        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]
        print(self.labels)

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        # 通过[-max_delta, max_delta)的范围随机调整图像的亮度.
        images = tf.image.random_brightness(images, max_delta=0.3)
        #通过随机因子调整图像的对比度，相当于adjust_contrast(),但使用了在区间[lower, upper]中随机选取的contrast_factor
        images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        # 要将python的数据类型转换成TensorFlow可用的tensor数据类型
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        # 一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
        # 设置 num_epochs=None,生成器可以无限次遍历tensor列表
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)
        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        #图片解码
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=3), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([128, 128], dtype=tf.int32)
        #裁剪图片
        images = tf.image.resize_images(images, new_size)
        #从队列中读取数据
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=500,min_after_dequeue=100)
        return image_batch, label_batch


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

    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 512, activation_fn=tf.nn.relu, scope='fc1')
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob), 9, activation_fn=None, scope='fc2')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    #通过变量维持图graph的状态
    #trainable：此集合用于优化器Optimizer类优化的的默认变量列表【可为optimizer指定其他的变量集合】，可就是要训练的变量列表
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    #实现指数衰减学习率
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(logits)
    tf.summary.scalar('loss', loss)#绘制loss曲线
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    #主要是用于计算预测的结果和实际结果的是否相等
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


def train():
    train_feeder = DataIterator(data_dir='./data/train/')
    test_feeder = DataIterator(data_dir='./data/test/')

    with tf.Session() as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=100, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=100)

        graph = build_graph(top_k=1)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        # -------------------------------------------------------------------------------------------------

        step = []
        train_acc = []
        test_acc = []

        for i in range(100):
            train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
            test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
            feed_dict = {graph['images']: train_images_batch, graph['labels']: train_labels_batch, graph['keep_prob']: 0.8}
            _, accuracy_train = sess.run([graph['train_op'], graph['accuracy']], feed_dict=feed_dict)

            if i % 10 == 0:
                feed_dict = {graph['images']: test_images_batch, graph['labels']: test_labels_batch, graph['keep_prob']: 0.8}
                accuracy_test = sess.run(graph['accuracy'], feed_dict=feed_dict)
                print("---the step {%d} ---train accuracy {%.3f} ---test accuracy {%.3f}" % (i, accuracy_train, accuracy_test))
                step.append(i)
                train_acc.append(accuracy_train)
                test_acc.append(accuracy_test)

            if i % 100 == 0 and i > 0:
                saver.save(sess, './model/model.ckpt')

        plt.plot(step, train_acc)
        plt.plot(step, test_acc)
        plt.show()


if __name__ == "__main__":
    train()





