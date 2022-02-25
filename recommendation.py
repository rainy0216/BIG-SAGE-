# v4 使用基于时间戳的采样
import time
import stellargraph as sg
from stellargraph.layer import link_regression, HinSAGE
from tensorflow.keras import optimizers, losses, metrics, Model, callbacks
from sklearn import model_selection
from data_helper import TimeAwaredDataset
from metrics import recall, root_mean_square_error, recall_at_k
from time_awared_sage import TimeAwaredLinkGenerator
import os
import tensorflow as tf
import tensorflow.keras.backend as ktf

# GPU设置
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# print(gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# 参数设置
batch_size = 1024
epochs = 10
# 使用70%的边用于训练, 30%用于测试:
train_size = 0.8
test_size = 0.2
# train thread num
num_workers = 10
# subgraph rate
# sub_rate = 0.01
# learning rate
lr = 0.01

# 加载数据集
dataset = "amazon-musics"
G, G2, G3, edges_with_ratings = TimeAwaredDataset.amazon_c(dataset)

# G, G2, G3, edges_with_ratings = TimeAwaredDataset.amazon_c(dataset,mini=True,num_edges=1000000)
print('Data prepared.')
# 取子图
# edges_with_ratings = edges_with_ratings.head(int(edges_with_ratings.shape[0] * sub_rate))
# nodes = list(set(edges_with_ratings['source'].tolist())) + list(set(edges_with_ratings['target'].tolist()))
# G = G.subgraph(nodes)

# 将图的边集划分为训练集和测试集
edges_train, edges_test = model_selection.train_test_split(
    edges_with_ratings, train_size=train_size, test_size=test_size
)
edgelist_train = list(edges_train[["source", "target"]].itertuples(index=False))
edgelist_test = list(edges_test[["source", "target"]].itertuples(index=False))

# 将对应的标签(评分)划分为训练集标签和测试集标签
labels_train = edges_train["weight"]
labels_test = edges_test["weight"]

# *采样数(一阶邻居节点采样数为8，2阶邻居节点采样数为4)
num_samples = [8, 4]
# 初始化邻居节点序列生成器
generator = TimeAwaredLinkGenerator(
    G3,
    batch_size,
    num_samples,
    head_node_types=["source", "target"]
)
# generator = HinSAGELinkGenerator(
#     G,
#     batch_size,
#     num_samples,
#     head_node_types=["user", "movie"]
# )
# 分别定义训练集和测试集的生成序列数据流
train_gen = generator.flow(edgelist_train, labels_train, shuffle=False)
test_gen = generator.flow(edgelist_test, labels_test)

# HinSage 每一层的神经元数目(第一层32，第二层32)
hinsage_layer_sizes = [80, 80]
# 检测HinSage的层数和采样的层数是否一致
assert len(hinsage_layer_sizes) == len(num_samples)

# 初始化HinSage层
hinsage = HinSAGE(
    layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0
)

# attention
# hidden = tf.concat(x_out, axis=1)
# out_shape = hidden.shape[-1]
# WK = tf.Variable(tf.ones([out_shape, 1]))
# WQ = tf.Variable(tf.ones([out_shape, 1]))
# WV = tf.Variable(tf.ones([out_shape, 1]))
# K = tf.matmul(hidden, WK)
# Q = tf.matmul(hidden, WQ)
# V = tf.matmul(hidden, WV)
# score_prediction = tf.matmul(tf.nn.softmax(tf.matmul(Q , tf.transpose(K))), V)

# 获取HinSage中输入输出层的张量，可以看到输出层有两个部分
x_inp, x_out = hinsage.in_out_tensors()
print(x_inp)
print(x_out)

# 初始化评分预测函数，这里是对输出层进行拼接后输入到回归层，输出预测评分
score_prediction = link_regression(edge_embedding_method="concat")(x_out)

# 创建模型
metrics_list = [metrics.mean_squared_error,
                metrics.mae]
model = Model(inputs=x_inp, outputs=score_prediction)
model.compile(
    optimizer=optimizers.Adam(lr=lr),
    loss=losses.mean_squared_error,
    metrics=metrics_list,
)

# 输出模型信息
model.summary()

# 训练前在测试集上评估模型
test_metrics = model.evaluate(
    test_gen, verbose=1, use_multiprocessing=False, workers=num_workers
)
print("Untrained model's Test Evaluation:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# 模型训练
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.7, patience=2, verbose=1, mode='min'),
    callbacks.ModelCheckpoint(filepath='models/' + dataset + '.model', monitor='val_mean_squared_error',
                              save_weights_only=False, mode='min', save_best_only=True),
    callbacks.CSVLogger(filename='logs/' + dataset + '.csv', append=True),
    callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.5, patience=2, mode='min', min_delta=0.01,
                                min_lr=0),
    callbacks.TensorBoard()]
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=False,
    workers=num_workers,
    callbacks=callbacks_list,
)

# 绘制历史评估并保存文本和图片
sg.utils.plot_history(history, individual_figsize=(8, 5), return_figure=True).savefig(
    'output/' + dataset + '-metrics.png')
with open('output/' + dataset + 'metrics.txt', 'w') as json_file:
    json_file.write(str(history.history))

# 训练后评估模型
test_metrics = model.evaluate(
    test_gen, use_multiprocessing=False, workers=num_workers, verbose=1
)
print("Test Evaluation:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# logs_loss.end_draw()

# h_true = plt.hist(y_true, bins=30, facecolor="green", alpha=0.5)
# h_pred = plt.hist(y_pred, bins=30, facecolor="blue", alpha=0.5)
# plt.xlabel("ranking")
# plt.ylabel("count")
# plt.legend(("True", "Predicted"))

# loss: 1.0950 - root_mean_square_error: 1.0458 - mean_absolute_error: 0.7899
