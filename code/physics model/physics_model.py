'''
@author yunyuan
'''
import numpy as np
import tensorflow as tf
import sys, time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
conf.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=conf)
tf.keras.backend.set_session(sess)

# physics model
kfm = "LWR"
# kfm = "PW"
# kfm = "ARZ"

# select runtime env
# local computer
directory = "./"

# colab env
# put data files in this dir
# directory = "./drive/My Drive/testPIGP/"

# add noise to data
# NOISE = False
# noise_level = 0.5
# noise_std_dev = 50
# noise_bias = 100

# sample_size = 288 * 18
# test_size = 288 * 4

# for replication
np.random.seed(1234)
tf.set_random_seed(1234)

# for display precision
np.set_printoptions(precision=3, suppress=True)

# for none 0 dividendE
jitter = 1e-6

# data
xx = tf.keras.layers.Input(shape=(1,), dtype=tf.float64, name="x")
tt = tf.keras.layers.Input(shape=(1,), dtype=tf.float64, name="t")

xxtt = tf.keras.layers.Concatenate(axis=1)([xx, tt])

# output
layer_width = 800
layer_depth = 1
act = "relu"

qq = tf.keras.layers.Dense(layer_width, activation=act)(xxtt)
for _ in range(layer_depth):
    qq = tf.keras.layers.Dense(layer_width, activation=act)(qq)
qq = tf.keras.layers.Dense(1, activation="linear", name="q")(qq)

vv = tf.keras.layers.Dense(layer_width, activation=act)(xxtt)
for _ in range(layer_depth):
    vv = tf.keras.layers.Dense(layer_width, activation=act)(vv)
vv = tf.keras.layers.Dense(1, activation="linear", name="v")(vv)

model = tf.keras.Model(inputs=[xx, tt], outputs=[qq, vv])

# parameter
# LWR
if kfm == "LWR" or kfm == "PW" or kfm == "ARZ":
    tf_log_rho_c = tf.keras.backend.variable(np.log(12.18), dtype=tf.float64)
    tf_log_rho_max = tf.keras.backend.variable(np.log(22.19), dtype=tf.float64)
    tf_log_q_max = tf.keras.backend.variable(np.log(812.40), dtype=tf.float64)
    model.layers[-1].trainable_weights.append(tf_log_rho_c)
    model.layers[-1].trainable_weights.append(tf_log_rho_max)
    model.layers[-1].trainable_weights.append(tf_log_q_max)

# PW
if kfm == "PW":
    tf_t_0 = tf.keras.backend.variable(-4.23, dtype=tf.float64)
    tf_c_0 = tf.keras.backend.variable(-1.61, dtype=tf.float64)
    model.layers[-1].trainable_weights.append(tf_t_0)
    model.layers[-1].trainable_weights.append(tf_c_0)

# ARZ
if kfm == "ARZ":
    tf_t_1 = tf.keras.backend.variable(-3.34, dtype=tf.float64)
    model.layers[-1].trainable_weights.append(tf_t_1)

# pde
def conservative_law_constraint(x, t, q, v, rho):
    """
    partial_t rho + partial_x q = 0
    """
    g = 0
    g += tf.keras.backend.gradients(rho, t)[0]
    g += tf.keras.backend.gradients(q, x)[0]
    return g


# fd
def fd_V(rho, qm, rm, rc):
    """
    if rho < rc:
        return u
    else:
        return q/rho
    
    u=qm/rc
    (0, u)-(rho_c,u)-(rho_max,0)
    """
    u = tf.fill([tf.shape(xxtt)[0], 1], tf.divide(qm, rc + jitter))
    return tf.where(
        tf.math.less_equal(rho, rc), u, tf.divide(fd_Q(rho, qm, rm, rc), rho + jitter)
    )


def fd_Q(rho, qm, rm, rc):
    """
    if rho < rc:
        return u * rho
    else:
        return qm/(rm-rc)/(rm-rho)
        
    (0,0)-(rho_c,q_max)-(rho_max,0)
    a(rm-x)=0
    a(rm-rc)=qm
    a=qm/(rm-rc)
    """
    u = tf.divide(qm, rc + jitter)
    return tf.where(
        tf.math.less_equal(rho, rc),
        u * rho,
        tf.divide(qm, (rm - rc) * (rm - rho) + jitter),
    )


def fd_constraint1(v, rho, qm, rm, rc):
    """
    v - V(rho) = 0
    """
    g = 0
    g += v - fd_V(rho, qm, rm, rc)
    return g


def fd_constraint2(q, rho, qm, rm, rc):
    """
    q - Q(rho) = 0
    """
    g = 0
    g += q - fd_Q(rho, qm, rm, rc)
    return g


# 2nd order models
def PW_momentum(x, t, v, rho, qm, rm, rc, t_0, c_0):
    """
    \partial_t v+v\partial_x v+\frac{V-V(\rho)}{\tau_0}+\frac{c^2_0}{\rho}\partial_x\rho = 0
    """
    g = 0
    g += tf.keras.backend.gradients(v, t)[0]
    g += v * tf.keras.backend.gradients(v, x)[0]
    g += tf.divide((v - fd_V(rho, qm, rm, rc)), (t_0 + jitter))
    g += tf.divide((c_0 * c_0), (rho + jitter)) * tf.keras.backend.gradients(rho, x)[0]
    return g


def ARZ_momentum(x, t, v, rho, qm, rm, rc, t_0):
    """
    u'\partial_t(v-V(\rho) + v\partial_x (v-V(\rho))+\frac{v-V(\rho)}{\tau_0} = 0'
    """
    g = 0
    g += tf.keras.backend.gradients(v - fd_V(rho, qm, rm, rc), t)[0]
    g += v * tf.keras.backend.gradients(v - fd_V(rho, qm, rm, rc), x)[0]
    g += tf.divide((v - fd_V(rho, qm, rm, rc)), t_0 + jitter)
    return g


def loss1(y_true, y_pred):
    rho = tf.divide(qq, vv + jitter)
    qm = tf.exp(tf_log_q_max)
    rm = tf.exp(tf_log_rho_max)
    rc = tf.exp(tf_log_rho_c)
    gl = 0
    if kfm == "PW" or kfm == "ARZ" or kfm == "LWR":
        g1 = fd_constraint1(vv, rho, qm, rm, rc)
        g2 = fd_constraint2(qq, rho, qm, rm, rc)
        g3 = conservative_law_constraint(xx, tt, qq, vv, rho)
        gl += tf.abs(g1)
        gl += tf.abs(g2)
        gl += tf.abs(g3)

    if kfm == "PW":
        t_0 = tf.exp(tf_t_0)
        c_0 = tf.exp(tf_c_0)
        g4 = PW_momentum(xx, tt, vv, rho, qm, rm, rc, t_0, c_0)
        gl += tf.abs(g4)

    if kfm == "ARZ":
        t_1 = tf.exp(tf_t_1)
        g5 = ARZ_momentum(xx, tt, vv, rho, qm, rm, rc, t_1)
        gl += tf.abs(g5)
    return gl


def loss2(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)


def customized_loss(y_true, y_pred):
    # w = 1
    w = 100
    return loss1(y_true, y_pred) * w + loss2(y_true, y_pred)


# data--------------------------------------------------------------
# case 1
data = np.load(directory + "data.npz")

X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["Y_train"]
Y_test = data["Y_test"]
Y_test[:, 0] = Y_test[:, 0] / 12
Y_train[:, 0] = Y_train[:, 0] / 12

# case2
# data = np.load(directory+'data4.npz')
# X_train = data["X_train"]
# X_test = data["X_test"]
# Y_train = data["Y_train"]
# Y_test = data["Y_test"]

# idx = (data['X_train'][:, 1] < 1440)
# X_train = X_train[idx]
# Y_train = Y_train[idx]
# idx1 = (data['X_test'][:, 1] < 1440)
# X_test = X_test[idx1]
# Y_test = Y_test[idx1]


# early noise
# if NOISE:
#     l = int(X.shape[0] * noise_level)
#     # X_train[:l,:] = X_train[:l,:] + np.random.normal(0, 15, X_train[:l,:].shape)
#     Y[:l, 0] = Y[:l, 0] + np.random.normal(noise_bias, noise_std_dev, Y[:l, 0].shape)
#     Y[:l, 1] = Y[:l, 1] + np.random.normal(5, 5, Y[:l, 1].shape)

# from sklearn.utils import shuffle

# X, Y = shuffle(X, Y)
# X_train = X[:sample_size]
# X_test = X[sample_size : sample_size + test_size]
# Y_train = Y[:sample_size]
# Y_test = Y[sample_size : sample_size + test_size]


# normalization
# y_mean = np.mean(Y_train, axis=0)
# y_std = np.std(Y_train, axis=0)

# x_mean = np.mean(X_train, axis=0)
# x_std = np.std(X_train, axis=0)

# # x_std[x_std == 0] = 1
# # y_std[y_std == 0] = 1

# Y_train = (Y_train - y_mean) / y_std
# Y_test = (Y_test - y_mean) / y_std

# X_train = (X_train - x_mean) / x_std
# X_test = (X_test - x_mean) / x_std

# training----------------------------------------------------------
model.compile(
    optimizer="adam",
    #   loss=customized_loss,
    loss=loss2,
    metrics=[loss1, "mse"],
)
# csvlogger
this_time = time.strftime("%Y-%m-%dT%H%M%S")
filename = directory + "PK_" + kfm + "_" + this_time + ".csv"
csvlogger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=False)
model.load_weights(directory + "weights.h5")

# # lrscheduler
# def scheduler(epoch):
#   if epoch < 10:
#     return 0.001
#   else:
#     return 0.001 * np.exp(0.1 * (10 - epoch))

# lrscheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.fit(
    [X_train[:, 0], X_train[:, 1]],
    [Y_train[:, 0], Y_train[:, 1]],
    epochs=200,
    batch_size=20,
    verbose=2,
    validation_data=(
        {"x": X_test[:, 0], "t": X_test[:, 1]},
        {"q": Y_test[:, 0], "v": Y_test[:, 1]},
    ),
    #   callbacks = [csvlogger, lrscheduler]
    callbacks=[csvlogger],
)
Y_train_predict = model.predict([X_train[:, 0], X_train[:, 1]])
Y_test_predict = model.predict([X_test[:, 0], X_test[:, 1]])
np.savez(
    directory + "Macro_PK_" + kfm + "_predictions_" + this_time,
    X_train=X_train,
    X_test=X_test,
    Y_train=Y_train,
    Y_test=Y_test,
    Y_train_predict=Y_train_predict,
    Y_test_predict=Y_test_predict,
)
model.save_weights(directory + "weights" + this_time + ".h5")

# output parameters
qm = tf.exp(tf_log_q_max).eval(session=sess)
rm = tf.exp(tf_log_rho_max).eval(session=sess)
rc = tf.exp(tf_log_rho_c).eval(session=sess)
print("qm =", qm, "rm =", rm, "rc =", rc)
if kfm == "PW":
    t_0 = tf.exp(tf_t_0).eval(session=sess)
    c_0 = tf.exp(tf_c_0).eval(session=sess)
    print("t0 =", t_0, "c0", c_0)
if kfm == "ARZ":
    t_1 = tf.exp(tf_t_1).eval(session=sess)
    print("t1 =", t_1)
