"""
@author zwang
@author yunyuan
requirement:
    tf version (<1.15)
"""


import numpy as np
import tensorflow as tf
import sys, time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# config
# 1
tfm = "PIGP"
kfm = "LWR"

# 2
# tfm = "GP"
# kfm = 'pure'

# 3
# tfm = "PIGP"
# kfm = 'PW'

# 4
# tfm = 'PIGP'
# kfm = 'ARZ'

# 5
# tfm = 'PIGP'
# kfm = 'HEAT'

# select runtime env
directory = "./"

# select dataset
# DATA 4 loc timestep overall
DATA = 1

# DATA 20 loc
# DATA = 2
# DATA 4 timestep starting each day
# DATA = 3

# seperate set
DATA_MODE = 1

# slice no shuffle
# DATA_MODE = 2

# slice and shuffle
# DATA_MODE = 3

# shuffle and slice then add noise
# DATA_MODE = 4

# slice and shuffle x=[pos] for DATA==3
# DATA_MODE = 5

# testing gradients
STUDY = False

# # of iteration 500 to 10000
EPOCHS = 500

# sample size for training
# subject to GPU RAM
# 288 per day
SAMPLE_SIZE = 288 * 10

# save tf intermedian weights
TEST = True

# add noise to data
NOISE = True
noise_level = 0.5
noise_std_dev = 0
noise_bias = 1200

# for replication
np.random.seed(1234)
tf.set_random_seed(1234)

# for display precision
np.set_printoptions(precision=3, suppress=True)

# keep positive semidefinite
jitter = 1e-6

# # of eqs
num_eq = {"LWR": 3, "pure": 0, "PW": 4, "ARZ": 4, "HEAT": 1}

# ----------------------------------------
class Kernel_ARD:
    def __init__(self, jitter=0):
        # noise
        self.jitter = tf.constant(jitter, dtype=tf.float64)

    def matrix(self, X, amp, ls):
        # K(x,x)
        K = self.cross(X, X, amp, ls)
        K = K + self.jitter * tf.eye(tf.shape(K)[0], tf.shape(K)[1], dtype=tf.float64)
        return K

    def cross(self, X1, X2, amp, ls):
        # hat{K}
        norm1 = tf.reshape(tf.reduce_sum(X1 ** 2, 1), [-1, 1])
        norm2 = tf.reshape(tf.reduce_sum(X2 ** 2, 1), [1, -1])
        K = norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2
        K = amp * tf.exp(-1.0 * K / ls)
        return K


class Kernel_RBF:
    def __init__(self, jitter=0):
        self.jitter = tf.constant(jitter, dtype=tf.float64)

    def matrix(self, X, ls):
        K = self.cross(X, X, ls)
        K = K + self.jitter * tf.eye(tf.shape(K)[0], tf.shape(K)[1], dtype=tf.float64)
        return K

    def cross(self, X1, X2, ls):
        norm1 = tf.reshape(tf.reduce_sum(X1 ** 2, 1), [-1, 1])
        norm2 = tf.reshape(tf.reduce_sum(X2 ** 2, 1), [1, -1])
        K = norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2
        K = tf.exp(-1.0 * K / ls)
        return K


class PIGPR:
    # main class
    def __init__(self, cfg):
        # tf config
        self.cfg = cfg

        # tf placeholders and graph
        self.layers = cfg["layers"]

        self.input_dim = cfg["input_dim"]
        self.output_dim = cfg["output_dim"]

        self.imax = cfg["imax"].reshape((-1, self.input_dim))
        self.imin = cfg["imin"].reshape((-1, self.input_dim))

        self.gamma = cfg["gamma"]
        self.batch_sz = cfg["batch_sz"]

        # (x, t)
        self.tf_X_f = tf.placeholder(tf.float64, shape=[None, self.input_dim])

        # (q, v)
        self.tf_y = tf.placeholder(tf.float64, shape=[None, self.output_dim])

        # sampled z for g
        self.tf_X_g = [
            tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(self.input_dim)
        ]

        # Initialize
        self.kernel_ard = Kernel_ARD(jitter)
        self.kernel_rbf = Kernel_RBF(jitter)

        # learnable model parameters
        # f kernel
        self.tf_log_ls = [
            tf.Variable(0.0, dtype=tf.float64),
            tf.Variable(0.0, dtype=tf.float64),
        ]
        self.tf_log_amp = [
            tf.Variable(0.0, dtype=tf.float64),
            tf.Variable(0.0, dtype=tf.float64),
        ]

        # noise level
        self.tf_log_tau = [
            tf.Variable(0.0, dtype=tf.float64),
            tf.Variable(0.0, dtype=tf.float64),
        ]

        self.num_eq = num_eq[kfm]
        self.tf_log_ls_g = [
            tf.Variable(0.0, dtype=tf.float64) for i in range(self.num_eq)
        ]

        # -------------------------------------------
        # pde parameter
        # fd
        self.tf_log_rho_c = tf.Variable(0.0, dtype=tf.float64)
        self.tf_log_rho_max = tf.Variable(0.0, dtype=tf.float64)
        self.tf_log_q_max = tf.Variable(0.0, dtype=tf.float64)

        # pw
        self.tf_t_0 = tf.Variable(0.0, dtype=tf.float64)
        self.tf_c_0 = tf.Variable(0.0, dtype=tf.float64)

        # ARZ
        self.tf_t_1 = tf.Variable(0.0, dtype=tf.float64)

        # heat
        self.tf_u = [tf.Variable(0.0, dtype=tf.float64) for i in range(self.output_dim)]

        # --------------------------------------------

        # testing
        self.tf_X_test = tf.placeholder(tf.float64, shape=[None, self.input_dim])
        self.tf_y_test = self.pred_multi_dist(self.tf_X_test)
        self.tf_y_train_pred = self.pred_multi_dist(self.tf_X_f)

        # setup
        self.optimizer = tf.train.AdamOptimizer(cfg["lr"])
        self.optimizer1 = tf.train.AdamOptimizer(cfg["lr"])
        self.optimizer2 = tf.train.AdamOptimizer(cfg["lr"])

        # evidence lower bound
        self.nELBO = self.get_ELBO_multi()
        self.minimizer = self.optimizer.minimize(self.nELBO)
        self.minimizer1 = self.optimizer1.minimize(self.get_lpy())
        if tfm == "PIGP":
            self.minimizer2 = self.optimizer2.minimize(self.get_lpg())

        # tf config
        conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=True
        )
        conf.gpu_options.allow_growth = True

        self.sess = tf.compat.v1.Session(config=conf)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def solve(self, A, b):
        return tf.linalg.solve(A, b)

    # -----------------------------
    def get_ELBO_multi(self):
        return self.get_lpy() + self.get_lpg()

    def get_lpy(self):
        # likelihood part I log[p(y|x)]
        lpy = 0
        # for each column of Y create one likelihood term
        N = tf.cast(tf.shape(self.tf_X_f)[0], dtype=tf.float64)
        for i in range(self.output_dim):
            NN_X_f = self.tf_X_f
            Knn = self.kernel_ard.matrix(
                NN_X_f, tf.exp(self.tf_log_amp[i]), tf.exp(self.tf_log_ls[i])
            )
            S = Knn + (1.0 / tf.exp(self.tf_log_tau[i])) * tf.eye(N, dtype=tf.float64)
            y = tf.slice(self.tf_y, [0, i], [-1, 1])
            lpy += 0.5 * tf.linalg.logdet(S) + 0.5 * tf.matmul(
                tf.transpose(y), self.solve(S, y)
            )
        return lpy

    def get_lpg(self):
        # produce g via differential operator
        # e.g. fd, cl, momentum
        if self.output_dim == 2:
            q, v, eta = self.pred_q_v_eta(self.tf_X_g)
        elif self.output_dim == 1:
            q, eta = self.pred_q_v_eta(self.tf_X_g)
        rho = tf.divide(q, (v + 1e-2))
        x = self.tf_X_g[0]
        t = self.tf_X_g[1]
        g1 = self.conservative_law_constraint(x, t, q, v, rho)
        g2 = self.fd_constraint1(v, rho)
        g3 = self.fd_constraint2(q, rho)
        gl = []
        if kfm == "PW" or kfm == "ARZ" or kfm == "LWR":
            gl.extend([g1, g2, g3])
        if kfm == "PW":
            t_0 = tf.exp(self.tf_t_0)
            c_0 = tf.exp(self.tf_c_0)
            g4 = self.PW_momentum(x, t, v, rho, t_0, c_0)
            gl.extend([g4])
        if kfm == "ARZ":
            t_1 = tf.exp(self.tf_t_1)
            g5 = self.ARZ_momentum(x, t, v, rho, t_1)
            gl.extend([g5])
        if kfm == "HEAT":
            g6 = self.heat(t, q, self.tf_u[0])
            # g7 = self.heat(t, v, self.tf_u[1])
            gl.extend([g6])

        # normal distribution rng
        dist = tf.distributions.Normal(loc=0.0, scale=1.0)

        # for each eq create one regularization term
        # likelihood part II gamma*log[p(0|D)]
        lpg = tf.constant(0.0, dtype=tf.float64)

        for i in range(len(gl)):
            C = self.kernel_rbf.matrix(
                tf.concat(self.tf_X_g, 1), tf.exp(self.tf_log_ls_g[i])
            )
            lpg += (
                self.gamma
                * 0.5
                * (
                    tf.linalg.logdet(C)
                    + tf.matmul(tf.transpose(gl[i]), self.solve(C, gl[i]))
                )
            )
        return lpg

    def pred_q_v_eta(self, X):
        # p(f|Z)
        means, stds = self.pred_multi_dist(tf.concat(X, 1))
        if self.output_dim == 2:
            q_mean, v_mean = means[0], means[1]
            q_std, v_std = stds[0], stds[1]
            eta = tf.random_normal(tf.shape(q_mean), dtype=tf.float64)
            q = q_mean + q_std * eta
            v = v_mean + v_std * eta
            return q, v, eta
        elif self.output_dim == 1:
            q_mean = means[0]
            q_std = stds[0]
            eta = tf.random_normal(tf.shape(q_mean), dtype=tf.float64)
            q = q_mean + q_std * eta
            return q, eta

    # encoding physical knowledge ------------------------------------------------
    # tf.gradients(y,x) symbolic derivatives \partial_x y
    # LWR model
    def conservative_law_constraint(self, x, t, q, v, rho):
        """
        partial_t rho + partial_x q = 0
        """
        g = 0
        g += tf.gradients(rho, t)[0]
        g += tf.gradients(q, x)[0]
        return g

    # fd pde
    def fd_V(self, rho):
        """
        if rho < rc:
            return u
        else:
            return q/rho
        
        u=qm/rc
        (0, u)-(rho_c,u)-(rho_max,0)
        """
        qm = tf.exp(self.tf_log_q_max)
        rc = tf.exp(self.tf_log_rho_c)
        u = tf.fill([self.batch_sz, 1], tf.divide(qm, rc + 1e-2))
        return tf.where(
            tf.math.less_equal(rho, rc), u, tf.divide(self.fd_Q(rho), rho + 1e-2)
        )

    def fd_Q(self, rho):
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
        qm = tf.exp(self.tf_log_q_max)
        rm = tf.exp(self.tf_log_rho_max)
        rc = tf.exp(self.tf_log_rho_c)
        u = tf.divide(qm, rc + 1e-2)
        return tf.where(
            tf.math.less_equal(rho, rc),
            u * rho,
            tf.divide(qm, (rm - rc) * (rm - rho) + 1e-2),
        )

    def fd_constraint1(self, v, rho):
        """
        v - V(rho) = 0
        """
        g = 0
        g += v - self.fd_V(rho)
        return g

    def fd_constraint2(self, q, rho):
        """
        q - Q(rho) = 0
        """
        g = 0
        g += q - self.fd_Q(rho)
        return g

    # 2nd order models
    def PW_momentum(self, x, t, v, rho, t_0, c_0):
        """
        \partial_t v+v\partial_x v+\frac{V-V(\rho)}{\tau_0}+\frac{c^2_0}{\rho}\partial_x\rho = 0
        """
        g = 0
        g += tf.gradients(v, t)[0]
        g += v * tf.gradients(v, x)[0]
        g += tf.divide((v - self.fd_V(rho)), (t_0 + 1e-2))
        g += tf.divide((c_0 * c_0), (rho + 1e-2)) * tf.gradients(rho, x)[0]
        return g

    def ARZ_momentum(self, x, t, v, rho, t_0):
        """
        \partial_t(v-V(\rho) + v\partial_x (v-V(\rho))+\frac{v-V(\rho)}{\tau_0} = 0
        """
        g = 0
        g += tf.gradients(v - self.fd_V(rho), t)[0]
        g += v * tf.gradients(v - self.fd_V(rho), x)[0]
        g += tf.divide((v - self.fd_V(rho)), t_0 + 1e-2)
        return g

    def heat(self, t, v, u):
        """
        partial_v / partial_t=D \nabla \nabla v
        """
        g = 0
        g += tf.gradients(v, t)[0]
        g += -u * tf.gradients(tf.gradients(v, t)[0], t)[0]
        return g

    # -----------------------------------------------------------

    def pred_multi_dist(self, X):
        """
        predict f* at X*
        e.g.
        (x, t) -> (q, v)
        @return [q_mean, v_mean], [q_std, v_std]
        """
        # output means and stds
        means = []
        stds = []
        for i in range(self.output_dim):
            NN_X_f = self.tf_X_f
            Knn = self.kernel_ard.matrix(
                NN_X_f, tf.exp(self.tf_log_amp[i]), tf.exp(self.tf_log_ls[i])
            )
            N = tf.shape(self.tf_X_f)[0]
            S = Knn + 1.0 / tf.exp(self.tf_log_tau[i] + jitter) * tf.eye(
                N, dtype=tf.float64
            )
            NN_X = X
            Kmn = self.kernel_ard.cross(
                NN_X, NN_X_f, tf.exp(self.tf_log_amp[i]), tf.exp(self.tf_log_ls[i])
            )
            y = tf.slice(self.tf_y, [0, i], [tf.shape(self.tf_y)[0], 1])

            # Kmn*S^-1*y
            post_mean = tf.matmul(Kmn, self.solve(S, y))
            means.append(post_mean)

            post_var = tf.exp(self.tf_log_amp[i]) - tf.reduce_sum(
                Kmn * tf.transpose(self.solve(S, tf.transpose(Kmn))), 1
            )
            post_std = tf.reshape(tf.sqrt(post_var), tf.shape(post_mean))
            stds.append(post_std)

        return means, stds

    def create_data_dict(self):
        X_train = self.cfg["X_train"].reshape((-1, self.input_dim))
        y_train = self.cfg["Y_train"].reshape((-1, self.output_dim))
        X_test = self.cfg["X_test"].reshape((-1, self.input_dim))
        X_g = np.zeros((self.batch_sz, self.input_dim))
        for i in range(self.input_dim):
            X_g[:, i] = np.arange(self.imin[0][i], self.imax[0][i], self.batch_sz)

        tf_dict = {
            self.tf_X_g[i]: X_g[:, i].reshape([-1, 1]) for i in range(self.input_dim)
        }
        tf_dict.update(
            {self.tf_X_f: X_train, self.tf_y: y_train, self.tf_X_test: X_test}
        )
        return tf_dict

    def train(self, ofile, y_mean_data=[0] * 2, y_std_data=[1] * 2, verbose=False):
        nepoch = self.cfg["nepoch"]

        y_test = self.cfg["Y_test"].reshape((-1, self.output_dim))

        # training monitor
        writer = tf.summary.FileWriter("./output", self.sess.graph)

        # parameters
        param = []
        for i in range(self.num_eq):
            param.append(
                tf.summary.scalar(name="ls_g" + str(i), tensor=self.tf_log_ls_g[i])
            )

        for i in range(self.output_dim):
            param.append(
                tf.summary.scalar(name="ls_f" + str(i), tensor=self.tf_log_ls[i])
            )
            param.append(
                tf.summary.scalar(name="tau" + str(i), tensor=self.tf_log_tau[i])
            )

        if kfm == "LWR" or kfm == "PW" or kfm == "ARZ":
            param.append(tf.summary.scalar(name="rho_c", tensor=self.tf_log_rho_c))
            param.append(tf.summary.scalar(name="rho_max", tensor=self.tf_log_rho_max))
            param.append(tf.summary.scalar(name="q_max", tensor=self.tf_log_q_max))

        s = time.time()
        # header composing
        of = open(ofile, "a")
        header = "epoch, ELBO, lpy, lpg,"
        for i in range(self.output_dim):
            header += "MSETest" + str(i) + ", "
            header += "MAETest" + str(i) + ","
            header += "MAPETest" + str(i) + ","
            header += "R2Test" + str(i) + ","
            header += "MSETrain" + str(i) + ", "
            header += "MAETrain" + str(i) + ","
            header += "MAPETrain" + str(i) + ","
            header += "R2Train" + str(i) + ","
            header += "tau" + str(i) + ","
            header += "ls" + str(i) + ","
            header += "amp" + str(i) + ","

        for i in range(self.num_eq):
            header += "ls_g" + str(i) + ","

        if kfm == "LWR" or kfm == "PW" or kfm == "ARZ":
            header += "rho_c, rho_max, q_max,"
        if kfm == "PW":
            header += "t0, c0,"
        if kfm == "ARZ":
            header += "t1,"
        header += "\n"
        of.write(header)

        for i in range(nepoch):
            try:
                # training
                tf_dict = self.create_data_dict()
                # weighted objective fun
                #                 self.sess.run(self.minimizer, feed_dict=tf_dict)
                self.sess.run(self.minimizer1, feed_dict=tf_dict)
                if tfm == "PIGP":
                    self.sess.run(self.minimizer2, feed_dict=tf_dict)

                # testing
                if verbose or i % 10 == 0:

                    # obj
                    elbo = self.sess.run(self.nELBO, feed_dict=tf_dict)
                    lpy = self.sess.run(self.get_lpy(), feed_dict=tf_dict)
                    lpg = self.sess.run(self.get_lpg(), feed_dict=tf_dict)

                    print(
                        "epoch = %d,\nELBO = %e,\nlpy = %e,\nlpg = %e"
                        % (i, elbo, lpy, lpg)
                    )
                    of.write("%d, %e, %e, %e," % (i, elbo, lpy, lpg))

                    # y_pred
                    m_test, _ = self.sess.run(self.tf_y_test, feed_dict=tf_dict)
                    m_train, _ = self.sess.run(self.tf_y_train_pred, feed_dict=tf_dict)

                    # performance metrics
                    y_mean_data = self.cfg["y_mean"]
                    y_std_data = self.cfg["y_std"]
                    y_train = self.cfg["Y_train"].reshape((-1, self.output_dim))

                    # reverse normalization
                    for j in range(self.output_dim):
                        # testing set
                        y_true = y_test[:, j] * y_std_data[j] + y_mean_data[j]
                        y_pred = m_test[j] * y_std_data[j] + y_mean_data[j]
                        mse = mean_squared_error(y_true, y_pred)
                        mae = mean_absolute_error(y_true, y_pred)
                        mape = mean_absolute_percentage_error(
                            y_true, np.squeeze(y_pred)
                        )
                        r2 = r2_score(y_true, y_pred)

                        # training set
                        y_true_train = y_train[:, j] * y_std_data[j] + y_mean_data[j]
                        y_pred_train = m_train[j] * y_std_data[j] + y_mean_data[j]
                        mset = mean_squared_error(y_true_train, y_pred_train)
                        maet = mean_absolute_error(y_true_train, y_pred_train)
                        mapet = mean_absolute_percentage_error(
                            y_true_train, np.squeeze(y_pred_train)
                        )
                        r2t = r2_score(y_true_train, y_pred_train)

                        # gp parameter
                        tau = np.exp(self.tf_log_tau[j].eval(session=self.sess))
                        ls = np.exp(self.tf_log_ls[j].eval(session=self.sess))
                        amp = np.exp(self.tf_log_amp[j].eval(session=self.sess))

                        print("MSETest%d = %e" % (j, mse))
                        print("MAETest%d = %e" % (j, mae))
                        print("MAPETest%d = %e" % (j, mape))
                        print("R2Test%d = %e" % (j, r2))
                        print("MSETrain%d = %e" % (j, mset))
                        print("MAETrain%d = %e" % (j, maet))
                        print("MAPETrain%d = %e" % (j, mapet))
                        print("R2Train%d = %e" % (j, r2t))
                        print("tau%d = %e" % (j, tau))
                        print("ls%d = %e" % (j, ls))
                        print("amp%d = %e" % (j, amp))
                        of.write(
                            "%e,%e,%e,%e,%e,%e,%e,%e,%e,%e,%e,"
                            % (mse, mae, mape, r2, mset, maet, mapet, r2t, tau, ls, amp)
                        )

                        if i == EPOCHS - 1:
                            np.save(
                                kfm + str(j) + "_" + this_time,
                                y_true=y_true,
                                y_pred=y_pred,
                            )

                    for j in range(self.num_eq):
                        ls_g = np.exp(self.tf_log_ls_g[j].eval(session=self.sess))
                        print("ls_g%d = %g" % (j, ls_g))
                        of.write("%g," % (ls_g))

                    if kfm == "LWR" or kfm == "PW" or kfm == "ARZ":
                        rho_c = self.tf_log_rho_c.eval(session=self.sess)
                        rho_max = self.tf_log_rho_max.eval(session=self.sess)
                        q_max = self.tf_log_q_max.eval(session=self.sess)
                        print(
                            "rho_c = %g,\nrho_max = %g,\nq_max = %g"
                            % (rho_c, rho_max, q_max)
                        )

                        of.write("%g, %g, %g," % (rho_c, rho_max, q_max))
                    if kfm == "PW":
                        t0 = self.tf_t_0.eval(session=self.sess)
                        c0 = self.tf_c_0.eval(session=self.sess)
                        print("t0 =", t0)
                        print("c0 =", c0)
                        of.write("%g,%g," % (t0, c0))

                    if kfm == "ARZ":
                        t1 = self.tf_t_1.eval(session=self.sess)
                        print("t1 =", t1)
                        of.write("%g," % (t1))

                    of.write("\n")
                    of.flush()
                    print("=" * 32)

                # param monitoring
                for p in param:
                    writer.add_summary(self.sess.run(p), i)
            except:
                import traceback

                traceback.print_exc()
                break
        of.close()
        writer.close()
        if TEST:
            for j in range(self.output_dim):
                # testing set
                y_true = y_test[:, j] * y_std_data[j] + y_mean_data[j]
                y_pred = m_test[j] * y_std_data[j] + y_mean_data[j]
                np.savez(
                    directory + "test" + str(j) + "_" + this_time,
                    y_true=y_true,
                    y_pred=np.squeeze(y_pred),
                )
            # save tf graph
            # self.save()
        print("duration %f" % (time.time() - s))

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, this_time)

    def load(self):
        new_saver = tf.train.import_meta_graph("my_test_model-1000.meta")
        new_saver.restore(self.sess, tf.train.latest_checkpoint(directory))


# ------------------------

if __name__ == "__main__":
    # redirect output
    this_time = time.strftime("%Y-%m-%dT%H%M%S")
    # sys.stdout = open(directory + this_time + "result" + ".txt", "w")
    # sys.stderr = open(directory + this_time + "error" + ".log", "w")

    # parameters
    layers = [2, 20, 20, 20, 20, 20]
    g_list = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    g_idx = 4
    gamma = g_list[g_idx]

    # read data
    if DATA == 1:
        data = np.load(directory + "data.npz")
    elif DATA == 2:
        data = np.load(directory + "data2.npy")
    elif DATA == 3:
        data = np.load(directory + "data3.npz")
    sample_size = SAMPLE_SIZE
    test_size = 288 * 2

    # 1 seperate set
    if DATA_MODE == 1:
        X_train = data["X_train"][:sample_size].astype(np.float64)
        X_test = data["X_test"][:test_size].astype(np.float64)
        Y_train = data["Y_train"][:sample_size].astype(np.float64)
        Y_test = data["Y_test"][:test_size].astype(np.float64)

    elif DATA_MODE == 3:
        # 3 slice and shuffle
        X = data["X_train"][: sample_size + test_size].astype(np.float64)
        Y = data["Y_train"][: sample_size + test_size].astype(np.float64)
        from sklearn.utils import shuffle

        X, Y = shuffle(X, Y)
        X_train = X[:sample_size]
        X_test = X[sample_size : sample_size + test_size]
        Y_train = Y[:sample_size]
        Y_test = Y[sample_size : sample_size + test_size]

    elif DATA_MODE == 4:
        # 4 shuffle and slice
        if DATA == 1 or DATA == 3:
            X = data["X_train"].astype(np.float64)
            X = np.append(X, data["X_test"], axis=0).astype(np.float64)
            Y = data["Y_train"].astype(np.float64)
            Y = np.append(Y, data["Y_test"], axis=0).astype(np.float64)

            # early noise
            if NOISE:
                l = int(X.shape[0] * noise_level)
                # X_train[:l,:] = X_train[:l,:] + np.random.normal(0, 15, X_train[:l,:].shape)
                Y[:l, 0] = Y[:l, 0] + np.random.normal(
                    noise_bias, noise_std_dev, Y[:l, 0].shape
                )
                Y[:l, 1] = Y[:l, 1] + np.random.normal(5, 0, Y[:l, 1].shape)

            from sklearn.utils import shuffle

            X, Y = shuffle(X, Y)
            X_train = X[:sample_size]
            X_test = X[sample_size : sample_size + test_size]
            Y_train = Y[:sample_size]
            Y_test = Y[sample_size : sample_size + test_size]

        elif DATA == 2:
            X = data[:, :2]
            Y = data[:, 2:]
            from sklearn.utils import shuffle

            X, Y = shuffle(X, Y)
            X_train = X[:sample_size]
            X_test = X[sample_size : sample_size + test_size]
            Y_train = Y[:sample_size]
            Y_test = Y[sample_size : sample_size + test_size]

    elif DATA_MODE == 2:
        # 2 slice no shuffle
        X_train = data["X_train"][:sample_size].astype(np.float64)
        X_test = data["X_train"][sample_size : sample_size + test_size].astype(
            np.float64
        )
        Y_train = data["Y_train"][:sample_size].astype(np.float64)
        Y_test = data["Y_train"][sample_size : sample_size + test_size].astype(
            np.float64
        )

    elif DATA_MODE == 5 and DATA == 3:
        # 5 slice and shuffle x=[pos]
        X = data["X_train"][: sample_size + test_size].astype(np.float64)
        Y = data["Y_train"][: sample_size + test_size].astype(np.float64)
        from sklearn.utils import shuffle

        X, Y = shuffle(X, Y)
        X_train = X[:sample_size][:, 1:2]
        X_test = X[sample_size : sample_size + test_size][:, 1:2]
        Y_train = Y[:sample_size]
        Y_test = Y[sample_size : sample_size + test_size]

    else:
        print("error: no such data mode")

    # normalization
    y_mean = np.mean(Y_train, axis=0)
    y_std = np.std(Y_train, axis=0)

    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)

    x_std[x_std == 0] = 1
    y_std[y_std == 0] = 1

    Y_train = (Y_train - y_mean) / y_std
    Y_test = (Y_test - y_mean) / y_std

    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    cfg = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_mean": y_mean,
        "y_std": y_std,
        "imax": np.max(X_train, axis=0),
        "imin": np.min(X_train, axis=0),
        "batch_sz": 100,
        "input_dim": 2,
        "output_dim": 2,
        "gamma": gamma,
        "layers": layers,
        "nepoch": EPOCHS,
        "lr": 1e-2,
    }
    tf.config.optimizer.set_jit(True)
    model = PIGPR(cfg)
    filename = (
        directory
        + tfm
        + "_"
        + kfm
        + "_"
        + str(EPOCHS)
        + "_"
        + str(SAMPLE_SIZE)
        + "_D"
        + str(DATA)
    )
    if NOISE:
        filename += "_noise" + str(noise_level) + "_" + str(noise_bias)
    ofile = filename + "_" + this_time + ".csv"
    model.train(ofile)
