# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
import numpy as np
import random as ran

#设置实验的环境
env = gym.make('CartPole-v0')


#经验池
REPLAY_MEMORY = []
#批次大小
MINIBATCH = 50
#输入，即为某个时刻的环境状态
INPUT = env.observation_space.shape[0]
#输出，这个环境中即为左右两个动作
OUTPUT = env.action_space.n
#优化算法中的学习率
LEARNING_LATE = 0.001
#折扣因子
DISCOUNT = 0.99
#保存模型的路径
model_path= "./save/model.ckpt"
x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT), name='x')
y = tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT), name='y')


# 与环境互动的Q网络
with tf.variable_scope('eval_net'):

    W1 = tf.get_variable('W1', shape=[INPUT, 200], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable('W2', shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable('W3', shape=[200, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.zeros([1], dtype=tf.float32))
    b2 = tf.Variable(tf.zeros([1], dtype=tf.float32))

    L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    Q_pre = tf.matmul(L2, W3)

#为了测试而将变量加入集合
tf.add_to_collection("pre_net",Q_pre)


# 目标Q网络，与评估用的Q网络结构一致
with tf.variable_scope('target_net'):

    W1_r = tf.get_variable('W1_r', shape=[INPUT, 200])
    W2_r = tf.get_variable('W2_r', shape=[200, 200])
    W3_r = tf.get_variable('W3_r', shape=[200, OUTPUT])

    b1_r = tf.Variable(tf.zeros([1], dtype=tf.float32))
    b2_r = tf.Variable(tf.zeros([1], dtype=tf.float32))

    L1_r = tf.nn.relu(tf.matmul(x, W1_r) + b1_r)
    L2_r = tf.nn.relu(tf.matmul(L1_r, W2_r) + b2_r)
    Q_pre_r = tf.matmul(L2_r, W3_r)



recent_rlist = [0]

episode = 0

#损失函数和优化函数
cost = tf.reduce_sum(tf.square(y - Q_pre))
optimizer = tf.train.AdamOptimizer(LEARNING_LATE, epsilon=0.01)
train = optimizer.minimize(cost)

#更新目标Q网络的参数
t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
replace_target_para = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

#学习过程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    sess.run(replace_target_para)

    while np.mean(recent_rlist) < 195:
        episode += 1
        #每个episode都要将环境恢复初始状态
        s = env.reset()
        #超过200 episode时，用最新的reward更新列表
        if len(recent_rlist) > 200:
            del recent_rlist[0]
        # e-greedy
        e = 1. / ((episode / 25) + 1)

        rall = 0  # reward_all
        d = False  #杆子是否倒下，即回合结束
        count = 0 #一个episode里训练的步数

        while not d and count < env.spec.max_episode_steps:

            # env.render()
            count += 1

            # 环境状态塑形为一行,
            s = np.reshape(s, [1, INPUT])

            Q = sess.run(Q_pre, feed_dict={x: s})

            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(Q)

            s1, r, d, _ = env.step(a)
            #填充经验池
            REPLAY_MEMORY.append([s, a, r, s1, d])

            #经验池数量多了则更新
            if len(REPLAY_MEMORY) > 50000:
                del REPLAY_MEMORY[0]

            rall += r

            s = s1
        #每10个episode更新目标Q函数的参数
        if episode % 10 == 1 and len(REPLAY_MEMORY) > 50:

            for sample in ran.sample(REPLAY_MEMORY, MINIBATCH):
                s_r, a_r, r_r, s1_r, d_r = sample

                Q = sess.run(Q_pre, feed_dict={x: s_r})

                if d_r:
                   Q[0, a_r] = r_r
                else:
                    s1_r = np.reshape(s1_r, [1, INPUT])
                    Q1 = sess.run(Q_pre_r, feed_dict={x: s1_r})
                    Q[0, a_r] = r_r + DISCOUNT * np.max(Q1)
					
                _, loss = sess.run([train, cost], feed_dict={x: s_r, y: Q})

            sess.run(replace_target_para)
            
            print('loss:{} '.format(loss))

        recent_rlist.append(rall)

        print("Episode%d: reward:%.2f recent reward:%.4f" %(episode, rall,np.mean(recent_rlist)))
    #保存模型
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path)
    print("Model saved in file: ", save_path)



