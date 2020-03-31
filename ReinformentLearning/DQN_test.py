import tensorflow as tf
import gym
import numpy as np

env = gym.make('CartPole-v0')
model_path= "./save/model.ckpt"
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n

rlist = []

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./save/') #检查是否有预训练的模型
    if ckpt:
        #恢复模型
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./save/model.ckpt.meta')
        saver.restore(sess, './save/model.ckpt')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        Q_pre = tf.get_collection("pre_net")[0] #从模型中恢复Q_pre

        print("Model restored form file: ", model_path)
        for episode in range(100):
            s = env.reset()
            rall = 0
            d = False
            count = 0

            while not d:
                env.render()
                count += 1
                s_t = np.reshape(s, [1, INPUT])
                Q = sess.run(Q_pre, feed_dict={x: s_t})
                a = np.argmax(Q)
                s, r, d, _ = env.step(a)
                rall += r

            rlist.append(rall)

            print("Episode%d: steps: %d Reward=%.2f Averge_Reward=%.4f"%(episode+1, count, rall,
                                                                            np.mean(rlist)))
    else:
        print('No checkpoint file found.')