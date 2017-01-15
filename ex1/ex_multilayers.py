import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size,Name ,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]),Name+" weights")
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,Name+" biases")
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    print(Name+" complete!!")

    #activation_function
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# set input and output layer
# tensorflow will put the data into "none" automatically
xs = tf.placeholder(tf.float32, [None, 2],name="input")
ys = tf.placeholder(tf.float32, [None, 2],name="output")

# add hidden layer
l1 = add_layer(xs, 2,10, "l1",activation_function=tf.nn.relu)
l2 = add_layer(l1, 10,10,"l2" ,activation_function=tf.nn.relu)
l3 = add_layer(l2, 10,10,"l3", activation_function=tf.nn.relu)

# add output layer
prediction = add_layer(l3, 10, 2,"prediction", activation_function=None)

#make 2 row:300 col:1 array
x_data1 = np.linspace(-6,6,500)[:, np.newaxis]
x_data2 = np.linspace(-9,9,500)[:, np.newaxis]
#combine x data 1&2 into a 2D array
x = np.concatenate((x_data1, x_data2), axis=1)
noise = np.random.normal(0, 1, x_data1.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = x_data1 * x_data2 - 0.5 + noise
y_data2 = (-1)*x_data1 * x_data2 - 0.5 + noise
print(y_data.shape)
y=np.concatenate((y_data, y_data2), axis=1)
print(y.shape)

# 定義loss function 並且選擇減低loss 的函數 這裡選擇AdadeltaOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
 
#initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
 
# show the figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data1, y_data)
ax.scatter(x_data2, y_data2)

plt.ion()

plt.ylim((-50,50))
plt.show()
 
# train for 10,000 times
for i in range(10000):
   
     # feed the data
     # x_data:[300,1]   y_data:[300,1]
    sess.run(train_step, feed_dict={xs: x, ys: y})
    if i % 50 == 0:
        # remove lines
        try:
            ax.lines.pop(0)
            ax.lines.remove(lines[0])
        except Exception:
            print("cannot remove")
        
        # 要取出預測的數值 必須再run 一次才能取出
        prediction_value = sess.run(prediction, feed_dict={xs: x})
        # 每隔0.1 秒畫出來        
        lines = ax.plot(x_data1, prediction_value[:,0],'r-',lw=5)#, 'r-', lw=5  
        lines2 = ax.plot(x_data2, prediction_value[:,1],'blue',lw=5)

        plt.pause(0.01)

#saver = tf.train.Saver()
#save_path = saver.save(sess, "my_net/save_net.ckpt")
#print("Save to path: ", save_path)