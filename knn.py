import tensorflow as tf
matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[2]])
product=tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)

state=tf.Variable(0,name="counter")
print(state.name)
one=tf.constant(1)

#new_value=tf.add(state+one)
update=tf.assign_add(state,one)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


imput1=tf.placeholder(tf.float32)
imput2=tf.placeholder(tf.float32)

output=tf.multiply(imput1,imput2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={imput2:[7.],imput1:[3.]}))

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
ax.spines['bottom'].set_position(['data',0])
plt.ion()
plt.show()

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

layer1=add_layer(x,1,10,activation_function=tf.nn.relu)
prediction=add_layer(layer1,10,1,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-prediction),reduction_indices=[1]))
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
        if i%50==0:
            print(sess.run(loss,feed_dict={x:x_data,y:y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_y=sess.run(prediction,feed_dict={x:x_data})
            lines=ax.plot(x_data,prediction_y,'r-',lw=5)
            plt.pause(0.5)
