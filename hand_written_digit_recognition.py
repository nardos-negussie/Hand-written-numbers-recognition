
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf

#import dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

#set parameters
learning_rate = 0.01
training_iteration = 40
batch_size = 100
display_step = 2

#tf graph input
x = tf.placeholder('float', [None, 784]) #image shape 28*28 = 784
y = tf.placeholder('float', [None, 10]) #0-9 digits


### create the model

#set the model weights
W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros(10))


with tf.name_scope('Wx_b') as scope:
    #construct a linear model
    model = tf.nn.softmax(tf.matmul(x,W) + b)  #softmax
    

w_h = tf.summary.histogram('weights', W)
b_h = tf.summary.histogram('biases', b)

#more name scopes will clean up graph representation
with tf.name_scope('cost_function') as scope:
    #minimize the error using cross-entropy
    cost_function = -tf.reduce_sum(y*tf.log(model+0.0001)) # cross-entropy
    tf.summary.scalar('cost_function', cost_function)

with tf.name_scope('train') as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_function)
    
                                                                             

#initialize the variables
init = tf.global_variables_initializer()

merged_summary_op = tf.summary.merge_all()


### Launch the graph

with tf.Session() as sess:
    sess.run(init)
    
    #set the logs writer to the folder
    #summary_writer = tf.train.write_graph(sess.graph_def,'/home/nardos/Documents/PLSight/log', False)
    
    #training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for b in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            #fit training using batch data
            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
            
            #compute the average cost
            avg_cost += sess.run(cost_function, feed_dict={x:batch_xs, y:batch_ys})/total_batch
            
            #summary_str = sess.run(merged_summary_op, feed_dict={x:batch_xs, y:batch_ys})
            #summary_writer.add_summary(summary_str, iteration*total_batch+i)
        #display logs per iteration step
        if iteration % display_step ==0:
            print('Iteration:', '%04d' % (iteration+1), 'cost=', '{:.9f}'.format(avg_cost))
    print('Training completed.')
    
    predictions = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
    
    accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

