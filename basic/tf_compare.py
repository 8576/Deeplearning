# tensorflow 比较类接口

TensorFlow program that uses reduce_all, reduce_any

import tensorflow as tf

x = [1., 2., 3.]
y = [1., 2., 3.]
z = [0., 1., 3.]

result1 = tf.equal(x, y)
result2 = tf.equal(y, z)

# Use reduce_all and reduce_any to test the results of equal.
result3 = tf.reduce_all(result1)
result4 = tf.reduce_all(result2)
result5 = tf.reduce_any(result1)
result6 = tf.reduce_any(result2)

session = tf.Session()
print("EQUAL     ", session.run(result1))
print("EQUAL     ", session.run(result2))
print("REDUCE ALL", session.run(result3))
print("REDUCE ALL", session.run(result4))
print("REDUCE ANY", session.run(result5))
print("REDUCE ANY", session.run(result6))

Output

EQUAL      [ True  True  True]
EQUAL      [False False  True]
REDUCE ALL True
REDUCE ALL False
REDUCE ANY True
REDUCE ANY True





import tensorflow as tf
#判断每一个数是否大于0.5
greater = tf.greater([1.,0.2,0.5,0.,2.,3.], 0.5)
#判断每一个数是否小于0.5
less = tf.less([1.,0.2,0.5,0.,2.,3.], 0.5)
#判断每一个数是否大于等于0.5
greater_equal=tf.greater_equal([1.,0.2,0.5,0.,2.,3.], 0.5)
#判断每一个数是否小于等于0.5
less_equal=tf.less_equal([1.,0.2,0.5,0.,2.,3.], 0.5)
#判断每一个数是否等于0.5
equal = tf.equal([1.,0.2,0.5,0.,2.,3.], 0.5)
#判断每一个数是否不等于0.5
not_equal=tf.not_equal([1.,0.2,0.5,0.,2.,3.], 0.5)
with tf.Session() as sess:
    print(sess.run(greater))
    print(sess.run(less))
    print(sess.run(equal))
    print(sess.run(greater_equal))
    print(sess.run(less_equal))
    print(sess.run(not_equal))

[ True False False False  True  True]
[False  True False  True False False]
[False False  True False False False]
[ True False  True False  True  True]
[False  True  True  True False False]
[ True  True False  True  True  True]
