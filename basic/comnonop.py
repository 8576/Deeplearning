# 指定batch，切片。
start=(i*batch_size)%dataset
        end=min(start+batch_size,dataset)
        sess.run(train_step,feed_dict={x:X[start:end],y_true:Y[start:end]})
