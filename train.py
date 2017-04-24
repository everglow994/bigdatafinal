import numpy
import net
import data
import metric
numpy.random.seed(0)

# data
train_users, train_x, test_users, test_x = data.load_data()
train_x_users = numpy.array(train_users, dtype=numpy.int32).reshape(len(train_users), 1)
test_x_users = numpy.array(test_users, dtype=numpy.int32).reshape(len(test_users), 1)

# model
model = net.create(I=train_x.shape[1], U=len(train_users)+1, K=50,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# train
history = model.fit(x=[train_x, train_x_users], y=train_x,
                    batch_size=128, nb_epoch=1000, verbose=1,
                    validation_data=[[test_x, test_x_users], test_x])

# predict
pred = model.predict(x=[train_x, train_x_users])
# remove watched items from predictions
pred *= (train_x == 0)
pred_arg = numpy.argsort(pred)


N = [1, 10, 20]
for i in range(3):
    sr = 0.000
    for j in range(pred.shape[0]):
        sr += metric.apk(test_x[j], pred_arg[j, -N[i]:], N[i])
    print("mAP at Top@{:d} Recommendation is: {:f}".format(N[i], sr/pred.shape[0]))

