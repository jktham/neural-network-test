import loader
import network as n

training_data, validation_data, test_data = loader.load_data_wrapper()

net = n.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
