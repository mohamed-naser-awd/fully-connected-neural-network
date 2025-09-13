from network import Network, InputLayer, HiddenLayer, OutputLayer
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], -1))


samples = []

for i in range(10):
    number = int(y_train[i])  # الرقم (label)
    input_arr = x_train[i].tolist()  # الصورة كـ 1D list
    samples.append((number, input_arr))  # نحطهم كـ tuple

print(samples[0])


def get_mnist_network() -> Network:
    network = Network("mnist")
    network.add_layer(InputLayer(size=28 * 28))
    network.add_layer(HiddenLayer(size=256))
    network.add_layer(HiddenLayer(size=128))
    network.add_layer(OutputLayer(size=10))
    network.set_info()
    return network


if __name__ == "__main__":
    network = get_mnist_network()

    for sample in samples:
        number, input_arr = sample
        res = network.compute(input_arr)
        print(res, number)
