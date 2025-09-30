from export import export_network
from loader import load_network_from_filename
from sample import get_mnist_sample
from trainer.loss import loss_function


if __name__ == "__main__":
    network = load_network_from_filename("mnistnetwork.json")
    img, y_true = get_mnist_sample()
    y_pred, confidence = network.predict(img)

    loss = loss_function(
        y_true,
        network.output_layer.activated_output,
    )

    print(
        f"loss: {loss:.6f}, predicted: {y_pred}, actual: {y_true}, confidence: {confidence:.6f}"
    )

    export_network(network)
