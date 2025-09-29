from export import export_network
from loader import load_network_from_filename
from trainer.train import NetworkTrainer


if __name__ == "__main__":
    network = load_network_from_filename("mnistnetwork.json")
    trainer = NetworkTrainer(network)
    trainer.train()

    export_network(network)
