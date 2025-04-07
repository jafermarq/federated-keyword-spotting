"""flower-kws: A Flower / PyTorch app."""

from datasets import load_dataset
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from flower_kws.audio import get_apply_transforms_fn
from flower_kws.task import Net, get_device, get_weights, set_weights, test


def get_evaluate_fn(
    validation_loader,
):
    """Return a callback that the strategy will call after models are aggregated."""

    def evaluate(server_round: int, parameters, config):
        """Evaluate global model on a centralized dataset."""

        model = Net()
        set_weights(model, parameters)

        # Determine device
        device = get_device()

        # Run test
        loss, accuracy = test(model, validation_loader, device)
        return loss, {"accuracy": accuracy}

    return evaluate


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load validation split
    sc_val = load_dataset("speech_commands", "v0.02", split="validation", token=False)
    # Apply processing
    validation_set = sc_val.map(get_apply_transforms_fn(), batch_size=32)
    validation_set.set_format(type="torch", columns=["mfcc", "target"])
    # Create data loader
    valloader = DataLoader(validation_set, batch_size=32)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(valloader),
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
