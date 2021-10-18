from typing import TypedDict


class ModelParameters(TypedDict):
    filters: int
    kernel_size: int
    pool_type: str
    regularizer: str
    activation_type: str
    epochs: int
    batch_size: int
    loss_func: str
    optimizer: str


# TODO: Use .ini for parameters
def get_parameters(file_name: str) -> ModelParameters:
    dict = ModelParameters()
    with open(file_name) as f:
        for line in f:
            (key, val) = line.split()
            dict[key] = val

    # change string values to integer values
    dict["filters"] = int(dict["filters"])
    dict["kernel_size"] = int(dict["kernel_size"])
    dict["epochs"] = int(dict["epochs"])
    dict["batch_size"] = int(dict["batch_size"])

    return dict
