"""
Compute the sensitivities of a network trained on a
particular ``model`` to its input parameters.
"""
from nn_adapt.ann import *
from nn_adapt.features import collect_features
from nn_adapt.parse import argparse, positive_int
from nn_adapt.plotting import *

import git
import importlib
import numpy as np


# Parse model
parser = argparse.ArgumentParser(
    prog="compute_importance.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model",
    help="The model",
    type=str,
    choices=["turbine"],
)
parser.add_argument(
    "num_training_cases",
    help="The number of training cases",
    type=positive_int,
)
parser.add_argument(
    "-a",
    "--approaches",
    nargs="+",
    help="Adaptive approaches to consider",
    choices=["isotropic", "anisotropic"],
    default=["anisotropic"],
)
parser.add_argument(
    "--adaptation_steps",
    help="Steps to learn from",
    type=positive_int,
    default=3,
)
parser.add_argument(
    "--preproc",
    help="Data preprocess function",
    type=str,
    choices=["none", "arctan", "tanh", "logabs"],
    default="arctan",
)
parser.add_argument(
    "--tag",
    help="Model tag (defaults to current git commit sha)",
    default=git.Repo(search_parent_directories=True).head.object.hexsha,
)
parsed_args = parser.parse_args()
model = parsed_args.model
preproc = parsed_args.preproc
tag = parsed_args.tag

# Load the model
layout = importlib.import_module(f"{model}.network").NetLayout()
nn = SimpleNet(layout).to(device)
nn.load_state_dict(torch.load(f"{model}/model_{tag}.pt"))
nn.eval()
loss_fn = Loss()

# Compute (averaged) sensitivities of the network to the inputs
sensitivity = torch.zeros(layout.num_inputs)
count = 0
data_dir = f"{model}/data"
approaches = parsed_args.approaches
for step in range(parsed_args.adaptation_steps):
    for approach in approaches:
        for test_case in range(1, parsed_args.num_training_cases + 1):
            if test_case == 1 and approach != approaches[0]:
                continue
            suffix = f"{test_case}_GO{approach}_{step}"

            # Load some data and mark inputs as independent
            data = {
                key: np.load(f"{data_dir}/feature_{key}_{suffix}.npy")
                for key in layout.inputs
            }
            target = np.load(f"{data_dir}/target_{suffix}.npy")
            features = torch.from_numpy(collect_features(data)).type(torch.float32)
            expected = torch.from_numpy(target).type(torch.float32)
            features.requires_grad_(True)

            # Evaluate the loss
            loss = loss_fn(expected, nn(features))

            # Backpropagate and average to get the sensitivities
            loss.backward()
            sensitivity += features.grad.abs().mean(axis=0)
            count += 1
sensitivity /= count
np.save(f"{model}/data/sensitivities_{tag}.npy", sensitivity)
