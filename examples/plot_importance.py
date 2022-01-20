from nn_adapt.ann import *
from plotting import *

import argparse
import numpy as np


def Loss():
    """
    Custom loss function.

    Needed when there is only one output value.
    """
    def mse(tens1, tens2):
        return torch.nn.MSELoss(reduction='mean')(tens1, tens2.reshape(*tens1.shape))
    return mse


# Hard-coded parameters
num_test_cases = 12
num_inputs = 29
num_runs = 7

# Parse model
parser = argparse.ArgumentParser(prog='test_importance.py')
parser.add_argument('model', help='The equation set being solved')
parsed_args = parser.parse_args()
model = parsed_args.model
assert model in ['stokes', 'turbine']

# Load the model
nn = SimpleNet().to(device)
nn.load_state_dict(torch.load(f'{model}/model.pt'))
nn.eval()
loss_fn = Loss()

sensitivity = torch.zeros(29)
for approach in ['isotropic', 'anisotropic']:
    for run in range(4):
        if run == 0 and approach == 'anisotropic':
            continue
        for test_case in range(num_test_cases):

            # Load some data and mark inputs as independent
            features = torch.from_numpy(np.load(f'{model}/data/features{test_case}_GO{approach}_{run}.npy')).type(torch.float32)
            expected = torch.from_numpy(np.load(f'{model}/data/targets{test_case}_GO{approach}_{run}.npy')).type(torch.float32)
            features.requires_grad_(True)

            # Evaluate the loss
            loss = loss_fn(expected, nn(features))

            # Backpropagate and average to get the sensitivities
            loss.backward()
            sensitivity += features.grad.abs().mean(axis=0)
sensitivity /= num_test_cases*num_runs

# Plot increases as a bar chart
fig, axes = plt.subplots()
colours = ['C0'] + 4*['deepskyblue'] + 12*['mediumturquoise'] + 12*['mediumseagreen']
axes.bar(list(range(1)), sensitivity[:1], color='C0', label='Physics')
axes.bar(list(range(1, 5)), sensitivity[1:5], color='deepskyblue', label='Mesh')
axes.bar(list(range(5, 17)), sensitivity[5:17], color='mediumturquoise', label='Forward')
axes.bar(list(range(17, 29)), sensitivity[17:], color='mediumseagreen', label='Adjoint')
axes.set_xticks([])
axes.set_xlabel('Input parameters')
axes.set_ylabel('Network sensitivity')
axes.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{model}/plots/importance.pdf')
