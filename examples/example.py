# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import json
import logging
from pathlib import Path
from random import randint

import matplotlib.pyplot as plt
import pyhf
import pymc as pm
from pyhf_pymc import infer, make_op, plotting, prepare_inference

# %%
logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)

# %%
# Ensure the figures directory exists
figure_path = Path().cwd() / "figures"
figure_path.mkdir(exist_ok=True)
figure_file_extensions = ["png", "pdf"]

# %%
pyhf.set_backend("jax")

# %%
nBins = 3
model = pyhf.simplemodels.correlated_background(
    [10, 15, 20], [50, 40, 30], [55, 45, 35], [45, 35, 25]
)

unconstr_priors = {"mu": {"type": "unconstrained", "input": [[5.0], [1.0]]}}

data = [60, 55, 50]
truth = [0.0, 1.0]

prior_dict = prepare_inference.prepare_priors(model, unconstr_priors)
prepared_model = prepare_inference.prepare_model(
    model=model, observations=data, priors=prior_dict
)

expData_op = make_op.make_op(model)

# %%
# https://www.hepdata.net/record/ins1869695
with open("ttbar_ljets_xsec_inclusive_pruned.json") as serialized:
    spec = json.load(serialized)

workspace = pyhf.Workspace(spec)
model = workspace.model()

data = workspace.data(model, include_auxdata=False)

nBins = len(model.expected_actualdata(model.config.suggested_init()))
nPars = len(model.config.suggested_init())

# Prepare the priors for sampling
# Unconstrained parameters
unconstr_priors = {
    "uncon1": {"type": "unconstrained", "type2": "normal", "input": [[10], [10]]}
}

# Create dictionary with all priors (unconstrained, constrained by normal and poisson)
prior_dict = prepare_inference.prepare_priors(model, unconstr_priors)

# dictionary with keys 'model', 'obs', 'priors', 'precision'
prepared_model = prepare_inference.prepare_model(
    model=model, observations=data, priors=prior_dict
)

# %%
with infer.model(model, unconstr_priors, data):
    step = pm.Metropolis()
    post_data = pm.sample(draws=100, chains=1, step=step)
    post_pred = pm.sample_posterior_predictive(post_data)
    prior_pred = pm.sample_prior_predictive(100)

# %%
fig, ax = plt.subplots()

plotting.prior_posterior_predictives(
    model=model, data=data, post_pred=post_pred, prior_pred=prior_pred, bin_steps=5
)

for ext in figure_file_extensions:
    fig.savefig(figure_path / f"prior_posterior_predictives.{ext}")
