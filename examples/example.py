# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import json
import logging
from random import randint

import arviz as az
import corner
import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhf
import pymc as pm
import pytensor
from jax import grad, jit, random, value_and_grad, vmap
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pyhf_pymc import infer, make_op, plotting, prepare_inference
from pytensor import tensor as pt
from pytensor.graph import Apply, Op
from pytensor.graph.basic import Apply
from pytensor.tensor.type import TensorType

# %%
logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)

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
