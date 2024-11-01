import numpy
import pandas
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


def get_weight_bias(path_file):
    data_params = pandas.read_csv(path_file, dtype={"name": "string"})
    params_names = data_params["name"].values
    params_values = data_params["val"].values

    # Get number of layers + dims of everything 
    layer_names = params_names[["layer_" in param_name for param_name in params_names]]
    n_layers = len(numpy.unique([layer_name.split("_")[1] for layer_name in layer_names]))
    _w_new = []
    _bias_new = []
    for i in range(1, n_layers + 1):
        # Get weights
        w_names = params_names[["layer_" + str(i) + "_w_" in param_name for param_name in params_names]]
        i_w = numpy.where(["layer_" + str(i) + "_w_" in param_name for param_name in params_names])[0]
        w_dim1 = numpy.max([int(w_name.split("_")[3]) for w_name in w_names])
        w_dim2 = numpy.max([int(w_name.split("_")[4]) for w_name in w_names])
        _w = numpy.zeros((w_dim1, w_dim2))
        for j in range(len(i_w)):
            dim1 = int(w_names[j].split("_")[3]) - 1  # 0 indexing
            dim2 = int(w_names[j].split("_")[4]) - 1  # 0 indexing
            _w[dim1, dim2] = params_values[i_w[j]]
        _w_new.append(_w)
        # Get bias 
        bias_names = params_names[["layer_" + str(i) + "_bias_" in param_name for param_name in params_names]]
        i_bias = numpy.where(["layer_" + str(i) + "_bias_" in param_name for param_name in params_names])[0]
        bias_dim = numpy.max([int(bias_name.split("_")[3]) for bias_name in bias_names])
        _bias = numpy.zeros(bias_dim)
        for j in range(len(i_bias)):
            dim = int(bias_names[j].split("_")[3]) - 1  # 0 indexing
            _bias[dim] = params_values[i_bias[j]]
        _bias_new.append(_bias)

    # Transform to Jax 
    w_new = [jax.numpy.array(_w) for _w in _w_new]
    bias_new = [jax.numpy.array(_bias) for _bias in _bias_new]
    return params_values[0:2], w_new, bias_new


def mlp_from_dir(dir_meta, path_parameters, key):
    meta_data = pandas.read_csv(dir_meta + "/Meta_nn.csv")
    hidden_layers = meta_data["hidden_layers"].values[0]
    width_size = meta_data["width_size"].values[0]
    _activation_function = meta_data["actvation_function"].values[0]
    if _activation_function == "tanh":
        activation_function = jnn.tanh
    _mlp = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=width_size,
        depth=hidden_layers,
        activation=activation_function,
        key=key)
    _dyn_param = jnp.array([1.3, 1.8])
    if path_parameters is not None:
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_bias = lambda m: [x.bias
                              for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                              if is_linear(x)]
        get_weight = lambda m: [x.weight
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                if is_linear(x)]
        _dyn_param, new_weights, new_bias = get_weight_bias(path_parameters)
        _mlp = eqx.tree_at(get_bias, _mlp, new_bias)
        _mlp = eqx.tree_at(get_weight, _mlp, new_weights)
        _dyn_param = jnp.array(_dyn_param)
    return _dyn_param, _mlp
