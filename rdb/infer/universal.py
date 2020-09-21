"""Experimental

Universal Planning Network
"""


from jax.experimental import stax


def create_model(input_dim, output_dim, hidden_dim=200, dropout=0.2, mode="train"):

    init, predict = stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(output_dim, W_init=stax.randn()),
        # stax.Dropout(dropout, mode=mode),
    )

    return init, predict
