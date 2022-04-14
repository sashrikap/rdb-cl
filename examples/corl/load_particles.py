from rdb.infer import *


def load_file(rng_name, save_dir, save_name):
    weight_params = {
        "normalized_key": "dist_cars",
        "max_weights": 15.0,
        "bins": 200,
        "feature_keys": [
            "dist_cars",
            "dist_lanes",
            "dist_fences",
            "dist_objects",
            "speed",
            "control",
            "speed_over",
        ],
    }
    ps = Particles(
        rng_name=rng_name,
        rng_key=None,
        env_fn=None,
        controller=None,
        runner=None,
        save_name=save_name,
        save_dir=save_dir,
        normalized_key="dist_cars",
        weight_params=weight_params,
    )
    ps.load()
    weights = ps.map_estimate(4, log_scale=False).weights
    digits = ps.digitize(
        weight_params["bins"], weight_params["max_weights"], log_scale=False
    )
    unique_vals, indices, counts = onp.unique(
        digits, return_index=True, return_counts=True
    )
    # ps.log_prob(weights[0])
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    save_dir = (
        "data/200309/active_ird_sum_ibeta_20_true_w_irdvar_3_dbeta_1_602_adam/save"
    )
    save_name = "ird_belief_method_random_itr_00"
    rng_name = "[0 0]"
    load_file(rng_name, save_dir, save_name)
