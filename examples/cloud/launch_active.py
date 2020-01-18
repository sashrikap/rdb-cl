from doodad.launch import launch_api
from doodad import mode, mount
from rdb.exps.utils import create_params, load_params, save_params
from tqdm import tqdm


def launch(params_dict):
    locals().update(params_dict)
    print(params_dict)
    save_params("examples/params/active_params.yaml", params_dict)

    filter_ext = (".pyc", ".log", ".git", ".mp4", ".npz", ".ipynb")

    if LOCAL_MODE:
        local_mnt = mount.MountLocal(local_dir="../rdb", mount_point="./rdb")
        launch_mode = (
            mode.LocalMode()
        )  ## Having a 2nd output mount in local mode seems to cause file race condition
        mounts = [local_mnt]
    else:
        gcp_label = ""
        if "RANDOM_KEYS" in params_dict and "EXP_NAME" in params_dict:
            key_str = "_".join([str(k) for k in params_dict["RANDOM_KEYS"]])
            gcp_label = f"{params_dict['EXP_NAME']}_{key_str}"

        launch_mode = mode.GCPMode(
            # zone="us-west1-a",
            # instance_type="n1-standard-4",
            zone="us-west2-a",  # 40 c2 isntance
            # zone="us-east1-b",  # 8 c2 isntance
            instance_type="c2-standard-4",
            preemptible=True,
            gcp_image="active-ird-rss-v01",
            gcp_image_project="aerial-citron-264318",
            gcp_project="aerial-citron-264318",
            gcp_bucket="active-ird-experiments",
            gcp_log_path="rss-logs",  # Folder to store log files under
            terminate_on_end=True,
            gcp_label=gcp_label,
        )
        # By default, /rdb/rdb -> /dar_payload/rdb/rdb
        gcp_mnt = mount.MountLocal(
            local_dir="../rdb", mount_point="./rdb", filter_ext=filter_ext
        )
        output_mnt = mount.MountGCP(
            gcp_path="output",  # Directory on GCP
            mount_point="/gcp_output",  # Directory visible to the running job.
            output=True,
        )
        mounts = [gcp_mnt, output_mnt]

    # This will run locally
    launch_api.run_command(
        command="bash /dar_payload/rdb/examples/cloud/run_active.sh",
        # command="echo 'check out' && ls /gcp_input/200110_test_eval_all",
        # command="pwd && touch good.txt",
        # command="bash ./rdb/examples/cloud/run_pyglet.sh",
        mounts=mounts,
        mode=launch_mode,
    )


if __name__ == "__main__":
    LOCAL_MODE = False

    # ========== Parameters =============
    template = load_params("examples/params/active_template.yaml")

    if LOCAL_MODE:
        launch(template)
    else:
        # params = {"RANDOM_KEYS": list(range(26, 36)), "NUM_EVAL_WORKERS": 8}
        params = {
            # "RANDOM_KEYS": list(range(6)),
            "RANDOM_KEYS": list(range(8)),
            # "RANDOM_KEYS": [20, 21, 22, 23, 24],
            "NUM_EVAL_WORKERS": 16,
        }
        # params = {"RANDOM_KEYS": [9], "NUM_EVAL_WORKERS": 8}
        all_params = create_params(template, params)
        for param in tqdm(all_params, desc="Launching jobs"):
            launch(param)
    # ===================================
