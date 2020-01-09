from doodad.launch import launch_api
from doodad import mode, mount

local = mode.LocalMode()
gcp_mode = mode.GCPMode(
    zone="us-west1",
    instance_type="n1-standard-4",
    gcp_image="active-ird-rss-v00",
    gcp_image_project="aerial-citron-264318",
    gcp_project="aerial-citron-264318",
    gcp_bucket="active-ird-experiments",
    gcp_log_path="rss",  # Folder to store log files under
    terminate_on_end=True,  # Whether to terminate on finishing job
)

mnt = mount.MountLocal(
    local_dir="../rdb",
    # exclude_regex="data/",
    mount_point="./rdb",
    output=False,
)

# This will run locally
launch_api.run_command(
    command="bash ./rdb/examples/cloud/run_acquisition.sh",
    # command="bash ./rdb/examples/cloud/run_pyglet.sh",
    mounts=[mnt],
    mode=local,
    docker_image="ubuntu:16.04",
)

# launch_api.run_python(
#     target='./examples/run_highway.py',
#     mounts=[mnt],
#     mode=local,
# )
