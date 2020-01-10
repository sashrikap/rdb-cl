from doodad.launch import launch_api
from doodad import mode, mount

LOCAL_MODE = False

if LOCAL_MODE:
    local_mnt = mount.MountLocal(local_dir="../rdb", mount_point="./rdb")
    launch_mode = mode.LocalMode()
    ## Having a 2nd output mount in local mode seems to cause file race condition
    mounts = [local_mnt]
else:
    launch_mode = mode.GCPMode(
        # zone="us-west1-a",
        # instance_type="n1-standard-4",
        zone="us-west2-a",
        instance_type="c2-standard-4",
        gcp_image="active-ird-rss-v01",
        gcp_image_project="aerial-citron-264318",
        gcp_project="aerial-citron-264318",
        gcp_bucket="active-ird-experiments",
        gcp_log_path="rss-logs",  # Folder to store log files under
        terminate_on_end=True,  # Whether to terminate on finishing job
    )
    gcp_mnt = mount.MountLocal(local_dir="../rdb", mount_point="./rdb")
    output_mnt = mount.MountGCP(
        gcp_path="output",  # Directory on GCP
        mount_point="/gcp_output",  # Directory visible to the running job.
        output=True,
    )
    mounts = [gcp_mnt, output_mnt]

# This will run locally
launch_api.run_command(
    command="bash /dar_payload/rdb/examples/cloud/run_acquisition.sh",
    # command="pwd && touch good.txt",
    # command="bash ./rdb/examples/cloud/run_pyglet.sh",
    mounts=mounts,
    # mounts=[launch_mnt],
    # mode=local,
    mode=launch_mode,
    # docker_image="ubuntu:16.04",
)

# launch_api.run_python(
#     target='./examples/run_highway.py',
#     mounts=[mnt],
#     mode=local,
# )
