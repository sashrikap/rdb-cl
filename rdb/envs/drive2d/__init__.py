from gym.envs.registration import registry, register, make, spec

# Base class
# ----------------------------------------

register(
    id="InteractiveDrive-v0",
    entry_point="rdb.envs.drive2d.worlds.interactive:SingleInteractiveEnv",
)


register(
    id="HighwayDrive-v0",
    entry_point="rdb.envs.drive2d.worlds.highway:HighwayDriveWorld",
)

# Drive world
from .worlds import week3

# Automatically register all Week3_xx environmnts
for k, c in week3.__dict__.items():
    if "Week3_" in k:
        register(id=f"{k}-v0", entry_point=f"{c.__module__}:{c.__name__}")
