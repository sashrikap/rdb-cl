from gym.envs.registration import registry, register, make, spec

# Base class
# ----------------------------------------

register(
    id="InteractiveDrive-v0",
    entry_point="rdb.envs.drive2d.worlds.interactive:SingleInteractiveEnv",
)


# Drive world
from .worlds import week3, week4, week5

# Automatically register all Week3_xx environmnts
for k, c in week3.__dict__.items():
    if "Week3_" in k:
        register(id=f"{k}-v0", entry_point=f"{c.__module__}:{c.__name__}")


# Automatically register all Week4_xx environmnts
for k, c in week4.__dict__.items():
    if "Week4_" in k:
        register(id=f"{k}-v0", entry_point=f"{c.__module__}:{c.__name__}")

# Automatically register all Week5_xx environmnts
for k, c in week5.__dict__.items():
    if "Week5_" in k:
        register(id=f"{k}-v0", entry_point=f"{c.__module__}:{c.__name__}")
