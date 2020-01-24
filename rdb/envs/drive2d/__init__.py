from gym.envs.registration import registry, register, make, spec

# Base class
# ----------------------------------------

register(
    id="InteractiveDrive-v0",
    entry_point="rdb.envs.drive2d.worlds.interactive:SingleInteractiveEnv",
)


# Drive world
from .worlds import week3, week4, week5, week6


def register_key_class(key, cls):
    if "_v1" in key:
        key = key.replace("_v1", "")
        register(id=f"{key}-v1", entry_point=f"{c.__module__}:{c.__name__}")
    elif "_v0" in key:
        key = key.replace("_v0", "")
        register(id=f"{key}-v0", entry_point=f"{c.__module__}:{c.__name__}")
    else:
        register(id=f"{key}-v0", entry_point=f"{c.__module__}:{c.__name__}")


# Automatically register all Week3_xx environmnts
for k, c in week3.__dict__.items():
    if "Week3_" in k:
        register_key_class(k, c)
for k, c in week4.__dict__.items():
    if "Week4_" in k:
        register_key_class(k, c)
for k, c in week5.__dict__.items():
    if "Week5_" in k:
        register_key_class(k, c)
for k, c in week6.__dict__.items():
    if "Week6_" in k:
        register_key_class(k, c)
