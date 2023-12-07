from gym.envs.registration import registry, register, make, spec

# Base class
# ----------------------------------------

register(
    id="InteractiveDrive-v0",
    entry_point="rdb.envs.drive2d.worlds.interactive:SingleInteractiveEnv",
)


# Drive world
from .worlds import week3, week4, week5, week6, week7, week8, week9, week10


def register_key_class(key, cls):
    if "_v1" in key:
        key = key.replace("_v1", "")
        register(
            id=f"{key}-v1",
            entry_point=f"{c.__module__}:{c.__name__}",
            max_episode_steps=10,
        )
    elif "_v0" in key:
        key = key.replace("_v0", "")
        register(
            id=f"{key}-v0",
            entry_point=f"{c.__module__}:{c.__name__}",
            max_episode_steps=10,
        )
    else:
        register(
            id=f"{key}-v0",
            entry_point=f"{c.__module__}:{c.__name__}",
            max_episode_steps=10,
        )


# Automatically register all WeekN_xx environmnts
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
for k, c in week7.__dict__.items():
    if "Week7_" in k:
        register_key_class(k, c)
for k, c in week8.__dict__.items():
    if "Week8_" in k:
        register_key_class(k, c)
for k, c in week9.__dict__.items():
    if "Week9_" in k:
        register_key_class(k, c)
for k, c in week10.__dict__.items():
    if "Week10_" in k:
        register_key_class(k, c)