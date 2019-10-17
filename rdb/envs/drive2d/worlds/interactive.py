from rdb.envs.drive2d.worlds.highway import HighwayDriveWorld
from rdb.envs.drive2d.core.car import UserControlCar

"""
For manually driving & collecting user data
"""


class SingleInteractiveEnv(HighwayDriveWorld):
    def __init__(self):
        super().__init__(num_lanes=3)
        self._cars = []
        self._user_car = UserControlCar
        self._action_set = [ord(k) for k in "wasd"]

    @property
    def cars(self):
        return self._cars

    def step(self, action):
        obs = None
        rew = 0.0
        done = False
        info = {}
        return obs, rew, done, info

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            "FORWARD": ord("w"),
            "BACKWARD": ord("s"),
            "LEFT": ord("a"),
            "RIGHT": ord("d"),
        }
        keys_to_action = {}
        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))
            assert keys not in keys_to_action
            keys_to_action[keys] = action_id
        return keys_to_action


ACTION_MEANING = {
    ord("w"): "FORWARD",
    ord("s"): "BACKWARD",
    ord("a"): "LEFT",
    ord("d"): "RIGHT",
}
