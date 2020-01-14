"""An environment wrapper to convert binary to discrete action space."""
import gym
from gym import Env
from gym import Wrapper


class CustomJoypad(Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    # a mapping of buttons to binary values
    _button_map = {
        'right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'start':  0b00001000,
        'select': 0b00000100,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOOP':   0b00000000,
    }

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: Env):
        super().__init__(env)
       
    def step(self, button_list):
        byte_action = 0
        for button in button_list:
                byte_action |= self._button_map[button]
        return self.env.step(byte_action)

    def reset(self):
        """Reset the environment and return the initial observation."""
        return self.env.reset()

# explicitly define the outward facing API of this module
__all__ = [CustomJoypad.__name__]
