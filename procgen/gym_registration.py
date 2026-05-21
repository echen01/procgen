import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gym3 import ViewerWrapper, ExtractDictObWrapper
from gym3.types import Discrete, Real, TensorType, ValType
from gym3.interop import multimap
from gym3 import types_np
from .env import ENV_NAMES, ProcgenGym3Env


def _vt2space(vt: ValType):
    from gymnasium import spaces

    def tt2space(tt: TensorType):
        if isinstance(tt.eltype, Discrete):
            if tt.ndim == 0:
                return spaces.Discrete(tt.eltype.n)
            else:
                return spaces.Box(
                    low=0,
                    high=tt.eltype.n - 1,
                    shape=tt.shape,
                    dtype=types_np.dtype(tt),
                )
        elif isinstance(tt.eltype, Real):
            return spaces.Box(
                shape=tt.shape,
                dtype=types_np.dtype(tt),
                low=float("-inf"),
                high=float("inf"),
            )
        else:
            raise NotImplementedError

    space = multimap(tt2space, vt)

    def dict2dict_space(d):
        if isinstance(d, dict):
            return spaces.Dict({k: dict2dict_space(v) for k, v in d.items()})
        else:
            return d

    return dict2dict_space(space)


class ToGymnasiumEnv(gym.Env):
    """
    Create a gymnasium environment from a gym3 environment.
    """

    def __init__(self, env, render_mode=None):
        self.env = env
        assert env.num == 1
        self.observation_space = _vt2space(env.ob_space)
        self.action_space = _vt2space(env.ac_space)
        self.metadata = {"render_modes": ["human", "rgb_array"]}
        self.render_mode = render_mode
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _rew, ob, first = self.env.observe()
        if not first[0]:
            print("Warning: early reset ignored")
        obs = multimap(lambda x: x[0], ob)
        info = self.env.get_info()[0]
        return obs, info

    def step(self, ac):
        _, prev_ob, _ = self.env.observe()
        self.env.act(np.array([ac]))
        rew, ob, first = self.env.observe()
        if first[0]:
            ob = prev_ob
        obs = multimap(lambda x: x[0], ob)
        terminated = bool(first[0])
        truncated = False
        info = self.env.get_info()[0]
        return obs, float(rew[0]), terminated, truncated, info

    def render(self):
        info = self.env.get_info()[0]
        if "rgb" in info:
            return info["rgb"]
        return None

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self


def make_env(render_mode=None, render=False, **kwargs):
    # the render option is kept here for backwards compatibility
    # users should use `render_mode="human"` or `render_mode="rgb_array"`
    if render:
        render_mode = "human"

    use_viewer_wrapper = False
    kwargs["render_mode"] = render_mode
    if render_mode == "human":
        # procgen does not directly support rendering a window
        # instead it's handled by gym3's ViewerWrapper
        # procgen only supports a render_mode of "rgb_array"
        use_viewer_wrapper = True
        kwargs["render_mode"] = "rgb_array"

    env = ProcgenGym3Env(num=1, num_threads=0, **kwargs)
    env = ExtractDictObWrapper(env, key="rgb")
    if use_viewer_wrapper:
        env = ViewerWrapper(env, tps=15, info_key="rgb")
    gym_env = ToGymnasiumEnv(env, render_mode=render_mode)
    return gym_env


def register_environments():
    for env_name in ENV_NAMES:
        register(
            id=f'procgen-{env_name}-v0',
            entry_point='procgen.gym_registration:make_env',
            kwargs={"env_name": env_name},
        )