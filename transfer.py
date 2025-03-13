import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub

from pipoli.core import Dimension, Context, DimensionalPolicy, ScaledPolicy
from pipoli.sources.sb3 import SB3Policy

from make_cheetah import make_cheetah_xml


## Start of original context definition

base_dimensions = [
    M := Dimension([1, 0, 0]),
    L := Dimension([0, 1, 0]),
    T := Dimension([0, 0, 1]),
]
Unit = Dimension([0, 0, 0])

original_context = Context(
    base_dimensions,
    *zip(
        ("m", M, 14),
        ("g", L/T**2, 9.81),
        ("taumax", M*L**2/T**2, 1),
        ("L", L, 0.5),
        ("Lh", L, 0.15),
        ("l0", L, 0.145),
        ("l1", L, 0.15),
        ("l2", L, 0.094),
        ("l3", L, 0.133),
        ("l4", L, 0.106),
        ("l5", L, 0.07),
        ("k0", M*L**2/T**2, 240),
        ("k1", M*L**2/T**2, 180),
        ("k2", M*L**2/T**2, 120),
        ("k3", M*L**2/T**2, 180),
        ("k4", M*L**2/T**2, 120),
        ("k5", M*L**2/T**2, 60),
        ("b0", M*L**2/T, 6),
        ("b1", M*L**2/T, 4.5),
        ("b2", M*L**2/T, 3),
        ("b3", M*L**2/T, 4.5),
        ("b4", M*L**2/T, 3),
        ("b5", M*L**2/T, 1.5),
    )
)

cheetah_file = make_cheetah_xml(original_context, "original")


## Creation of original dimensionnal policy

halfcheetah_v5_sac_expert =  load_from_hub(
    repo_id="farama-minari/HalfCheetah-v5-SAC-expert",
    filename="halfcheetah-v5-sac-expert.zip",
)
model = SAC.load(halfcheetah_v5_sac_expert)

sb3_policy = SB3Policy(
    model,
    model_obs_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float64),
    model_act_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
    predict_kwargs=dict(deterministic=True)
)

original_policy = DimensionalPolicy(
    sb3_policy,
    original_context,
    obs_dims=[L] + [Unit] * 7 + [L/T] * 2 + [1/T] * 7,
    act_dims=[M*L**2/T**2] * 6
)

## Test scaled policy in a new scaled context

base = ["m", "L", "g"]

new_context = original_context.scale_to(base, [14 * 2, 0.5, 9.81])

print("adimensional distance", new_context.adimensional_distance(original_context, base))
print("euclidian distance", new_context.euclidian_distance(original_context))
print("cosine similarity", new_context.cosine_similarity(original_context))

new_cheetah_file = make_cheetah_xml(new_context, "new")

new_policy = original_policy.to_scaled(new_context, base)

env = gym.make("HalfCheetah-v5", xml_file=new_cheetah_file, render_mode="human", frame_skip=2)
obs, _ = env.reset()

for _ in range(1000):
    act = new_policy.action(obs)
    obs, _, _, trunc, _ = env.step(act)
    if trunc:
        obs, _ = env.reset()

env.close()