from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from sb3_contrib import TQC
from huggingface_sb3 import load_from_hub

from pipoli.core import DimensionalPolicy, Dimension, Context
from pipoli.sources.sb3 import SB3Policy

from make_cheetah import make_cheetah, make_cheetah_xml


BASE_DIMENSIONS = [
    M := Dimension([1, 0, 0]),
    L := Dimension([0, 1, 0]),
    T := Dimension([0, 0, 1]),
]
Unit = Dimension([0, 0, 0])


def load_original_policy():
    halfcheetah_v5_tqc_expert =  load_from_hub(
        repo_id="farama-minari/HalfCheetah-v5-TQC-expert",
        filename="halfcheetah-v5-TQC-expert.zip",
    )
    model = TQC.load(halfcheetah_v5_tqc_expert, device="cpu")

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

    return original_policy


def evaluate_policy(context, xml_file, base, nb_episodes):
    original_policy = load_original_policy()
    policy = original_policy.to_scaled(context, base)

    nb_steps = 1000
    env = gym.make("HalfCheetah-v5", xml_file=xml_file)

    observations = np.zeros((nb_episodes, nb_steps, 17))
    actions = np.zeros((nb_episodes, nb_steps, 6))
    rewards = np.zeros((nb_episodes, nb_steps))
    infos = np.full((nb_episodes, nb_steps), None)

    for ep in range(nb_episodes):
        print("ep", ep)
        trunc = False
        step = 0
    
        obs, info = env.reset()

        while not trunc:
            act = policy.action(obs)

            observations[ep, step] = obs
            actions[ep, step] = act

            obs, rew, _, trunc, info = env.step(act)

            rewards[ep, step] = rew
            infos[ep, step] = info

            step += 1

    env.close()

    return observations, actions, rewards, infos


def process_context(context, base, nb_episodes, xml_dir):
    b1 = context.value(base[0])
    b2 = context.value(base[1])
    b3 = context.value(base[2])
    index = f"cheetah-{base[0]}-{base[1]}-{base[2]}_{b1:.3e}_{b2:.3e}_{b3:.3e}"

    xml = make_cheetah(context)
    xml_file = Path(xml_dir) / (index + ".xml")
    xml_file.write_text(xml)

    evaluation = evaluate_policy(context, str(xml_file.absolute()), base, nb_episodes)
    
    return index, (context, xml, b1, b2, b3) + evaluation


if __name__ == "__main__":
    ROOT = Path() / "output"
    XML_FILES = ROOT / "xml_files"
    DATA = ROOT / "data"

    #
    # Data generation parameters
    #
    base = ["m", "L", "g"]

    space = "log"
    b = 10
    range_1 = (-1, 1)
    range_2 = (-1, 1)
    range_3 = (0, 0)
    num_1 = 5
    num_2 = 5
    num_3 = 1

    nb_eval_episodes = 10

    #
    # Other metadata
    #
    observations_shape = "(nb_episodes, nb_steps, 17)"
    actions_shape = "(nb_episodes, nb_steps, 6)"
    rewards_shape = "(nb_episodes, nb_steps)"
    infos_shape = "(nb_episodes, nb_steps)"

    policy_info = {
        "repo_id": "farama-minari/HalfCheetah-v5-TQC-expert",
        "filename": "halfcheetah-v5-TQC-expert.zip",
        "commit": "995505a"
    }
    env_id = "HalfCheetah-v5"
    comment = "the env is default, except for xml_file"

    #
    # Original context instanciation
    #
    original_context = Context(
        BASE_DIMENSIONS,
        *zip(
            ("dt", T, 0.01),
            ("m", M, 14),
            ("g", L/T**2, 9.81),
            ("taumax", M*L**2/T**2, 1),
            ("d", L, 0.046),
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
    original_cheetah_file = make_cheetah_xml(original_context, "original", outdir=XML_FILES)
    original_cheetah_xml = Path(original_cheetah_file).read_text()

    #
    # Make all contexts
    #
    b1s = np.logspace(*range_1, base=b, num=num_1) * original_context.value(base[0])
    b2s = np.logspace(*range_2, base=b, num=num_2) * original_context.value(base[1])
    b3s = np.logspace(*range_3, base=b, num=num_3) * original_context.value(base[2])

    all_contexts = []

    for b1 in b1s:
        for b2 in b2s:
            for b3 in b3s:
                context = original_context.scale_to(base, [b1, b2, b3])
                all_contexts.append(context)
    
    #
    # Evaluation of transfer on all contexts
    #
    df = pd.DataFrame(columns=["context", "xml", "b1", "b2", "b3", "observations", "actions", "rewards", "infos"])
    df.attrs["base"] = base
    df.attrs["space"] = space
    df.attrs["b"] = b
    df.attrs["range_1"] = range_1
    df.attrs["range_2"] = range_2
    df.attrs["range_3"] = range_3
    df.attrs["num_1"] = num_1
    df.attrs["num_2"] = num_2
    df.attrs["num_3"] = num_3
    df.attrs["nb_eval_episodes"] = nb_eval_episodes
    df.attrs["observations_shape"] = observations_shape
    df.attrs["actions_shape"] = actions_shape
    df.attrs["rewards_shape"] = rewards_shape
    df.attrs["infos_shape"] = infos_shape
    df.attrs["policy_info"] = policy_info
    df.attrs["env"] = env_id
    df.attrs["comment"] = comment

    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    import time

    # start = time.time()
    with ProcessPoolExecutor(1) as executor:
        pbar = tqdm(total=len(all_contexts))

        def worker(c):
            return process_context(c, base, nb_eval_episodes, XML_FILES)

        print("Processing...")
        for index, data in executor.map(worker, all_contexts):
            df.loc[index] = data
            pbar.update()

        pbar.close()
    # stop = time.time()
    # print(stop-start, "s")
    
    memory = df.memory_usage(deep=True).sum()
    print(f"Pickling {memory / 1e9:.3f} GB of data...")
    df.to_pickle(DATA / f"data-similar-{str(base)[1:-1].replace(', ', '-')}-{range_1}-{range_2}-{range_3}-{num_1}-{num_2}-{num_3}-{b}.pkl.gz")
    