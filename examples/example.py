#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym

import habitat
import habitat.gym  # noqa: F401
from habitat.utils.visualizations.utils import images_to_video


def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README
    config_path = 'benchmark/rearrange/skills/pick.yaml'
    config = habitat.get_config(
      config_path,
      overrides=["habitat.environment.max_episode_steps=20"]
    )
    # env = habitat.Env(config)
    env = habitat.gym.make_gym_from_config(config)
    print("Environment creation successful")
    observations = env.reset()  # noqa: F841

    print("Agent acting inside environment.")
    count_steps = 0
    terminal = False
    images = []
    print("Agent stepping around inside environment.")
    while not terminal:
        observations, reward, terminal, info = env.step(
            env.action_space.sample()
        )  # noqa: F841
        count_steps += 1
        
        print(observations.keys())
        im = observations["head_rgb"]
        images.append(im)

        print(info)
        images_to_video(images, 'raghav', "trajectory")
        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
