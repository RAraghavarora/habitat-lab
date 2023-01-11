import os
import shutil

import math

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
# from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.config.default import get_agent_config
from habitat.utils.visualizations.utils import images_to_video
from habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
    HeadRGBSensorConfig
)
from collections import OrderedDict
import random
import magnum as mn
from habitat.utils.geometry_utils import quaternion_rotate_vector

cv2 = try_cv2_import()
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, size):
    return maps.colorize_draw_agent_and_fit_to_height(info['top_down_map'],size)


def shortest_path_example():
    config_path = 'habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_gibson.yaml'
    config = habitat.get_config(config_path=config_path)
    with habitat.config.read_write(config):
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {
                "third_rgb_sensor": ThirdRGBSensorConfig(height=512, width=512),
                "head_rgb_sensor": HeadRGBSensorConfig(height=512, width=512)
            }
        )
        config.habitat.dataset.data_path="data/datasets/gibson/val/val.json.gz"

        # config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    with SimpleRLEnv(config=config) as env:
        goal_radius = 0.1
        # goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE

        obj_to_gl_dist = 0
        print("Environment creation successful")
        for episode in range(2):
            env.reset()
            num_steps = 0
            dirname = os.path.join(
                IMAGE_DIR, "shortest_path_example", "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            print("Agent stepping around inside environment.")
            images = []
            step_count = 0
            while not env.habitat_env.episode_over:
                # try:
                #     pos = env.habitat_env.current_episode.goals[0].position
                # except:
                #     pos = [-5.307608604431152, 0.2628794312477112, 17.28000259399414]
                random_action = {
                    "action": "turn_right",
                    'action_args': None,
                }
                if num_steps == 500:
                    random_action['action'] = 'STOP'
                num_steps+=1

                # agent_state = env.habitat_env.sim.get_agent_state()
                # current_position = agent_state.position
                # # Calculate the forward direction vector of the agent
                # forward_vector = quaternion_rotate_vector(agent_state.rotation, mn.Vector3(0, 0, 1))

                # next_position = current_position + forward_vector
                
                observations, reward, done, info = env.step(random_action)
                # if not done and info['object_to_goal_distance']['0'] == obj_to_gl_dist:
                #     # Agent cannot move forward
                #     random_action = {
                #         "action": "turn_left",
                #         'action_args': None,
                #     }
                #     observations, reward, done, info = env.step(random_action)
                # obj_to_gl_dist = info['object_to_goal_distance']['0']
                
                try:
                    # import pdb; pdb.set_trace()
                    im1 = observations["rgb"]
                    im2 = observations["robot_third_rgb"]
                except KeyError:
                    im = observations["rgb"]

                from PIL import Image
                im2 = Image.fromarray(im1)
                # im2 = im2.rotate(90)

                # im2.save(dirname+'/image_' + str(info['num_steps']) + '.png')

                # top_down_map = draw_top_down_map(info, im.shape[0])
                # output_im = np.concatenate((im1, im2), axis=1)
                images.append(np.array(im2))
            images_to_video(images, dirname, "trajectory")
            print("Episode finished")


def main():
    shortest_path_example()


if __name__ == "__main__":
    main()
