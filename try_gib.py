import os
import shutil

import math

import numpy as np
import clip


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
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import torch
from collections import OrderedDict
import random
import omegaconf
import magnum as mn
from habitat.utils.geometry_utils import quaternion_rotate_vector
import matplotlib.patches as mpatches

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


def get_new_pallete(num_colors):
    """Generate a color pallete given the number of colors needed. First color is always black."""
    pallete = []
    for j in range(num_colors):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        pallete.append([r, g, b])
    return torch.tensor(pallete).float() / 255.0

def lseg(rgb_im):

    pwd = os.getcwd()
    path = '/home2/raghav.arora/lseg-minimal/'
    os.chdir(path)
    from lseg import LSegNet
    os.chdir(pwd)

    # Initialize the model
    net = LSegNet(
        backbone='clip_vitl16_384',
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation='relu',
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load('/home2/raghav.arora/lseg-minimal/examples/checkpoints/lseg_minimal_e200.ckpt'))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text
    # prompts = ["other"]  # begin with the catch-all "other" class
    label_classes = ['others', 'carpet', 'couch', 'plant', 'chair']

    # Cosine similarity module
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    with torch.no_grad():

        # Extract and normalize text features
        label_classes = ['others', 'carpet', 'couch', 'plant', 'chair']
        prompt = [clip.tokenize(lc).cuda() for lc in label_classes]
        text_feat_list = [clip_text_encoder(p) for p in prompt]
        text_feat_norm_list = [
            torch.nn.functional.normalize(tf) for tf in text_feat_list
        ]

        # Load the input image
        img = Image.fromarray(rgb_im)
        print(f"Original image shape: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img = torch.from_numpy(img).float() / 255.0
        img = img[..., :3]  # drop alpha channel, if present
        img = img.cuda()
        img = img.permute(2, 0, 1)  # C, H, W
        img = img.unsqueeze(0)  # 1, C, H, W
        print(f"Image shape: {img.shape}")

        # Extract per-pixel CLIP features (1, 512, H // 2, W // 2)
        img_feat = net.forward(img)
        # Normalize features (per-pixel unit vectors)
        img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
        print(f"Extracted CLIP image feat: {img_feat_norm.shape}")

        # Compute cosine similarity across image and prompt features
        similarities = []
        for _i in range(len(label_classes)):
            similarity = cosine_similarity(
                img_feat_norm, text_feat_norm_list[_i].unsqueeze(-1).unsqueeze(-1)
            )
            similarities.append(similarity)

        similarities = torch.stack(
            similarities, dim=0
        )  # num_classes, 1, H // 2, W // 2
        similarities = similarities.squeeze(1)  # num_classes, H // 2, W // 2
        similarities = similarities.unsqueeze(0)  # 1, num_classes, H // 2, W // 2
        class_scores = torch.max(similarities, 1)[1]  # 1, H // 2, W // 2
        class_scores = class_scores[0].detach()
        print(f"class scores: {class_scores.shape}")

        pallete = get_new_pallete(len(label_classes))

        disp_img = torch.zeros(240, 320, 3)
        for _i in range(len(label_classes)):
            disp_img[class_scores == _i] = pallete[_i]
        rawimg = cv2.imread('../images/136.png')
        rawimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)
        rawimg = cv2.resize(rawimg, (320, 240))
        rawimg = torch.from_numpy(rawimg).float() / 255.0
        rawimg = rawimg[..., :3]  # drop alpha channel, if present

        disp_img = 0.5 * disp_img + 0.5 * rawimg
        cv2.imwrite('try_save', disp_img)
        import pdb; pdb.set_trace()

        plt.imshow(disp_img.detach().cpu().numpy())
        plt.legend(
            handles=[
                mpatches.Patch(
                    color=(
                        pallete[i][0].item(),
                        pallete[i][1].item(),
                        pallete[i][2].item(),
                    ),
                    label=label_classes[i],
                )
                for i in range(len(label_classes))
            ]
        )
        plt.savefig('please5.png')



def shortest_path_example():
    config_path = 'habitat-lab/habitat/config/benchmark/rearrange/nav_to_obj.yaml'
    config = habitat.get_config(config_path=config_path)
    with habitat.config.read_write(config):
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {
                "third_rgb_sensor": ThirdRGBSensorConfig(height=512, width=512),
                "head_rgb_sensor": HeadRGBSensorConfig(height=512, width=512)
            }
        )
        config.habitat.dataset.data_path = 'data/datasets/replica_cad/rearrange/v1/val/rep_try.json.gz'
        ob_gl = {'type': 'ObjectToGoalDistance'}
        ob_gl = omegaconf.dictconfig.DictConfig(ob_gl)
        if "force_terminate" in config.habitat.task.measurements:
            config.habitat.task.measurements.force_terminate.max_accum_force = -1.0
            config.habitat.task.measurements.force_terminate.max_instant_force = (
                    -1.0
                )
    with SimpleRLEnv(config=config) as env:
        info = {}
        obj_to_gl_dist = 0
        done = False
        print("Environment creation successful")
        for episode in range(2):
            env.reset()
            info['num_steps'] = 0
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
                    "action": "base_velocity",
                    'action_args': {"base_vel": [1, 0]},
                }

                if info['num_steps'] == 5000:
                    random_action['action'] = 'REARRANGE_STOP'

                # agent_state = env.habitat_env.sim.get_agent_state()
                # current_position = agent_state.position
                # # Calculate the forward direction vector of the agent
                # forward_vector = quaternion_rotate_vector(agent_state.rotation, mn.Vector3(0, 0, 1))

                # next_position = current_position + forward_vector
                
                observations, reward, done, info = env.step(random_action)
                
                if not done and round(info['object_to_goal_distance']['0'],2) == round(obj_to_gl_dist,2):
                    # Agent cannot move forward
                    random_action = {
                        "action": "base_velocity",
                        'action_args': {"base_vel": [0.5, -0.5]},
                    }
                    observations, reward, done, info = env.step(random_action)
                obj_to_gl_dist = info['object_to_goal_distance']['0']
                
                try:
                    # import pdb; pdb.set_trace()
                    im1 = observations["robot_head_rgb"]
                    im2 = observations["robot_third_rgb"]
                except KeyError:
                    im = observations["rgb"]

                im3 = Image.fromarray(im1)
                draw = ImageDraw.Draw(im3)
                agent_position = env.habitat_env.sim.get_agent(0).state.sensor_states['robot_head_rgb'].position
                # agent_position = env.habitat_env.sim.get_agent_state().position
                # agent_position = env._env.sim.get_agent_state().position
                draw.text((0,0), str(agent_position), fill="white")

                lseg(im2)
                # im2.save(dirname+'/image_' + str(info['num_steps']) + '.png')

                # top_down_map = draw_top_down_map(info, im.shape[0])
                output_im = np.concatenate((im3, im2), axis=1)
                images.append(output_im)
            import pdb; pdb.set_trace()
            images_to_video(images, dirname, "trajectory")
            print("Episode finished")


def main():
    shortest_path_example()


if __name__ == "__main__":
    main()
