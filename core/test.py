import os

import torch

from .mcts import MCTS, Node
from .utils import select_action
import multiprocessing

def _test(config, model, ep_i, device, render, save_video, save_path, ep_data):
    with torch.no_grad():
        env = config.new_game(save_video=save_video, save_path=save_path,
                              episode_trigger=lambda episode_id: True, uid=ep_i)
        done = False
        ep_reward = 0
        obs = env.reset()
        while not done:
            if render:
                env.render()
            root = Node(0)
            obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
            root.expand(env.to_play(), env.legal_actions(), model_initial_inference(obs))
            MCTS(config).run(root, env.action_history(), model)
            action, _ = select_action(root, temperature=1, deterministic=True)
            obs, reward, done, info = env.step(action.index)
            ep_reward += reward
        env.close()
    ep_data[ep_i] = ep_reward


def test(config, model, episodes, device, render, save_video=False):
    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recording')

    manager = multiprocessing.Manager()
    ep_data = manager.dict()
    jobs = []
    for ep_i in range(episodes):
        p = multiprocessing.Process(target=_test, args=(config, model, ep_i, device, render, save_vide, save_path, 
                                                        ep_data))
        
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc_join()
        
    test_reward = sum(ep_data.values())

    return test_reward  / episodes