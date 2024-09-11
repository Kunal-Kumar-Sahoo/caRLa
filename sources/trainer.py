import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import pickle
import json
from dataclasses import dataclass
from threading import Thread

# Assuming these are defined elsewhere in your project
from sources import STOP, ACTIONS, ACTIONS_NAMES
import settings

class ARTDQNTrainer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self.create_model().to(self.device)

    def create_model(self, prediction=False):
        # Define your PyTorch model architecture here
        # This is a simple example and might need to be adjusted based on your specific requirements
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 512),  # Adjust these dimensions based on your input size
            nn.ReLU(),
            nn.Linear(512, len(ACTIONS))
        )
        return model

    def init2(self, stop, logdir, trainer_stats, episode, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, duration, optimizer, models, car_npcs):
        self.show_conv_cam = False
        self.target_model = self.create_model(prediction=True).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.replay_memory = deque(maxlen=settings.REPLAY_MEMORY_SIZE)
        
        self.logdir = logdir if logdir else f"logs/{settings.MODEL_NAME}-{int(time.time())}"
        self.tensorboard = SummaryWriter(log_dir=self.logdir)
        self.tensorboard_step = episode.value

        self.last_target_update = last_target_update
        self.last_log_episode = 0
        self.tps = 0
        self.last_checkpoint = 0
        self.save_model = False

        self.stop = stop
        self.trainer_stats = trainer_stats
        self.episode = episode
        self.epsilon = epsilon
        self.discount = discount
        self.update_target_every = update_target_every
        self.min_reward = min_reward
        self.agent_show_preview = agent_show_preview
        self.save_checkpoint_every = save_checkpoint_every
        self.seconds_per_episode = seconds_per_episode
        self.duration = duration
        self.optimizer = optimizer
        self.models = models
        self.car_npcs = car_npcs

        self.compile_model()

    def compile_model(self, lr=1e-3, decay=0):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < settings.MIN_REPLAY_MEMORY_SIZE:
            return False

        minibatch = random.sample(self.replay_memory, settings.MINIBATCH_SIZE)

        current_states = torch.FloatTensor([transition[0][0] for transition in minibatch]).to(self.device) / 255.0
        if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
            current_kmh = torch.FloatTensor([[transition[0][1]] for transition in minibatch]).to(self.device)
            current_kmh = (current_kmh - 50) / 50

        with torch.no_grad():
            current_qs_list = self.model(current_states)
            if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
                current_qs_list = torch.cat((current_qs_list, current_kmh), dim=1)

        new_current_states = torch.FloatTensor([transition[3][0] for transition in minibatch]).to(self.device) / 255.0
        if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
            new_current_kmh = torch.FloatTensor([[transition[3][1]] for transition in minibatch]).to(self.device)
            new_current_kmh = (new_current_kmh - 50) / 50

        with torch.no_grad():
            future_qs_list = self.target_model(new_current_states)
            if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
                future_qs_list = torch.cat((future_qs_list, new_current_kmh), dim=1)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + self.discount.value * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index].clone()
            current_qs[action] = new_q

            X.append(current_state[0])
            y.append(current_qs.cpu().numpy())

        log_this_step = False
        if self.tensorboard_step > self.last_log_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard_step

        X = torch.FloatTensor(X).to(self.device) / 255.0
        y = torch.FloatTensor(y).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(X)
        loss = nn.MSELoss()(predictions, y)
        loss.backward()
        self.optimizer.step()

        if log_this_step:
            self.tensorboard.add_scalar('Loss', loss.item(), self.tensorboard_step)

        if self.tensorboard_step >= self.last_target_update + self.update_target_every.value:
            self.target_model.load_state_dict(self.model.state_dict())
            self.last_target_update += self.update_target_every.value

        return True

    def get_lr_decay(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr'], param_group['weight_decay']

    def serialize_weights(self):
        return pickle.dumps(self.model.state_dict())

    def init_serialized_weights(self, weights, weights_iteration):
        self.weights = weights
        self.weights.raw = self.serialize_weights()
        self.weights_iteration = weights_iteration 

    def train_in_loop(self):
        self.tps_counter = deque(maxlen=20)

        while True:
            step_start = time.time()

            if self.stop.value == STOP.stopping:
                return

            if self.stop.value in [STOP.carla_simulator_error, STOP.restarting_carla_simulator]:
                self.trainer_stats[0] = TRAINER_STATE.paused
                time.sleep(1)
                continue

            if not self.train():
                self.trainer_stats[0] = TRAINER_STATE.waiting

                if self.stop.value in [STOP.at_checkpoint, STOP.now]:
                    self.stop.value = STOP.stopping

                time.sleep(0.01)
                continue

            self.trainer_stats[0] = TRAINER_STATE.training

            self.weights.raw = self.serialize_weights()
            with self.weights_iteration.get_lock():
                self.weights_iteration.value += 1

            frame_time = time.time() - step_start
            self.tps_counter.append(frame_time)
            self.trainer_stats[1] = len(self.tps_counter)/sum(self.tps_counter)

            save_model = self.save_model
            if save_model:
                torch.save(self.model.state_dict(), save_model)
                self.save_model = False

            checkpoint_number = self.episode.value // self.save_checkpoint_every.value

            if checkpoint_number > self.last_checkpoint or self.stop.value == STOP.now:
                self.models.append(f'checkpoint/{settings.MODEL_NAME}_{self.episode.value}.model')
                hparams = {
                    'duration': self.duration.value,
                    'episode': self.episode.value,
                    'epsilon': list(self.epsilon),
                    'discount': self.discount.value,
                    'update_target_every': self.update_target_every.value,
                    'last_target_update': self.last_target_update,
                    'min_reward': self.min_reward.value,
                    'agent_show_preview': [list(preview) for preview in self.agent_show_preview],
                    'save_checkpoint_every': self.save_checkpoint_every.value,
                    'seconds_per_episode': self.seconds_per_episode.value,
                    'model_path': f'checkpoint/{settings.MODEL_NAME}_{self.episode.value}.model',
                    'logdir': self.logdir,
                    'weights_iteration': self.weights_iteration.value,
                    'car_npcs': list(self.car_npcs),
                    'models': list(set(self.models))
                }

                torch.save(self.model.state_dict(), f'checkpoint/{settings.MODEL_NAME}_{hparams["episode"]}.model')

                with open('checkpoint/hparams_new.json', 'w', encoding='utf-8') as f:
                    json.dump(hparams, f)

                try:
                    os.remove('checkpoint/hparams.json')
                except:
                    pass
                try:
                    os.rename('checkpoint/hparams_new.json', 'checkpoint/hparams.json')
                    self.last_checkpoint = checkpoint_number
                except Exception as e:
                    print(str(e))

            if self.stop.value in [STOP.at_checkpoint, STOP.now]:
                self.stop.value = STOP.stopping

@dataclass
class TRAINER_STATE:
    starting = 0
    waiting = 1
    training = 2
    finished = 3
    paused = 4

TRAINER_STATE_MESSAGE = {
    0: 'STARTING',
    1: 'WAITING',
    2: 'TRAINING',
    3: 'FINISHED',
    4: 'PAUSED',
}

def check_weights_size(model_path, weights_size):
    trainer = ARTDQNTrainer(model_path)
    weights_size.value = len(trainer.serialize_weights())

# ... [Previous code remains the same]

def run(model_path, logdir, stop, weights, weights_iteration, episode, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, duration, transitions, tensorboard_stats, trainer_stats, episode_stats, optimizer, models, car_npcs, carla_settings_stats, carla_fps):
    if settings.TRAINER_GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.TRAINER_GPU)

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    trainer = ARTDQNTrainer(model_path)
    trainer.init2(stop, logdir, trainer_stats, episode, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, duration, optimizer, models, car_npcs)
    trainer.init_serialized_weights(weights, weights_iteration)

    trainer_stats[0] = TRAINER_STATE.waiting

    trainer_thread = Thread(target=trainer.train_in_loop, daemon=True)
    trainer_thread.start()

    raw_rewards = deque(maxlen=settings.AGENTS*10)
    weighted_rewards = deque(maxlen=settings.AGENTS*10)
    episode_times = deque(maxlen=settings.AGENTS*10)
    frame_times = deque(maxlen=settings.AGENTS*2)

    configured_actions = [getattr(ACTIONS, action) for action in settings.ACTIONS]

    while stop.value != 3:
        if episode.value > trainer.tensorboard_step:
            trainer.tensorboard_step = episode.value

        for _ in range(transitions.qsize()):
            try:
                trainer.update_replay_memory(transitions.get(True, 0.1))
            except:
                break

        while not tensorboard_stats.empty():
            agent_episode, reward, agent_epsilon, episode_time, frame_time, weighted_reward, *avg_predicted_qs = tensorboard_stats.get_nowait()

            raw_rewards.append(reward)
            weighted_rewards.append(weighted_reward)
            episode_times.append(episode_time)
            frame_times.append(frame_time)

            episode_stats[0] = min(raw_rewards)
            episode_stats[1] = sum(raw_rewards)/len(raw_rewards)
            episode_stats[2] = max(raw_rewards)
            episode_stats[3] = min(episode_times)
            episode_stats[4] = sum(episode_times)/len(episode_times)
            episode_stats[5] = max(episode_times)
            episode_stats[6] = sum(frame_times)/len(frame_times)
            episode_stats[7] = min(weighted_rewards)
            episode_stats[8] = sum(weighted_rewards)/len(weighted_rewards)
            episode_stats[9] = max(weighted_rewards)
            tensorboard_q_stats = {}
            for action, (avg_predicted_q, std_predicted_q, usage_predicted_q) in enumerate(zip(avg_predicted_qs[0::3], avg_predicted_qs[1::3], avg_predicted_qs[2::3])):
                if avg_predicted_q != -10**6:
                    episode_stats[action*3 + 10] = avg_predicted_q
                    tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_avg' if action else f'q_all_actions_avg'] = avg_predicted_q
                if std_predicted_q != -10 ** 6:
                    episode_stats[action*3 + 11] = std_predicted_q
                    tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_std' if action else f'q_all_actions_std'] = std_predicted_q
                if usage_predicted_q != -10 ** 6:
                    episode_stats[action*3 + 12] = usage_predicted_q
                    if action > 0:
                        tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_usage_pct'] = usage_predicted_q
            
            carla_stats = {}
            for process_no in range(settings.CARLA_HOSTS_NO):
                for index, stat in enumerate(['carla_{}_car_npcs', 'carla_{}_weather_sun_azimuth', 'carla_{}_weather_sun_altitude', 'carla_{}_weather_clouds_pct', 'carla_{}_weather_wind_pct', 'carla_{}_weather_rain_pct']):
                    if carla_settings_stats[process_no][index] != -1:
                        carla_stats[stat.format(process_no+1)] = carla_settings_stats[process_no][index]
                carla_stats[f'carla_{process_no + 1}_fps'] = carla_fps[process_no].value

            trainer.tensorboard.add_scalar('reward_raw_avg', episode_stats[1], agent_episode)
            trainer.tensorboard.add_scalar('reward_raw_min', episode_stats[0], agent_episode)
            trainer.tensorboard.add_scalar('reward_raw_max', episode_stats[2], agent_episode)
            trainer.tensorboard.add_scalar('reward_weighted_avg', episode_stats[8], agent_episode)
            trainer.tensorboard.add_scalar('reward_weighted_min', episode_stats[7], agent_episode)
            trainer.tensorboard.add_scalar('reward_weighted_max', episode_stats[9], agent_episode)
            trainer.tensorboard.add_scalar('epsilon', agent_epsilon, agent_episode)
            trainer.tensorboard.add_scalar('episode_time_avg', episode_stats[4], agent_episode)
            trainer.tensorboard.add_scalar('episode_time_min', episode_stats[3], agent_episode)
            trainer.tensorboard.add_scalar('episode_time_max', episode_stats[5], agent_episode)
            trainer.tensorboard.add_scalar('agent_fps_avg', episode_stats[6], agent_episode)
            trainer.tensorboard.add_scalar('optimizer_lr', optimizer[0], agent_episode)
            trainer.tensorboard.add_scalar('optimizer_decay', optimizer[1], agent_episode)

            for key, value in tensorboard_q_stats.items():
                trainer.tensorboard.add_scalar(key, value, agent_episode)

            for key, value in carla_stats.items():
                trainer.tensorboard.add_scalar(key, value, agent_episode)

            if episode_stats[7] >= min_reward.value:
                trainer.save_model = f'models/{settings.MODEL_NAME}__{episode_stats[2]:_>7.2f}max_{episode_stats[1]:_>7.2f}avg_{episode_stats[0]:_>7.2f}min__{int(time.time())}.model'

        time.sleep(0.01)

    trainer_thread.join()
    trainer_stats[0] = TRAINER_STATE.finished
