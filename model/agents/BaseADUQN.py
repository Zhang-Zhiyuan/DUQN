import time
import copy
import torch
import argparse
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
from model.R_RSSM_UserResponseModel import R_RSSM_UserResponseModel
from reader.RL4RSDataReader import RL4RSDataReader
from tqdm import tqdm


user_model_path = "output/r_rssm_urm/r_rssm_user_response_model"

class BaseADUQN(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - episode_batch_size
        - batch_size
        - actor_lr
        - critic_lr
        - actor_decay
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - n_iter
            - train_every_n_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - with_eval
            - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--episode_batch_size', type=int, default=8, 
                            help='episode sample batch size')
        parser.add_argument('--batch_size', type=int, default=32, 
                            help='training batch size')
        parser.add_argument('--actor_lr', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        return parser
    
    
    def __init__(self, args, facade):
        '''
        self.gamma
        self.n_iter
        self.check_episode
        self.with_eval
        self.save_path
        self.facade
        self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay
        
        self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        self.critic = facade.critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)


        self.tau = args.target_mitigate_coef
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        
        args.max_seq_len = 50
        args.n_worker = 4
        args.item_meta_file = "dataset/rl4rs/item_info.csv"
        args.meta_data_separator = "@"
        args.feature_dim = 16
        args.attn_n_head = 2
        args.hidden_dims = [256]
        args.dropout_rate = 0.2
        args.device = "cuda:0"
        args.model_path = user_model_path
        args.loss = 'bce'
        args.l2_coef = 0.001

        self.user_response_model = R_RSSM_UserResponseModel(args=args,device="cuda:0")
        self.user_response_model.load_from_checkpoint(args.model_path, with_optimizer = False)
        self.user_response_model.to(args.device)
        for param in self.user_response_model.parameters():
            param.requires_grad = False
        
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        self.facade.initialize_train() # buffer setup
        prepare_step = 0
        # random explore before training
        initial_epsilon = 1.0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        while not self.facade.is_training_available:
            observation = self.run_episode_step(0, initial_epsilon, observation, True)
            prepare_step += 1
        # training records
        self.training_history = {"critic_loss": [], "actor_loss": []}
        
        print(f"Total {prepare_step} prepare steps")
        
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation
            

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
        reward = torch.FloatTensor(reward)
        done_mask = torch.FloatTensor(done_mask)
        
        critic_loss, actor_loss = self.get_ddpg_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1])}
    
    def get_ddpg_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):

        # user feedback
        with torch.no_grad():
            current_policy_output = self.facade.apply_policy(observation, self.actor)
        
        exposure_features = current_policy_output['action_features']
        batch_data = {
            'user_profile': observation['user_profile'],
            'history_features': observation['history_features'],
            'exposure_features': exposure_features #有错
        }
        with torch.no_grad():
            output_dict = self.user_response_model(batch_data)
            response = torch.bernoulli(output_dict['probs']) # (B, slate_size)
            probs_under_temper = output_dict['probs'] # * prob_scale
            user_response = torch.bernoulli(probs_under_temper).detach() # (B, slate_size)       
            
        # user_response = torch.FloatTensor(user_response)
        # Get current Q estimate
        current_critic_output = self.facade.apply_critic(observation, 
                                                         utils.wrap_batch(policy_output, device = self.device),
                                                         user_response, 
                                                         self.critic)
        current_Q = current_critic_output['q']

        # shape = user_response.size() 
        # with open('tensor_shape.txt', 'w') as file:
        #     file.write(str(shape))


        # Compute the target Q value
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)

        next_exposure_features = next_policy_output['action_features']
        next_batch_data = {
            'user_profile': next_observation['user_profile'],
            'history_features': next_observation['history_features'],
            'exposure_features': next_exposure_features #有错
        }     

        with torch.no_grad():
            next_output_dict = self.user_response_model(next_batch_data)
            next_response = torch.bernoulli(next_output_dict['probs']) # (B, slate_size)
            next_probs_under_temper = next_output_dict['probs'] # * prob_scale
            next_user_response = torch.bernoulli(next_probs_under_temper).detach() # (B, slate_size)       
        
        # next_user_response = torch.FloatTensor(next_user_response)        
        target_critic_output = self.facade.apply_critic(next_observation, 
                                                        next_policy_output, 
                                                        next_user_response,
                                                        self.critic_target)
        target_Q = target_critic_output['q']
        target_Q = reward + self.gamma * (done_mask * target_Q).detach()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        
        # Regularization loss
#         critic_reg = current_critic_output['reg']

        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Compute actor loss
        policy_output = self.facade.apply_policy(observation, self.actor)
        critic_output = self.facade.apply_critic(observation, 
                                                 policy_output, 
                                                 user_response,
                                                 self.critic)
        actor_loss = -critic_output['q'].mean()
        
        # Regularization loss
#         actor_reg = policy_output['reg']

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
        return critic_loss, actor_loss


    def save(self):
        torch.save(self.critic.state_dict(), self.save_path + "_critic")
        torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

    # def test(self):
    #     with torch.no_grad():

    #         self.load()
            
    #         t = time.time()
            
    #         print("Run procedures before testing")
    #         self.action_before_test()
    #         t = time.time()
    #         start_time = t
    #         # testing
    #         print("Testing:")
    #         observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
    #         step_offset = sum(self.n_iter[:-1])
    #         for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
    #             observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation, True)
    #             if i % self.train_every_n_step == 0:
    #                 self.step_test()
    #             if i % self.check_episode == 0:
    #                 t_ = time.time()
    #                 print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
    #                 print(self.log_iteration(i))
    #                 t = t_
    #         self.action_after_test()   

    # def action_before_test(self):
    #     '''
    #     Action before testing:
    #     - facade setup:
    #         - buffer setup
    #     - run random episodes to build-up the initial buffer
    #     '''
    #     self.facade.initialize_test() # buffer setup
    #     prepare_step = 0
    #     # random explore before training
    #     initial_epsilon = 1.0
    #     observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
    #     while not self.facade.is_training_available:
    #         observation = self.run_episode_step(0, initial_epsilon, observation, True)
    #         prepare_step += 1
    #     # training records
    #     self.training_history = {"critic_loss": [], "actor_loss": []}
        
    #     print(f"Total {prepare_step} prepare steps")
    
    # def action_after_test(self):
    #     self.facade.stop_env()

    # def step_test(self):
    #     observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
    #     reward = torch.FloatTensor(reward)
    #     done_mask = torch.FloatTensor(done_mask)
        
    #     critic_loss, actor_loss = self.get_ddpg_loss(observation, policy_output, reward, done_mask, next_observation)

    #     # Update the frozen target models
    #     for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
    #         target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    #     for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
    #         target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
