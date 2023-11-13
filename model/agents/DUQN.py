import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import utils
from model.agents.BaseADUQN import BaseADUQN
# from model.agents.BehaviorDDPG import BehaviorDDPG
    
class DUQN(BaseADUQN):
    @staticmethod
    def parse_model_args(parser):
        '''
        - args from DUQNDDPG:
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
        parser = BaseADUQN.parse_model_args(parser)
        parser.add_argument('--behavior_lr', type=float, default=0.0001, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--hyper_actor_coef', type=float, default=0.1, 
                            help='hyper actor loss coefficient')
        return parser
    
    
    def __init__(self, args, facade):
        '''
        from DDPG:
            self.episode_batch_size
            self.batch_size
            self.actor_lr
            self.critic_lr
            self.actor_decay
            self.critic_decay
            self.actor
            self.actor_target
            self.actor_optimizer
            self.critic
            self.critic_target
            self.critic_optimizer
            self.tau
            from BaseRLAgent:
                self.gamma
                self.n_iter
                slef.train_every_n_step
                self.check_episode
                self.with_eval
                self.save_path
                self.facade
                self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.hyper_actor_coef = args.hyper_actor_coef
        self.actor_behavior_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.behavior_lr, 
                                                         weight_decay=args.behavior_decay)

                                    
    def action_before_train(self):
        super().action_before_train()
        self.training_history["hyper_actor_loss"] = []
        self.training_history["behavior_loss"] = []
        
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
#         reward = torch.FloatTensor(reward)
#         done_mask = torch.FloatTensor(done_mask)
        
        critic_loss, actor_loss, hyper_actor_loss = self.get_hac_loss(observation, policy_output, reward, done_mask, next_observation)
        behavior_loss = self.get_behavior_loss(observation, policy_output, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['hyper_actor_loss'].append(hyper_actor_loss.item())
        self.training_history['behavior_loss'].append(behavior_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['hyper_actor_loss'][-1], 
                              self.training_history['behavior_loss'][-1])}
    
    def get_hac_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        
        # critic loss

        # user feedback
        # Get current Q estimate

        hyper_output = self.facade.infer_hyper_action(observation, policy_output, self.actor)
        
        exposure_features = hyper_output['action_features']
        batch_data = {
            'user_profile': observation['user_profile'],
            'history_features': observation['history_features'],
            'exposure_features': exposure_features 
        }
        with torch.no_grad():
            output_dict = self.user_response_model(batch_data)
            response = torch.bernoulli(output_dict['probs']) # (B, slate_size)
            probs_under_temper = output_dict['probs'] # * prob_scale
            user_response = torch.bernoulli(probs_under_temper).detach() # (B, slate_size)       

        # reward = self.cal_reward(user_response)

        current_critic_output = self.facade.apply_critic(observation, hyper_output, user_response, self.critic)
        current_Q = current_critic_output['q']


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
        

        target_critic_output = self.facade.apply_critic(next_observation, next_policy_output, next_user_response, self.critic_target)
        target_Q = target_critic_output['q']
        target_Q = reward + self.gamma * (done_mask * target_Q).detach()
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # actor loss
        
        # Compute actor loss
        if do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            policy_output = self.facade.apply_policy(observation, self.actor)
            critic_output = self.facade.apply_critic(observation, policy_output, user_response, self.critic)
            actor_loss = -critic_output['q'].mean()
            # Optimize the actor 
            actor_loss.backward()
            self.actor_optimizer.step()
            
        # hyper actor loss
        
        if do_actor_update and self.hyper_actor_coef > 0:
            self.actor_optimizer.zero_grad()
            policy_output = self.facade.apply_policy(observation, self.actor)
            inferred_hyper_output = self.facade.infer_hyper_action(observation, policy_output, self.actor)
            hyper_actor_loss = self.hyper_actor_coef * F.mse_loss(inferred_hyper_output['Z'], policy_output['Z']).mean()
            # Optimize the actor 
            hyper_actor_loss.backward()
            self.actor_optimizer.step()
            
        return critic_loss, actor_loss, hyper_actor_loss

    def get_behavior_loss(self, observation, policy_output, next_observation, do_update = True):
        observation, exposure, feedback = self.facade.extract_behavior_data(observation, policy_output, next_observation)
        observation['candidate_ids'] = exposure['ids']
        observation['candidate_features'] = exposure['features']
        policy_output = self.facade.apply_policy(observation, self.actor, do_softmax = False)
        action_prob = torch.sigmoid(policy_output['candidate_prob'])
        behavior_loss = F.binary_cross_entropy(action_prob, feedback)
        
        if do_update and self.behavior_lr > 0:
            self.actor_behavior_optimizer.zero_grad()
            behavior_loss.backward()
            self.actor_behavior_optimizer.step()
        return behavior_loss
    
    
    # def cal_reward(self, user_reponse):
    #     # reward (B,)
    #     immediate_reward = self.mean_with_cost(user_reponse).detach()
    #     immediate_reward = -torch.abs(immediate_reward - self.temper_sweet_point) + 1
    #     return immediate_reward

    # def mean_with_cost(self, feedback, zero_reward_cost = 0.1):
    #     '''
    #     @input:
    #     - feedback: (B, L)
    #     @output:
    #     - reward: (B,)
    #     '''
    #     B,L = feedback.shape
    #     cost = torch.zeros_like(feedback)
    #     cost[feedback == 0] = -zero_reward_cost
    #     reward = torch.mean(feedback + cost, dim = -1)
    #     return reward
    
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

    def test(self):
        with torch.no_grad():
            
            self.load()
            
            t = time.time()
            
            print("Run procedures before testing")
            self.action_before_test()
            t = time.time()
            start_time = t
            # testing
            print("Testing:")
            observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
            step_offset = sum(self.n_iter[:-1])
            for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
                observation = self.run_episode_step_test(i, self.exploration_scheduler.value(i), observation, True)
                # if i % self.train_every_n_step == 0:
                #     self.step_test()
                if i % self.check_episode == 0:
                    t_ = time.time()
                    print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
                    print(self.log_iteration(i))
                    t = t_
            self.action_after_test()   

    def action_before_test(self):
        '''
        Action before testing:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        self.facade.initialize_test() # buffer setup
        prepare_step = 0
        # random explore before training
        initial_epsilon = 1.0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        while not self.facade.is_testing_available:
            observation = self.run_episode_step_test(0, initial_epsilon, observation, True)
            prepare_step += 1
        # training records
        self.training_history = {"critic_loss": [0,0,0,0,0,0,0,0,0,0], "actor_loss": [0,0,0,0,0,0,0,0,0,0], "hyper_actor_loss": [0,0,0,0,0,0,0,0,0,0]}
        
        print(f"Total {prepare_step} prepare steps")
    
    def action_after_test(self):
        self.facade.stop_env()

    def step_test(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
        reward = torch.FloatTensor(reward)
        done_mask = torch.FloatTensor(done_mask)
        
        # critic_loss, actor_loss, hac_loss = self.get_hac_loss_test(observation, policy_output, reward, done_mask, next_observation)

    def get_hac_loss_test(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = False, do_critic_update = False):
        
        # critic loss

        # user feedback
        # Get current Q estimate

        hyper_output = self.facade.infer_hyper_action(observation, policy_output, self.actor)
        
        exposure_features = hyper_output['action_features']
        batch_data = {
            'user_profile': observation['user_profile'],
            'history_features': observation['history_features'],
            'exposure_features': exposure_features 
        }
        with torch.no_grad():
            output_dict = self.user_response_model(batch_data)
            response = torch.bernoulli(output_dict['probs']) # (B, slate_size)
            probs_under_temper = output_dict['probs'] # * prob_scale
            user_response = torch.bernoulli(probs_under_temper).detach() # (B, slate_size)       

        # reward = self.cal_reward(user_response)

        current_critic_output = self.facade.apply_critic(observation, hyper_output, user_response, self.critic)
        current_Q = current_critic_output['q']


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
        

        target_critic_output = self.facade.apply_critic(next_observation, next_policy_output, next_user_response, self.critic_target)
        target_Q = target_critic_output['q']
        target_Q = reward + self.gamma * (done_mask * target_Q).detach()
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        # if do_critic_update and self.critic_lr > 0:
        #     # Optimize the critic
        #     self.critic_optimizer.zero_grad()
        #     critic_loss.backward()
        #     self.critic_optimizer.step()

        # actor loss
        
        # Compute actor loss
        if do_actor_update and self.actor_lr > 0:
            # self.actor_optimizer.zero_grad()
            policy_output = self.facade.apply_policy(observation, self.actor)
            critic_output = self.facade.apply_critic(observation, policy_output, user_response, self.critic)
            actor_loss = -critic_output['q'].mean()
            # Optimize the actor 
            # actor_loss.backward()
            # self.actor_optimizer.step()
            
        # hyper actor loss
        
        if do_actor_update and self.hyper_actor_coef > 0:
            # self.actor_optimizer.zero_grad()
            policy_output = self.facade.apply_policy(observation, self.actor)
            inferred_hyper_output = self.facade.infer_hyper_action(observation, policy_output, self.actor)
            hyper_actor_loss = self.hyper_actor_coef * F.mse_loss(inferred_hyper_output['Z'], policy_output['Z']).mean()
            # Optimize the actor 
            # hyper_actor_loss.backward()
            # self.actor_optimizer.step()
            
        return critic_loss, actor_loss, hyper_actor_loss

    def run_episode_step_test(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = False)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation
            