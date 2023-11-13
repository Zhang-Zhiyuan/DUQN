import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils

class BaseRLAgent():
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
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
        parser.add_argument('--gamma', type=float, default=0.95, 
                            help='reward discount')
        parser.add_argument('--n_iter', type=int, nargs='+', default=[2000], 
                            help='number of training iterations')
        parser.add_argument('--train_every_n_step', type=int, default=1, 
                            help='number of training iterations')
        parser.add_argument('--initial_greedy_epsilon', type=float, default=0.6, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--final_greedy_epsilon', type=float, default=0.05, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--elbow_greedy', type=float, default=0.5, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--check_episode', type=int, default=100, 
                            help='number of iterations to check output and evaluate')
        parser.add_argument('--with_eval', action='store_true',
                            help='do evaluation during training')
        parser.add_argument('--save_path', type=str, required=True, 
                            help='save path for networks')
        return parser
    
    def __init__(self, args, facade):
        self.device = args.device
        self.gamma = args.gamma
        self.n_iter = [0] + args.n_iter
        self.train_every_n_step = args.train_every_n_step
        self.check_episode = args.check_episode
        self.with_eval = args.with_eval
        self.save_path = args.save_path
        self.facade = facade
        self.exploration_scheduler = utils.LinearScheduler(int(sum(args.n_iter) * args.elbow_greedy), 
                                                           args.final_greedy_epsilon, 
                                                           initial_p=args.initial_greedy_epsilon)
        if len(self.n_iter) == 2:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
    
    def train(self):
        if len(self.n_iter) > 2:
            self.load()

        best_average_reward = 0.0
        t = time.time()
        
        print("Run procedures before training")
        self.action_before_train()
        t = time.time()
        start_time = t
        # training
        print("Training:")
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        step_offset = sum(self.n_iter[:-1])
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation, True)
            if i % self.train_every_n_step == 0:
                self.step_train()
            if i % self.check_episode == 0:
                t_ = time.time()
                print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
                print(self.log_iteration(i))
                t = t_
                if i % (3*self.check_episode) == 0:
                    current_average_reward = self.facade.get_episode_report(10)["average_total_reward"]
                    if current_average_reward > best_average_reward:
                        self.save()
                        best_average_reward = current_average_reward
                    # else:
                    #     self.load()
        self.action_after_train()
        
    
    
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
    
    def action_after_train(self):
        self.facade.stop_env()
        
    def get_report(self):
        episode_report = self.facade.get_episode_report(10)
        train_report = {k: np.mean(v[-10:]) for k,v in self.training_history.items()}
        return episode_report, train_report
        
#     def run_an_episode(self, epsilon, initial_observation = None, with_train = False):
#         pass

    def run_episode_step(self, *episode_args):
        pass
    
    
    def step_train(self):
        pass
    
    
    def log_iteration(self, step):
        episode_report, train_report = self.get_report()
        log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report}\n"
        with open(self.save_path + ".report", 'a') as outfile:
            outfile.write(log_str)
        return log_str
    
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
    #                 self.step_train()
    #             if i % self.check_episode == 0:
    #                 t_ = time.time()
    #                 print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
    #                 print(self.log_iteration(i))
    #                 t = t_
    #                 if i % (3*self.check_episode) == 0:
    #                     self.save()
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