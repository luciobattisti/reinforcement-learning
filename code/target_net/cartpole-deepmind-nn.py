import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import math
import time
import os

os.putenv("LC_ALL", "C.UTF-8")
os.putenv("LANG", "C.UTF-8")
import click

from ExperienceReplay import ExperienceReplay
from QNetAgent import QNetAgent


# Functions
def calculate_epsilon(steps, value, final, decay):
    epsilon = (final + (value - final) * math.exp(-1 * steps / decay))

    return epsilon

# If gpu is to be used
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@click.command()
@click.option("--env",
              default="CartPole-v0",
              type=str,
              help="Set open gym environment.")
@click.option("--seed",
              default=23,
              type=int,
              help="Set seed for reproducibility.")
@click.option("--learning_rate",
              default=0.001,
              type=float,
              help="Set learning rate")
@click.option("--num_episodes",
              default=500,
              type=int,
              help="Set number of episodes")
@click.option("--gamma",
              default=0.99,
              type=float,
              help="Set value for gamma in Bellman equation.")
@click.option("--egreedy",
              default=0.9,
              type=float,
              help="Set value for probability of random action")
@click.option("--egreedy_final",
              default=0.01,
              type=float,
              help="Set value for asymptotic probability of random action.")
@click.option("--egreedy_decay",
              default=500,
              type=float,
              help="Set the number of steps for egreedy decay.")
@click.option("--hidden_layer",
              default=64,
              type=int,
              help="Set size of hidden layer for neural network")
@click.option("--report_interval",
              default=10,
              type=int,
              help="Set number of episodes needed to show report")
@click.option("--score_to_solve",
              default=195,
              type=int,
              help="Set score to solve the game.")
@click.option("--replay_memory_size",
              default=50000,
              type=int,
              help="Set size of memory used for experience reply.")
@click.option("--batch_size",
              default=32,
              type=int,
              help="Set batch size for neural network")
@click.option("--update_target_frequency",
              default=500,
              type=int,
              help="Set number of steps needed to perform an update on target network parameters")
@click.option("--clip_error",
              is_flag=True,
              help="Perform error clipping between -1 and 1 if True.")
def main(env, seed, learning_rate, num_episodes, gamma, egreedy, egreedy_final, egreedy_decay, hidden_layer,
         report_interval,score_to_solve, replay_memory_size, batch_size, update_target_frequency, clip_error):
    # Set env
    env = gym.make(env)
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Init input and solved variables
    number_of_inputs = env.observation_space.shape[0]
    number_of_outputs = env.action_space.n
    solved_after = 0
    solved = False

    # Init more variables
    memory = ExperienceReplay(replay_memory_size)
    qnetAgent = QNetAgent(env, device, learning_rate, number_of_inputs, hidden_layer, number_of_outputs)
    steps_total = []
    frames_total = 0
    start_time = time.time()

    for i_episode in range(num_episodes):
        state = env.reset()

        step = 0
        # for step in range(100):
        while True:
            step += 1
            frames_total += 1
            epsilon = calculate_epsilon(frames_total, egreedy, egreedy_final, egreedy_decay)

            action = qnetAgent.select_action(state, epsilon)
            new_state, reward, done, info = env.step(action)
            memory.push(state, action, new_state, reward, done)
            qnetAgent.optimize(memory, batch_size, gamma, clip_error, update_target_frequency)

            state = new_state

            if done:
                steps_total.append(step)
                mean_reward_100 = sum(steps_total[-100:]) / 100

                if mean_reward_100 > score_to_solve and not solved:
                    print("SOLVED! After %i episodes" % i_episode)
                    solved_after = i_episode
                    solved = True

                if i_episode % report_interval == 0:
                    print("\n*** Episode %i *** "
                          "\nAv.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f"
                          "\nepsilon: %.2f, frames_total: %i"
                          % (i_episode, report_interval,
                             sum(steps_total[-report_interval:]) / report_interval,
                             mean_reward_100,
                             sum(steps_total) / len(steps_total),
                             epsilon,
                             frames_total)
                          )
                    elapsed_time = time.time() - start_time
                    print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                break

    print("Average reward: %.2f" % (sum(steps_total) / num_episodes))
    print("Average reward of steps (last 100 episodes): %.2f" % (sum(steps_total[-100:]) / 100))
    if solved:
        print("Solved after %i episodes" % solved_after)

    plt.figure(figsize=(12, 5))
    plt.title("Rewards")
    plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color="green")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
