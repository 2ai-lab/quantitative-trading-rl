import os
import time
from pre_processing import processed_data_dict
from agent import DQN
from env import Action, TradingEnvironment
import matplotlib.pyplot as plt
import numpy as np
import torch

# HPC cluster path location
# PREDS_DIR_PATH = "/home/usd.local/aniket.kumar/aniket/transformer-rl/preds"
# CHECKPOINTS_DIR = "/home/usd.local/aniket.kumar/aniket/transformer-rl/checkpoints"

# local path location
PREDS_DIR_PATH = "./preds"
CHECKPOINTS_DIR = "./checkpoints"

input_data_dict = processed_data_dict()
ticker_name = 'AAPL'
# ticker_name = 'TSLA'
# ticker_name = 'GOOG'
# ticker_name = 'MSFT'
# ticker_name = 'AMZN'
# testing purpose ONLY!! later change the variable name everywhere
df_aapl = input_data_dict[ticker_name]

print(f'Dataset used is {ticker_name}')


def add_label(df):
    df['Action'] = 0
    for i in range(3, len(df)):
        three_days_pred = 0.4 * (df['Close'][i-2] - df['Close'][i-3])/df['Close'][i-3] + 0.32 * (df['Close'][i-1] - df['Close'][i-2])/df['Close'][i-2] + 0.28 * (df['Close'][i] - df['Close'][i-1])/df['Close'][i-1]
        # sell, the stock price will rise in the next three days
        if three_days_pred > 0.01:
            df.iloc[i, df.columns.get_loc('Action')] = Action.SELL.value
        # buy, the stock price will fall in the next three days
        elif three_days_pred < -0.01:
            df.iloc[i, df.columns.get_loc('Action')] = Action.BUY.value
        # hold, the stock price will remain the same in the next three days
        else:
            df.iloc[i, df.columns.get_loc('Action')] = Action.HOLD.value
    return df

df_aapl = add_label(df_aapl)
# testing data from 2015-01-01 to last date of data
df_aapl_test = df_aapl[df_aapl['Date'] >= '2021-01-01'].reset_index(drop=True)
df_aapl_train = df_aapl[(df_aapl['Date'] >= '2010-01-01') & (df_aapl['Date'] < '2021-01-01')].reset_index(drop=True)

env = TradingEnvironment(df_aapl_train)

# Parameters
state_size = env.state_size
action_size = env.action_space
hidden_size = 128
buffer_size = 10000
batch_size = 128
gamma = 0.95
lr = 0.1
initial_exploration = 1.0
exploration_decay = 0.9
min_exploration = 0.001
max_iter_episode = 100
target_update_frequency = 2
num_layers = 3
num_heads = state_size
while_used = True
loss_fn_used = 'KL Div Loss'
optimizer_used = 'RMSProp'

dqn_agent = DQN(state_size, action_size, lr, gamma, num_layers, num_heads)
print(f'Running on: {dqn_agent.device.type.upper()}')

# load model from checkpoint directory if present
def load_model():
    if os.path.exists(CHECKPOINTS_DIR):
        checkpoints = os.listdir(CHECKPOINTS_DIR)
        if len(checkpoints) == 0:
            print('***** NO CHECKPOINTS PRESENT *****')
            return 0
        print('***** LOADING MODEL *****')
        sorted_files = sorted(checkpoints, key=lambda x: os.path.getmtime(CHECKPOINTS_DIR + '/' + x), reverse=True)
        latest_checkpoint = sorted_files[0]
        print(f'Latest checkpoint: {latest_checkpoint}')
        # get the starting episode number
        last_saved_episode = int(latest_checkpoint.split('_')[1].split('.')[0])
        print(f'Last saved episode: {last_saved_episode}')
        dqn_agent.q_network.load_state_dict(torch.load(f'{CHECKPOINTS_DIR}/{latest_checkpoint}'))
        dqn_agent.target_network.load_state_dict(torch.load(f'{CHECKPOINTS_DIR}/{latest_checkpoint}'))
        print('***** MODEL LOADED *****')
        # starting from next episode after the last saved checkpoint
        return last_saved_episode + 1

load_checkpoint = False
start_episode = load_model() if load_checkpoint else 0
num_episodes = start_episode + 100 if start_episode else 10000
loss = []
avg_reward = []
profit_list = []
accuracy = []
total_reward = 0
for episode in range(start_episode, num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    profit = 0
    epsilon = max(min_exploration, initial_exploration * (exploration_decay ** episode))
    count = 0

    while not done:
        action = dqn_agent.decide_action(state, epsilon)
        next_state, reward, done, labels = env.step(action)
        dqn_agent.remember(state, action, reward, next_state, done, labels)
        dqn_agent.train(state, action, reward, next_state, done, labels, count)
        episode_reward += reward
        state = next_state
        count += 1

    # updating the target network
    if episode % target_update_frequency == 0:
        dqn_agent.update_target_network()

    # save model after every 10 episodes
    if episode % 10 == 0:
       print('***** SAVING MODEL *****')
       if not os.path.exists(CHECKPOINTS_DIR):
           os.mkdir(CHECKPOINTS_DIR)
       torch.save(dqn_agent.q_network.state_dict(), f'{CHECKPOINTS_DIR}/checkpoint_{episode}.pth')
       print('***** MODEL SAVED *****')
    
    total_reward += episode_reward
    avg_reward.append(total_reward)
    loss.append(np.mean(dqn_agent.loss))
    # reset the loss after every episode
    # dqn_agent.loss = []
    mean_acc = np.mean(dqn_agent.accuracy)
    accuracy.append(mean_acc)
    if start_episode:
        print(f'Episode: {episode}, Reward per episode: {total_reward}, Accuracy: {mean_acc*100}%')
    else:
        print(f'Episode: {episode + 1}, Reward per episode: {episode_reward}, Accuracy: {mean_acc*100}%')

average_reward = total_reward/num_episodes
print(f'Average reward: {average_reward}')

#acc = np.mean(accuracy)
print(f'Accuracy: {accuracy[-1]*100}%')
print(f'Loss: {loss[-1]}')

print('***** SAVING FILES *****')
curr_time = time.strftime("%Y_%m_%d_%H_%M_%S")
def save_plots():
    # plot the reward
    plt.subplot(3, 1, 1)
    plt.plot(avg_reward)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward')
    
    # plot loss
    plt.subplot(3, 1, 2)
    plt.plot(loss)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Graph')

    # plot accuracy
    plt.subplot(3, 1, 3)
    plt.plot(list(map(lambda x: x*100, accuracy)))
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Graph')

    # save file
    plt.tight_layout()
    plt.savefig(f'{PREDS_DIR_PATH}/reward_loss_acc_graph.png')
    plt.show()

# create a log file for all the input parameters
def create_log_file():
    with open(f'{PREDS_DIR_PATH}/log.txt', 'w') as f:
        f.write(f'Num Episodes: {num_episodes}\n')
        f.write(f'Batch Size: {batch_size}\n')
        f.write(f'Learning Rate: {lr}\n')
        f.write(f'Initial Exploration: {initial_exploration}\n')
        f.write(f'Exploration Decay: {exploration_decay}\n')
        f.write(f'Min Exploration: {min_exploration}\n')
        f.write(f'Num layers: {num_layers}\n')
        f.write(f'Num heads: {num_heads}\n')
        f.write(f'While loop used: {while_used}\n')
        f.write(f'Loss function used: {loss_fn_used}\n')
        f.write(f'Optimizer used: {optimizer_used}\n')
        f.write(f'Average Reward: {total_reward/num_episodes}\n')
        # f.write(f'Test Accuracy: {test_acc*100}%\n')
        # f.write(f'Test Loss: {test_loss}\n')
        
if not os.path.exists(f'{PREDS_DIR_PATH}/{curr_time}'):
    PREDS_DIR_PATH += '/' + curr_time
    os.mkdir(f'{PREDS_DIR_PATH}')
save_plots()
create_log_file()
print('***** DONE *****')
