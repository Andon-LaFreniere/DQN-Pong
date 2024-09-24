import os
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

D = 80 * 80  # input dimensionality: 80x80 grid
if resume and os.path.isfile('model_46500.pkl'):
    with open('model_46500.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    model = {
        'W1': np.random.randn(H, D) / np.sqrt(D),  # "Xavier" initialization
        'W2': np.random.randn(H) / np.sqrt(H),
        'W1_val': np.random.randn(H, D) / np.sqrt(D),
        'W2_val': np.random.randn(H) / np.sqrt(H)
    }

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    I = I[0] if isinstance(I, tuple) else I
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    v = np.dot(model['W1_val'], x)
    v[h < 0] = 0
    v = np.dot(model['W2_val'], v)
    return p, v, h  # return probability of taking action 2, value, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backprop ReLU
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

def critic_backward(eph, advantages):
    dW2_val = np.dot(eph.T, advantages).ravel()
    dh_val = np.outer(advantages, model['W2_val'])
    dh_val[eph <= 0] = 0  # backprop ReLU
    dW1_val = np.dot(dh_val.T, epx)
    return {'W1_val': dW1_val, 'W2_val': dW2_val}

def save_model(model, filename='model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename='model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs, vs = [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
reward_history = []

while episode_number < 30000:
    if render:
        env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, value, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    xs.append(x)
    hs.append(h)
    y = 1 if action == 2 else 0
    dlogps.append(y - aprob)
    vs.append(value)

    observation, reward, done, truncated, info = env.step(action)
    reward_sum += reward

    drs.append(reward)

    if done:
        episode_number += 1
        reward_history.append(reward_sum)

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        epv = np.vstack(vs)
        xs, hs, dlogps, drs, vs = [], [], [], [], []

        discounted_epr = discount_rewards(epr)
        advantages = discounted_epr - epv

        epdlogp *= advantages

        grad = policy_backward(eph, epdlogp)
        for k in model:
            if 'val' not in k:
                grad_buffer[k] += grad[k]

        value_grad = critic_backward(eph, advantages)
        for k in model:
            if 'val' in k:
                grad_buffer[k] += value_grad[k]

        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'resetting env. episode reward total was {reward_sum}. running mean: {running_reward}')
        reward_sum = 0
        observation = env.reset()
        prev_x = None

plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.show()
