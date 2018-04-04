import gym
import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import pickle


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you're confused, see examples below. Please don't assume that states are integers (they'll get different later).
    """

    # reward_threshold = <Compute minimum reward for elite sessions. Hint: use np.percentile>
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []
    for i in range(len(states_batch)):
        for s, a in zip(states_batch[i], actions_batch[i]):
            if rewards_batch[i] >= reward_threshold:
                elite_states.append(s)
                elite_actions.append(a)
    return elite_states, elite_actions


def generate_session(t_max, env, agent, n_actions):
    states, actions = [], []
    total_reward = 0

    s = env.reset()

    for t in range(t_max):
        # predict array of action probabilities
        probs = agent.predict_proba([s])[0]

        # a = <sample action with such probabilities>
        a = np.random.choice(list(range(n_actions)), p=probs)
        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward


class CrossEntropyMethod:
    def __init__(self, env):
        self.env = env
        n_actions = env.action_space.n

        agent = MLPClassifier(hidden_layer_sizes=(20, 20),
                              activation='tanh',
                              warm_start=True,  # keep progress between .fit(...) calls
                              max_iter=1  # make only 1 iteration on each .fit(...)
                              )
        # initialize agent to the dimension of state an amount of actions
        agent.fit([env.reset()] * n_actions, range(n_actions))

        self.agent = agent
        print(agent.predict_proba([env.reset()]))

    def generate_session(self, t_max):
        return generate_session(t_max=t_max, agent=self.agent, env=self.env, n_actions=self.env.action_space.n)

    def generate_sessions(self, n_sessions, t_max_session):
        sessions = []
        for _ in tqdm(range(n_sessions)):
            sessions.append(self.generate_session(t_max=t_max_session))
        return sessions

    def train(self):
        n_sessions = 100
        num_iteration = 1000
        t_max_session = 10 ** 4

        # num_iteration = 10
        # t_max_session = 10

        percentile = 70
        learning_rate = 0.5  # add this thing to all counts for stability
        log = []

        pbar = tqdm(range(num_iteration))

        for i in pbar:
            # generate new sessions
            sessions = self.generate_sessions(n_sessions=n_sessions, t_max_session=t_max_session)

            states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))

            elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch)

            self.agent.fit(elite_states, elite_actions)
            # <fit agent to predict elite_actions(y) from elite_states(X)>

            # show_progress(rewards_batch, log, reward_range=[0, np.max(rewards_batch)])
            pbar.set_description(f"mean reward = {np.mean(rewards_batch)}")

            cross_entropy_method.save_agent('agent.pkl')

        print(f"final mean_rewards = {np.mean(rewards_batch)})")

    def save_agent(self, path_to_save: str):
        with open(path_to_save, 'wb') as f:
            pickle.dump(self.agent, f)

    def load_agent(self, path_to_save: str):
        with open(path_to_save, 'rb') as f:
            self.agent = pickle.load(f)

if __name__ == '__main__':
    env = gym.make("MountainCar-v0").env
    # env = gym.make("CartPole-v0").env
    env.reset()

    cross_entropy_method = CrossEntropyMethod(env)
    cross_entropy_method.train()
    cross_entropy_method.save_agent('agent.pkl')