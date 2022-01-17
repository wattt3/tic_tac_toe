import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--eps", default=0, type=float, dest="eps")
parser.add_argument("--alpha", default=0.1, type=float, dest="alpha")
parser.add_argument("--init_V", default=1, type=float, dest="init_V")

args = parser.parse_args()

# X가 선공
# X = -1, O = 1

class Agent:
    def __init__(self, args):
        self.eps = args.eps
        self.alpha = args.alpha
        self.states = []
        self.win = 0
        self.symbol = -1 # env.X

    def init_V(self, env, state_winner_end, args):
        V = np.zeros(env.max_states)
        for state, winner, end in state_winner_end:
            if end:
                if winner == self.symbol:
                    state_value = 1
                else:
                    state_value = 0
            else:
                state_value = args.init_V
            
            V[state] = state_value
        self.V = V

    def update_states(self, state):
        self.states.append(state)

    def move(self, env):
        move = None
        rand_num = np.random.rand()

        if rand_num < self.eps:
            move = env.random_move()
        else:
            move = env.best_move(self)
        
        env.board[move[0], move[1]] = self.symbol

    def update(self, env):
        after_move = env.reward(self.symbol)
        for prev in reversed(self.states):
            value = self.V[prev] + self.alpha * (after_move - self.V[prev])
            self.V[prev] = value
            after_move = value

        if self.symbol == env.winner:
            self.win += 1
        
        self.states = []

class Random_Agent:
    def __init__(self):
        self.symbol = 1 # env.O
    
    def move(self, env):
        env.board[env.random_move()] = self.symbol

class Env:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.X = -1
        self.O = 1
        self.winner = None
        self.end = False
        self.max_states = 3 ** 9
    
    def is_space(self, i, j):
        return self.board[i, j] == 0

    def get_space(self):
        space = []
        for i in range(3):
            for j in range(3):
                if self.is_space(i, j):
                    space.append((i, j))
        return space

    def reward(self, symbol):
        reward = 0
        if self.game_over() and self.winner == symbol:
            reward = 1
        return reward

    # state의 hash 구하기
    def get_hash(self):
        state = 0
        idx = 0
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == self.X:
                    v = 1
                elif self.board[i, j] == self.O:
                    v = 2
                else:
                    v = 0
                state += (idx ** 3) * v
                idx += 1
        return state

    def random_move(self):
        space = self.get_space()
        rand_idx = np.random.choice(len(space))
        rand_move = space[rand_idx]
        return rand_move

    def best_move(self, agent):
        best_value = -1
        best_move = None

        for i in range(3):
            for j in range(3):
                if self.is_space(i, j):
                    self.board[i, j] = agent.symbol
                    state = self.get_hash()
                    self.board[i, j] = 0
                    if agent.V[state] > best_value:
                        best_value = agent.V[state]
                        best_move = (i, j)
        
        return best_move

    def game_over(self):
        if self.end:
            return True
        
        player = [self.X, self.O] # -1, 1
        
        for p in player:
            for i in range(3): # row
                if self.board[i].sum() == p * 3:
                    self.winner = p
                    self.end = True
                    return True

            for j in range(3): # column
                if self.board[:, j].sum() == p * 3:
                    self.winner = p
                    self.end = True
                    return True
            
            # diagonal
            if self.board.trace() == p * 3 or np.fliplr(self.board).trace() == p * 3:
                self.winner = p
                self.end = True
                return True

        # draw
        if np.all(self.board != 0):
            self.winner = None
            self.end = True
            return True

        # game is not over
        self.winner = None
        self.end = None
        return False

# 3 ^ 9 경우의 수에 대한 state, winner, end
# init_V의 매개변수
def get_state_winner_end(env, i=0, j=0):
    results = []
    for v in [env.X, 0, env.O]: # -1, 0, 1
        env.board[i, j] = v
        if i == 2:
            if j == 2:
                state = env.get_hash()
                end = env.game_over()
                winner = env.winner
                results.append((state, winner, end))
            else:
                results += get_state_winner_end(env, 0, j + 1)
        else:
            results += get_state_winner_end(env, i + 1, j)
    return results

def play(agent, random_agent , env):
    current_player = None
    continue_game = True
    while continue_game:
        if current_player == agent:
            current_player = random_agent
        else:
            current_player = agent

        current_player.move(env)

        if current_player == agent:
            state = env.get_hash()
            agent.update_states(state)

        if env.game_over():
            agent.update(env)
            continue_game = False

def main(args):
    env = Env()

    agent = Agent(args)
    agent.init_V(env, get_state_winner_end(env), args)

    random_agent = Random_Agent()

    x_axis = np.arange(0, 10000, 100)
    y_axis = []

    for i in range(1, 10001):
        play(agent, random_agent, Env())
        if i % 100 == 0:
            y_axis.append(agent.win / i)
    
    plt.plot(x_axis, y_axis)
    plt.show()

if __name__ == '__main__':
    main(args)

# david silver