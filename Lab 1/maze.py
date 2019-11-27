import random
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = 0
    STEP_REWARD = 0


    def __init__(self, maze, weights=None, random_rewards=False, minotaur_stand_ground=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.stand_ground             = minotaur_stand_ground;
        self.states, self.map, self.exits, self.eaten = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        exits = set();
        eaten = set();
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for i_m in range(self.maze.shape[0]):
                    for j_m in range(self.maze.shape[1]):
                        states[s] = (i,j,i_m,j_m);
                        if (i == i_m and j == j_m):
                            eaten.add(s)
                        elif (self.maze[i][j] == 2):
                            exits.add(s)
                        map[(i,j,i_m,j_m)] = s;
                        s += 1;
        return states, map, exits, eaten

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # If eaten or escaped, only possible state is to remain in state (game over)
        if state in self.eaten or state in self.exits:
            return [state]
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        player_position = ()
        if hitting_maze_walls:
            player_position = (self.states[state][0],self.states[state][1])
        else:
            player_position = (row, col)
        # All possible future positions
        possible_next_s = []
        if (self.states[state][2] + 1) < len(self.maze):
            possible_next_s.append(self.map[(player_position[0], player_position[1], self.states[state][2] + 1, self.states[state][3])])
        if (self.states[state][3] + 1) < len(self.maze[0]):
            possible_next_s.append(self.map[(player_position[0], player_position[1], self.states[state][2], self.states[state][3] + 1)])
        if (self.states[state][2] - 1) >= 0:
            possible_next_s.append(self.map[(player_position[0], player_position[1], self.states[state][2] - 1, self.states[state][3])])
        if (self.states[state][3] - 1) >= 0:
            possible_next_s.append(self.map[(player_position[0], player_position[1], self.states[state][2], self.states[state][3] - 1)])
        if (self.stand_ground):
            possible_next_s.append(self.map[(player_position[0], player_position[1], self.states[state][2], self.states[state][3])])
        return possible_next_s

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                possible_next_s = self.__move(s,a);
                for next_s in possible_next_s:
                    transition_probabilities[next_s, s, a] = 1/len(possible_next_s);
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            #No reward if already eaten or exited
            if (s not in self.exits) and (s not in self.eaten):
                for a in range(self.n_actions):
                    possible_next_s = self.__move(s,a);
                    for next_s in possible_next_s:
                        if next_s in self.exits:
                            rewards[s,a] = rewards[s,a] + 1/len(possible_next_s)
                        else:
                            rewards[s,a] = rewards[s,a]
        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon:
                # Move to next state given the policy and the current state
                next_s = random.choice(self.__move(s,policy[s,t]));
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));

    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        # Erase Old Cells
        grid.get_celld()[(path[i-1][0],path[i-1][1])].set_facecolor(col_map[maze[path[i-1][0],path[i-1][1]]])
        grid.get_celld()[(path[i-1][0],path[i-1][1])].get_text().set_text('')
        grid.get_celld()[(path[i-1][2],path[i-1][3])].set_facecolor(col_map[maze[path[i-1][2],path[i-1][3]]])
        grid.get_celld()[(path[i-1][2],path[i-1][3])].get_text().set_text('')

        if (path[i][0] == path[i][2] and path[i][1] == path[i][3]):
            grid.get_celld()[(path[i][0],path[i][1])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][0],path[i][1])].get_text().set_text('Player is eaten')
        elif maze[path[i][0],path[i][1]] == 2: #maze object defines exit as 2
            grid.get_celld()[(path[i][0],path[i][1])].set_facecolor(LIGHT_GREEN)
            grid.get_celld()[(path[i][0],path[i][1])].get_text().set_text('Player is out')
            grid.get_celld()[(path[i][2],path[i][3])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i][2],path[i][3])].get_text().set_text('Minotaur')
        else:
            grid.get_celld()[(path[i][0],path[i][1])].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[(path[i][0],path[i][1])].get_text().set_text('Player')
            grid.get_celld()[(path[i][2],path[i][3])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i][2],path[i][3])].get_text().set_text('Minotaur')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
