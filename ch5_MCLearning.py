from world import Agent, GridWorld1


def main():
    # MC
    env = GridWorld1()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.001

    for _ in range(50000):
        done = False
        history = []

        while not done:
            action = agent.select_action()
            (x,y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        cum_reward = 0
        for transition in history[::-1]:
            x, y, reward = transition
            data[x][y] += alpha*(cum_reward-data[x][y])
            cum_reward = reward + gamma*cum_reward 
            
    for row in data:
        print(row)

if __name__ == '__main__':
    main()
