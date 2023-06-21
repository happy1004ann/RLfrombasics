import numpy as np

from world import AgentBase, GridWorld2


class QAgent(AgentBase):
    def __init__(self):
        super().__init__()
        self.alpha = 0.1

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        # Q러닝 업데이트 식을 이용 
        self.q_table[x,y,a] += self.alpha * (r + np.amax(self.q_table[next_x,next_y,:]) - self.q_table[x,y,a])

    def anneal_eps(self):
        self.eps -= 0.01  # Q러닝에선 epsilon 이 좀더 천천히 줄어 들도록 함.
        self.eps = max(self.eps, 0.2) 


def main():
    env = GridWorld2()
    agent = QAgent()

    for _ in range(1000):
        done = False

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s = s_prime
        agent.anneal_eps()

    agent.show_table()

if __name__ == '__main__':
    main()