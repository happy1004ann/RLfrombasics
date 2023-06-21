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
        a_prime = self.select_action(s_prime) # S'에서 선택할 액션 (실제로 취한 액션이 아님)
        # SARSA 업데이트 식을 이용
        self.q_table[x,y,a] += self.alpha * (r + self.q_table[next_x,next_y,a_prime] - self.q_table[x,y,a])

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

      
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