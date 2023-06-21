import numpy as np

from world import AgentBase, GridWorld2


class QAgent(AgentBase):
    def __init__(self):
        super().__init__()
        self.alpha = 0.01

    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x, y = s
            # 몬테 카를로 방식을 이용하여 업데이트.
            self.q_table[x,y,a] += self.alpha * (cum_reward-self.q_table[x,y,a])
            cum_reward += r 

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

      
def main():
    env = GridWorld2()
    agent = QAgent()

    for _ in range(1000): # 총 1,000 에피소드 동안 학습
        done = False
        history = []

        s = env.reset()
        while not done: # 한 에피소드가 끝날 때 까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime
        agent.update_table(history) # 히스토리를 이용하여 에이전트를 업데이트
        agent.anneal_eps()

    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()