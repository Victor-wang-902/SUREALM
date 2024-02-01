from scipy.stats import poisson
import numpy as np
from tqdm import tqdm

class thingy:
    def __init__(self, mu_a, mu_d):
        self.mu_a = mu_a
        self.mu_d = mu_d
        self.s = list(range(21))
        self.p = np.zeros((21,21))
        self.poisson_a = [0. for _ in range(21)]
        self.poisson_d = [0. for _ in range(21)]
        self.init_poisson()
        self.calc_p()
        self.v = [0. for _ in range(21)]
        self.r = [0. for _ in range(21)]
        self.calc_r()
        self.gamma = 0.9
        self.calc_v()


    def init_poisson(self):
        for x in range(21):
            self.poisson_a[x] += poisson.pmf(x, self.mu_a)
        for y in range(21):
            self.poisson_d[y] += poisson.pmf(y, self.mu_d)
    
    def calc_p(self):
        for i in range(21):
            for j in range(21):
                sum_pa = 0.
                for a in range(21):
                    sum_pd = 0.
                    for d in range(21):
                        if min(max(self.s[i] - d, 0) + a, 20) == self.s[j]:
                            sum_pd += self.poisson_d[d]
                        
                    sum_pa += self.poisson_a[a] * sum_pd
                self.p[i][j] += sum_pa

    def calc_r(self):
        for i in range(21):
            sum_s_next = 0.
            for j in range(21):
                sum_pa = 0.
                for a in range(21):
                    sum_pd = 0.
                    for d in range(21):
                        if min(max(self.s[i] - d, 0) + a, 20) == self.s[j]:
                            if self.s[i] >= d:
                                reward = 10 * d
                            else:
                                reward = 10 * self.s[i]
                            sum_pd += self.poisson_d[d] * reward
                    sum_pa += self.poisson_a[a] * sum_pd
                sum_s_next += sum_pa
            self.r[i] += sum_s_next
                            
    def calc_v(self):
        while True:
            delta = 0
            for i in range(21):
                v = self.v[i]
                sum_v = self.r[i]
                for j in range(21):
                    sum_v += self.gamma * self.p[i][j] * self.v[j]
                self.v[i] = sum_v
                delta = max(delta, abs(v - self.v[i]))
            if delta < 1e-8:
                break
        
    
                


class Problem2:
    def __init__(self, mu_a_1, mu_d_1, mu_a_2, mu_d_2):
        self.mu_a_1 = mu_a_1
        self.mu_d_1 = mu_d_1
        self.mu_a_2 = mu_a_2
        self.mu_d_2 = mu_d_2
        self.s = list(range(21*21))
        self.p = np.zeros((11,21*21,21*21))
        self.poisson_a_1 = [0. for _ in range(10)]
        self.poisson_d_1 = [0. for _ in range(10)]
        self.poisson_a_2 = [0. for _ in range(10)]
        self.poisson_d_2 = [0. for _ in range(10)]
        self.init_poisson()
        self.calc_p()
        self.action_space = range(-5,6)
        self.pi = [0 for _ in range(21*21)]
        self.v = [0. for _ in range(21*21)]
        self.r = np.zeros((11, 21*21))
        self.calc_r()
        self.gamma = 0.9
        self.policy_iter()


    def init_poisson(self):
        for x in range(10):
            self.poisson_a_1[x] += poisson.pmf(x, self.mu_a_1)
        for y in range(10):
            self.poisson_d_1[y] += poisson.pmf(y, self.mu_d_1)
        for x in range(10):
            self.poisson_a_2[x] += poisson.pmf(x, self.mu_a_2)
        for y in range(10):
            self.poisson_d_2[y] += poisson.pmf(y, self.mu_d_2)
    
    def calc_p(self):
        for A in range(11):
            act = A-5
            for i in tqdm(range(21*21)):
                for j in tqdm(range(21*21)):
                    sum_pa1 = 0.
                    for a1 in range(10):
                        sum_pd1 = 0.
                        for d1 in range(10):
                            sum_pa2 = 0.
                            for a2 in range(10):
                                sum_pd2 = 0.
                                for d2 in range(10):
                                    s1 = self.s[i] // 21
                                    s2 = self.s[i] % 21
                                    s1_prime = self.s[j] // 21
                                    s2_prime = self.s[j] % 21
                                    if (
                                        min(max(max(s1 - d1, 0) - act, 0) + a1, 20) == s1_prime and
                                        min(max(max(s2 - d2, 0) + act, 0) + a2, 20) == s2_prime and
                                        max(s1 - d1, 0) - act >= 0 and
                                        max(s2 - d2, 0) + act >= 0
                                        ):
                                        sum_pd2 += self.poisson_d_2[d2]
                                sum_pa2 += self.poisson_a_2[a2] * sum_pd2
                            sum_pd1 += self.poisson_d_1[d1] * sum_pa2
                        sum_pa1 += self.poisson_a_1[a1] * sum_pd1
                    self.p[A][i][j] += sum_pa1

    def calc_r(self):
        for A in range(11):
            act = A-5
            for i in range(21*21):
                sum_pa1 = 0.
                for a1 in range(10):
                    sum_pd1 = 0.
                    for d1 in range(10):
                        sum_pa2 = 0.
                        for a2 in range(10):
                            sum_pd2 = 0.
                            for d2 in range(10):
                                s1 = self.s[i] // 21
                                s2 = self.s[i] % 21
                                if (
                                    max(s1 - d1, 0) - act >= 0 and
                                    max(s2 - d2, 0) + act >= 0
                                    ):
                                    r1 = 10 * d1 if self.s1 >= d1 else 10 * s1
                                    r2 = 10 * d2 if self.s2 >= d2 else 10 * s2
                                    ra = abs(act - 1) if act > 0 else abs(act)
                                    re = max(max(max(s1 - d1, 0) - act, 0) - 10, 0)
                                    reward = r1 + r2 - 2 * ra - 2 * re
                                    sum_pd2 += self.poisson_d_2[d2] * reward
                            sum_pa2 += self.poisson_a_2[a2] * sum_pd2
                        sum_pd1 += self.poisson_d_1[d1] * sum_pa2
                    sum_pa1 += self.poisson_a_1[a1] * sum_pd1
                self.r[A][i] += sum_pa1

                            
    def calc_v(self):
        while True:
            delta = 0
            for i in range(21*21):
                v = self.v[i]
                sum_v = self.r[self.pi[i]][i]
                for j in range(21*21):
                    sum_v += self.gamma * self.p[self.pi[i]][i][j] * self.v[j]
                self.v[i] = sum_v
                delta = max(delta, abs(v - self.v[i]))
            if delta < 1e-8:
                break
    
    def calc_pi(self):
        policy_stable = True
        for i in range(21*21):
            old_action = self.pi[i]
            max_v = float("-inf")
            max_a = 0
            for A in range(-5,6):
                a=A+5
                sum_v=self.r[a][i]
                for j in range(21*21):
                    sum_v += self.gamma * self.p[a][i][j] * self.v[j]
                max_a= A if sum_v>=max_v else max_a
            self.pi[i] = max_a
            if old_action != self.pi[i]:
                policy_stable = False
        return policy_stable

    def policy_iter(self):
        policy_stable = False
        while not policy_stable:
            self.calc_v()
            policy_stable = self.calc_pi()


if __name__ == "__main__":
    thing = thingy(3, 3)
    print("reward array:",thing.r)
    print("value array:", thing.v)

    thing2 = Problem2(3, 3, 2, 4)
    print(thing2.v)
    print(thing2.pi)
