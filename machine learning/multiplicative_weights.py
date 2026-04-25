"""
Weighted-majority algorithm (a particular kind of multiplicative weights algo.)

Multiplicative weight algorithms are a class of algorithms, where a series of
decisions are made, after each decision the learner receives feedback whether
the decision was correct or not.
In the current file we look at the problem of learning from experts, where one
queries a set of experts and uses the knowledge regarding their past answers
inorder to achieve a prediction which is closest to one of the best expert.

Complexity:
"""
import numpy as np

# TODO: describe the complexity, test the algorithm



class Environment:
    """
    Environment class to represent the outcomes and a set of experts
    """

    def __init__(self, n: int, T: int, prob: float = 0.3):
        self.n = n  # number of experts
        self.T = T  # number of queries
        self.t = 0  # step
        self.outcomes = np.random.choice([0, 1], size=T, p=[1-prob, prob])  # array of size T with 0 or 1 picked
                                                                            # according to the associated probabilities p
        self.p_errors = np.random.random(size=n)        # array of size n with errors between 0.0 and 1.0


    # {np.random.choice([outcome, (outcome + 1)%2], p=[1-p_error, p_error]) for p_error in self.p_errors}

    def step(self):
        """
        Advances the counter and returns the next outcome.
        """
        self.t += 1
        return self.outcomes[self.t-1]

    def get_outcomes(self):
        if self.t < self.T:
            raise RuntimeError('Query process has not been completed.')
        return self.outcomes



class Expert:
    def __init__(self, env: Environment, error: float = 0.5):
        self.env = env
        self.error = np.random.rand()

    def answer(self, t):
        outcome = env.outcomes[t]
        return np.random.choice([outcome, (outcome + 1)%2], p=[1-self.error, self.error])

def errors(answers: list, outcomes: list ) -> float:
    """
    Returns the total number of errors
    """
    return np.sum(np.abs(np.array(answers) - np.array(outcomes)))




def weighted_majority(
        T: int,
        n: int,
        gamma: float,
        env: Environment,
) -> float:
    """
    Predicts the outcome, based on a weighted majority decision.

    Args:
        E (set[Expert]): set of experts
        T (int): interation number
        n (int): number of experts
        gamma (float): learning rate

    Returns:
        p (list): predictions
        o (list): outcomes
    """
    p = []      # predictions list
    o = []      # outcomes list
    w = np.zeros((T,n)).astype(float)
    E = {Expert(env) for _ in range(n)}
    # initialize weights
    for i in range(n):
        w[0,i] = 1
    for t in range(T-1):
        answers = np.array([e.answer(t) for e in E])
        U = np.where(answers == 1)
        up_weight =  np.sum(w[U])
        D = np.where(answers == 0)
        down_weight = np.sum(w[D])
        if up_weight > down_weight:
            p.append(1)
        else:
            p.append(0)
        outcome = env.step()  # the environment returns a result
        o.append(outcome)
        # if p[t] != outcome of the algorithm makes a mistake
        for i in range(n):
            if p[t] != outcome:
                w[t+1,i] = (1-gamma)*w[t,i]
            else:
                w[t+1,i] = w[t,i]
    return p, o




if __name__ == "__main__":
    n = 10   # number of experts
    T = 100  # number of queries
    gamma = 0.1
    env = Environment(n, T)
    p, o = weighted_majority(T, n, gamma, env)
    total_errors = errors(p, o)
    print(total_errors)




