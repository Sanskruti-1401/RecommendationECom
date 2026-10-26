"""Sequential recommendation engine: synthetic data, models, and offline evaluation."""
import math, random
from collections import defaultdict

CATS = ["Phones", "Phone Acc.", "Laptops", "Laptop Acc.", "Fashion", "Home"]
NAMES = [["iPhone 15", "Galaxy S24", "Pixel 8", "OnePlus 12", "Redmi Note 13"],
         ["Phone Case", "Screen Guard", "Fast Charger", "Earbuds", "Power Bank"],
         ["MacBook Air", "Dell XPS 13", "ThinkPad E14", "HP Pavilion", "Asus Vivobook"],
         ["Laptop Sleeve", "Wireless Mouse", "Mech Keyboard", "USB-C Hub", "Laptop Stand"],
         ["Sneakers", "Denim Jeans", "Cotton T-Shirt", "Backpack", "Sunglasses"],
         ["Air Fryer", "Mixer Grinder", "Bedsheet Set", "Desk Lamp", "Water Bottle"]]
ITEMS = [(n, c, k) for c, g in enumerate(NAMES) for k, n in enumerate(g)]  # (name, category, index)
N = len(ITEMS)
CM = [[.30, .55, 0, 0, .05, .10], [.10, .60, 0, 0, .10, .20], [0, 0, .25, .60, .05, .10],
      [0, 0, .10, .60, .10, .20], [.05, 0, .05, 0, .65, .25], [.05, .05, .05, .05, .20, .60]]
START, POP = [.2, .05, .2, .05, .25, .25], [1, .8, .6, .45, .3]


def _pick(r, w):
    return r.choices(range(len(w)), weights=w)[0]


def generate(n, seed=42):
    """Category funnels (phone -> case -> guard) + a hidden per-user taste that persists in a session."""
    r, out = random.Random(seed), []
    for _ in range(n):
        taste, length = r.randrange(5), 3 + r.randrange(4) + r.randrange(3)
        s, c, g = [], _pick(r, START), 0
        while len(s) < length and g < 60:
            g += 1
            w = [0 if j in s or it[1] != c else POP[it[2]] * (4 if it[2] == taste else 1) for j, it in enumerate(ITEMS)]
            if s:
                lc = ITEMS[s[-1]][1]
                if lc == 0 and c == 1: w[5] *= 2; w[6] *= 2
                if lc == 2 and c == 3: w[15] *= 2; w[16] *= 2
            if not any(w):
                c = _pick(r, CM[c]); continue
            s.append(_pick(r, w)); c = _pick(r, CM[c])
        out.append(s)
    return out


def _norm(a):
    s = sum(a) or 1.0
    return [v / s for v in a]


class Recommender:
    MODELS = ["Autoencoder", "Markov Chain", "RNN", "Transformer", "Hybrid"]

    def fit(self, sessions):
        self.pop = [0] * N
        self.co = [[0] * N for _ in range(N)]
        self.t1 = [[0] * N for _ in range(N)]
        self.t2 = defaultdict(lambda: [0] * N)
        hidden = max(4, min(12, N // 3))
        rng = random.Random(7)
        self.ae = [[(rng.random() - .5) * .1 for _ in range(hidden)] for _ in range(N)]
        self.rnn = [[(rng.random() - .5) * .1 for _ in range(hidden)] for _ in range(N)]
        for s in sessions:
            for i, a in enumerate(s):
                self.pop[a] += 1
                for j, b in enumerate(s):
                    if i != j: self.co[a][b] += 1
                if i > 0:
                    self.t1[s[i - 1]][a] += 1
                    if i > 1: self.t2[(s[i - 2], s[i - 1])][a] += 1
        self._fit_autoencoder(sessions)
        self._fit_rnn(sessions)
        return self

    def _fit_autoencoder(self, sessions):
        """Train a tied-weight session autoencoder without external dependencies."""
        rate = .08
        for _ in range(12):
            for session in sessions:
                hidden = [sum(self.ae[item][h] for item in session) for h in range(len(self.ae[0]))]
                hidden = [math.tanh(value) for value in hidden]
                output = [sum(hidden[h] * self.ae[item][h] for h in range(len(hidden)))
                          for item in range(N)]
                for item in range(N):
                    error = (1 if item in session else 0) - output[item]
                    for h, value in enumerate(hidden):
                        self.ae[item][h] += rate * error * value / len(session)

    def _fit_rnn(self, sessions):
        """Fit a compact recurrent state using item embeddings and next-item targets."""
        for session in sessions:
            state = [0.0] * len(self.rnn[0])
            for current, target in zip(session, session[1:]):
                state = [math.tanh(.7 * value + .3 * self.rnn[current][h])
                         for h, value in enumerate(state)]
                for h, value in enumerate(state):
                    self.rnn[target][h] += .02 * value

    def scores(self, model, p):
        p1 = _norm([v + .01 for v in self.t1[p[-1]]])
        row = self.t2.get((p[-2], p[-1])) if len(p) > 1 else None
        m2 = [.6 * a + .4 * b for a, b in zip(_norm(row), p1)] if row else p1  # back off to order 1
        if model == "Markov Chain":
            return [.5 * a + .5 * b for a, b in zip(m2, p1)]
        if model == "Autoencoder":
            hidden = [math.tanh(sum(self.ae[item][h] for item in p))
                      for h in range(len(self.ae[0]))]
            return _norm([max(0.0, sum(hidden[h] * self.ae[item][h]
                                      for h in range(len(hidden)))) for item in range(N)])
        if model == "RNN":
            state = [0.0] * len(self.rnn[0])
            for current in p:
                state = [math.tanh(.7 * value + .3 * self.rnn[current][h])
                         for h, value in enumerate(state)]
            return _norm([max(0.0, sum(state[h] * self.rnn[item][h]
                                      for h in range(len(state)))) for item in range(N)])
        if model == "Transformer":
            # Causal attention over the prefix, with recent items weighted more heavily.
            sc = [0.0] * N
            for i, current in enumerate(p):
                weight = (i + 1) / len(p)
                for item in range(N):
                    sc[item] += weight * self.co[current][item] / math.sqrt(
                        self.pop[current] * self.pop[item] or 1)
            return _norm(sc)
        if model == "Hybrid":
            ae = self.scores("Autoencoder", p)
            transformer = self.scores("Transformer", p)
            return [.4 * a + .35 * b + .25 * c for a, b, c in zip(m2, ae, transformer)]
        raise ValueError(model)

    def recommend(self, model, p, n=5):
        sc = self.scores(model, p)
        return sorted(((j, sc[j]) for j in range(N) if j not in p), key=lambda x: -x[1])[:n]


def evaluate(rec, test):
    """Leave-last-item-out: predict each test session's final item from the items before it."""
    out = {}
    for m in rec.MODELS:
        ranks, lists = [], []
        for s in test:
            order = [j for j, _ in rec.recommend(m, s[:-1], N)]
            ranks.append(order.index(s[-1]) + 1); lists.append(order[:10])
        out[m] = {"ranks": ranks, "lists": lists}
    return out


def metrics(r, K, idx=None):
    idx = range(len(r["ranks"])) if idx is None else idx
    R, L = [r["ranks"][i] for i in idx], [r["lists"][i] for i in idx]
    n = len(R) or 1
    return {"hr": sum(x <= K for x in R) / n,
            "ndcg": sum(1 / math.log2(x + 1) for x in R if x <= K) / n,
            "mrr": sum(1 / x for x in R if x <= 10) / n,
            "coverage": len({j for l in L for j in l[:K]}) / N}
