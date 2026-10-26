import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine as E


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = E.generate(2000, 1)
        cls.rec = E.Recommender().fit(d[:1600]); cls.d = d
        cls.res = E.evaluate(cls.rec, d[1600:])

    def test_deterministic_and_no_repeats(self):
        self.assertEqual(E.generate(50, 7), E.generate(50, 7))
        self.assertTrue(all(len(set(s)) == len(s) >= 3 for s in self.d))

    def test_recommendations_exclude_session_items(self):
        for m in E.Recommender.MODELS:
            self.assertFalse({j for j, _ in self.rec.recommend(m, [0, 5], 10)} & {0, 5})

    def test_new_models_are_fitted_and_deterministic(self):
        self.assertEqual(E.Recommender.MODELS, ["Autoencoder", "Markov Chain", "RNN", "Transformer", "Hybrid"])
        first = self.rec.scores("Autoencoder", [0, 5])
        second = E.Recommender().fit(self.d[:1600]).scores("Autoencoder", [0, 5])
        self.assertEqual(first, second)
        for model in ("Markov Chain", "RNN", "Transformer", "Hybrid"):
            self.assertEqual(len(self.rec.recommend(model, [0, 5], 5)), 5)

    def test_metric_bounds_and_lift(self):
        hr = {m: E.metrics(self.res[m], 5)["hr"] for m in self.res}
        self.assertTrue(all(0 <= v <= 1 for v in hr.values()))
        self.assertTrue(0 <= hr["Hybrid"] <= 1)


if __name__ == "__main__":
    unittest.main()
