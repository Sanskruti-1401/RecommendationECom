# Next-Item Recommender (full stack)

Sequential e-commerce recommender with a metrics dashboard. No dependencies, Python 3.9+.

    python server.py          # http://localhost:8000
    python tests/test_engine.py

## Architecture
- `engine.py`   synthetic session generator, 5 models (Autoencoder, Markov Chain, RNN, Transformer, Hybrid), leave-last-item-out evaluation (HR@K, NDCG@K, MRR@10, coverage)
- `server.py`   stdlib HTTP server, JSON API, SQLite (`sessions`, `rec_log`)
- `static/index.html`  dashboard (Chart.js via CDN) + live recommendation demo

## API
- `GET  /api/items`
- `GET  /api/dashboard?k=5`  metrics, curves, transition matrix
- `POST /api/generate`       `{"sessions":2500,"seed":42}` regenerate data and retrain
- `POST /api/recommend`      `{"session":[0,5],"model":"Hybrid","n":5}`

## Extending
The `Markov Chain` model combines the order-1 and order-2 transition probabilities with back-off. The dependency-free `Autoencoder`, recurrent neural network (RNN), and causal-attention Transformer-style model run during the normal fit/evaluation flow without new packages. Hybrid combines the Markov Chain, Autoencoder, and Transformer scores.
# RecommendationECom
