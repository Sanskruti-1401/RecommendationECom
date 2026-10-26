"""HTTP server + JSON API + SQLite storage. Zero dependencies: python server.py"""
import json, os, sqlite3, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import engine as E

ROOT = os.path.dirname(os.path.abspath(__file__))
db = sqlite3.connect(os.path.join(ROOT, "recsys.db"), check_same_thread=False)
db.executescript("create table if not exists sessions(id integer primary key, items text);"
                 "create table if not exists rec_log(ts real default (strftime('%s','now')), model text, session text, top text);")
LOCK, S = threading.Lock(), {}


def build(n, seed):
    data, cut = E.generate(n, seed), int(n * .8)
    rec = E.Recommender().fit(data[:cut])
    S.update(data=data, train=data[:cut], test=data[cut:], rec=rec, res=E.evaluate(rec, data[cut:]))
    with LOCK:
        db.execute("delete from sessions")
        db.executemany("insert into sessions(items) values(?)", [(json.dumps(s),) for s in data])
        db.commit()


def dashboard(K):
    ms, res, test = E.Recommender.MODELS, S["res"], S["test"]
    cm = [[0] * 6 for _ in range(6)]
    for s in S["train"]:
        for a, b in zip(s, s[1:]): cm[E.ITEMS[a][1]][E.ITEMS[b][1]] += 1
    lens = sorted({len(s) - 1 for s in test})
    with LOCK:
        served = db.execute("select count(*) from rec_log").fetchone()[0]
    return {"models": ms, "cats": E.CATS, "cm": cm, "served": served,
            "kpi": {"sessions": len(S["data"]), "interactions": sum(map(len, S["data"])), "items": E.N,
                    "train": len(S["train"]), "test": len(test)},
            "board": {m: E.metrics(res[m], K) for m in ms},
            "curve": {m: [E.metrics(res[m], k)["hr"] for k in range(1, 11)] for m in ms},
            "prefix": {"lens": lens, "hr": {m: [E.metrics(res[m], K, [i for i, s in enumerate(test) if len(s) - 1 == L])["hr"]
                                                 for L in lens] for m in ms}}}


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/":
            return self._send(open(os.path.join(ROOT, "static", "index.html"), "rb").read(), ctype="text/html; charset=utf-8")
        if u.path == "/api/items":
            return self._send([{"id": i, "name": n, "cat": E.CATS[c]} for i, (n, c, k) in enumerate(E.ITEMS)])
        if u.path == "/api/dashboard":
            return self._send(dashboard(int(q.get("k", ["5"])[0])))
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/generate":
                build(max(300, min(6000, int(body.get("sessions", 2500)))), int(body.get("seed", 42)))
                return self._send({"ok": True})
            if self.path == "/api/recommend":
                p = [int(x) for x in body.get("session", []) if 0 <= int(x) < E.N]
                m = body.get("model", "Hybrid")
                if not p or m not in E.Recommender.MODELS:
                    return self._send({"error": "need a non-empty session and a valid model"}, 400)
                recs = S["rec"].recommend(m, p, int(body.get("n", 5)))
                with LOCK:
                    db.execute("insert into rec_log(model,session,top) values(?,?,?)",
                               (m, json.dumps(p), json.dumps([j for j, _ in recs]))); db.commit()
                return self._send([{"id": j, "name": E.ITEMS[j][0], "score": s} for j, s in recs])
        except (ValueError, TypeError):
            return self._send({"error": "bad request"}, 400)
        self._send({"error": "not found"}, 404)

    def log_message(self, *a): pass


if __name__ == "__main__":
    build(2500, 42)
    port = int(os.environ.get("PORT", 8000))
    print(f"Dashboard: http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
