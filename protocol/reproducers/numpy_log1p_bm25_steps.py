import hashlib, sys
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
with np.load(sys.argv[1]) as d:
    train, n_users, n_items = d["train"], len(d["user_ids"]), len(d["item_ids"])
pos = train[train[:, 2] >= 4]
X = coo_matrix(csr_matrix((np.ones(len(pos), dtype=np.float32), (pos[:, 0], pos[:, 1])), shape=(n_users, n_items)).T.tocsr())
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]
N = float(X.shape[0]); counts = np.bincount(X.col)
step = {"bincount": counts, "log(N)": np.asarray(np.log(N)), "log1p": np.log1p(counts)}
step["idf"] = np.log(N) - np.log1p(counts)
row_sums = np.ravel(X.sum(axis=1)); step["row_sums"] = row_sums
step["mean"] = np.asarray(row_sums.mean())
length_norm = 0.2 + 0.8 * row_sums / row_sums.mean(); step["length_norm"] = length_norm
step["final"] = X.data * 101.0 / (100 * length_norm[X.row] + X.data) * step["idf"][X.col]
print(np.__version__, {k: (v.dtype.name, h(v)) for k, v in step.items()})
# targeted: same input, log1p alone
x = counts.astype(np.float64)
np.save(f"/w/log1p_{np.__version__}.npy", np.log1p(x)); np.save("/w/counts.npy", x)
