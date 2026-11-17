import hashlib, sys
import numpy as np
import implicit.nearest_neighbours as nn
from scipy.sparse import csr_matrix
old = np.load("/w/log1p_1.26.4.npy")
real = nn.log1p
nn.log1p = lambda values: old if np.array_equal(values, np.load("/w/counts.npy")) else real(values)
with np.load(sys.argv[1]) as d:
    train, n_users, n_items = d["train"], len(d["user_ids"]), len(d["item_ids"])
pos = train[train[:, 2] >= 4]
X = csr_matrix((np.ones(len(pos), dtype=np.float32), (pos[:, 0], pos[:, 1])), shape=(n_users, n_items))
W = nn.bm25_weight(X.T.tocsr(), K1=100, B=0.8).tocsr(); W.sort_indices()
print(np.__version__, "with 1.26.4 log1p values:", hashlib.sha256(W.data.tobytes() + W.indices.tobytes() + W.indptr.tobytes()).hexdigest()[:12])
