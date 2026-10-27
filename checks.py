"""Small mathematical checks independent of the large experiment."""
import numpy as np
import pandas as pd
from scipy import sparse
from models import IALS, CustomIALS, EASE, ItemKNN, FM
from pipeline import evaluate, evaluation_context


def run_checks():
    rng = np.random.default_rng(7)
    events = pd.DataFrame({'user_idx':[0,0,1,1,2,2],
                          'item_idx':[0,1,1,2,0,3],
                          'rating':[5.,2.,4.,5.,1.,4.],
                          'timestamp':[10,20,30,40,50,60]})
    x = sparse.csr_matrix((np.ones(6), (events.user_idx,events.item_idx)),shape=(3,4))
    model = IALS(factors=2, alpha=3, reg=.7)
    targets = sparse.csr_matrix(((events.rating >= 4).astype(float),
        (events.user_idx, events.item_idx)), shape=x.shape)
    y = rng.normal(size=(4,2))
    w = model._step(y, x, targets)
    # A numerical derivative of the DENSE weighted objective must vanish at
    # the sparse ALS update, including observed p=0 and missing c=1 pairs.
    c = 1 + 3 * x.toarray()
    p = targets.toarray()
    def objective(z):
        return np.sum(c * (p - z @ y.T) ** 2) + .7 * np.sum(z ** 2)
    for idx in np.ndindex(w.shape):
        plus, minus = w.copy(), w.copy()
        plus[idx] += 1e-5
        minus[idx] -= 1e-5
        assert abs((objective(plus) - objective(minus)) / 2e-5) < 1e-5
    custom = CustomIALS(tau_days=1)
    t = custom.targets(events, cutoff=100)
    assert t[1] < 0 and t[0] > 0 and not np.array_equal(t, model.targets(events,100))
    ease = EASE(reg=2).fit(x)
    assert np.allclose(np.diag(ease.weights),0)
    gram = (x.T @ x).toarray()
    residual = (gram + 2 * np.eye(4)) @ ease.weights - gram
    np.fill_diagonal(residual,0)
    assert np.max(np.abs(residual)) < 1e-10
    knn = ItemKNN(k=2,shrink=1).fit(x)
    assert np.all(knn.weights.diagonal()==0)
    fm = FM(epochs=1, batch_size=2).fit(x,events,np.array([[4,6],[5,7],[4,7],[5,6]]))
    users = np.repeat(np.arange(3),4)
    ids = np.tile(np.arange(4),3)
    feats = np.column_stack([users,fm.item_features[ids]])
    direct = fm._forward(feats)[0]
    assert np.allclose(fm.score(np.arange(3)).ravel(),direct)
    # Independent finite-difference check of the FM pairwise derivative.
    xp, xn = feats[:1], feats[1:2]
    sp,vp,sump = fm._forward(xp)
    sn,vn,sumn = fm._forward(xn)
    from scipy.special import expit
    g = -expit(-(sp-sn))[0]
    f, dim = int(xp[0,1]), 0
    analytic = g * np.sum((sump[:,None,:]-vp)[xp==f,dim])
    analytic -= g * np.sum((sumn[:,None,:]-vn)[xn==f,dim])
    original = fm.v[f,dim]
    fm.v[f,dim] = original + 1e-6
    plus = np.logaddexp(0, -(fm._forward(xp)[0]-fm._forward(xn)[0]))[0]
    fm.v[f,dim] = original - 1e-6
    minus = np.logaddexp(0, -(fm._forward(xp)[0]-fm._forward(xn)[0]))[0]
    fm.v[f,dim] = original
    assert np.isclose((plus-minus)/2e-6,analytic,atol=1e-7)
    # An already-seen highest-scoring item must never enter top-10.
    past = pd.DataFrame({'user_idx':[0]*11,'item_idx':list(range(11)), 'rating':[5]*11})
    held = pd.DataFrame({'user_idx':[0],'item_idx':[12],'rating':[5]})
    # Make all items warm through a different user's history.
    past = pd.concat([past,pd.DataFrame({'user_idx':[1]*22,'item_idx':range(22),'rating':[5]*22})])
    ctx = evaluation_context(past,held,(2,22))
    class Fixed:
        def score(self,users):
            return np.tile(-np.arange(22,dtype=float),(len(users),1))
    metrics, _ = evaluate(Fixed(),ctx)
    assert np.isclose(metrics['NDCG@10'],1/np.log2(3))
    assert metrics['Recall@10']==1
    print('Passed: weighted ALS optimum, custom target, EASE stationarity, KNN constraints, FM gradient/scoring, ranking and exclusion.')


if __name__ == '__main__':
    run_checks()
