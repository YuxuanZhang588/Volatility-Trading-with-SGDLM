#region imports
from AlgorithmImports import *
#endregion
# Use training data to get a informative prior for the testing data

import numpy as np
from scipy.stats import multivariate_normal, truncnorm, invgamma


def DLM_train(y, b, v, FF, G, m0, C0, delta):
    p = FF.shape[1]
    theta_sample = np.zeros((len(y), p))
    TT = len(y)
    f = np.zeros(TT)
    Q = np.zeros(TT)
    e = np.zeros(TT)
    a = np.zeros((TT, p))
    #a_back = np.zeros((TT + K, p))
    m = np.zeros((TT, p))
    A = np.zeros((TT, p))
    h = np.zeros((TT, p))
    R = np.full((TT, p, p), np.nan)
    #R_back = np.full((TT + K, p, p), np.nan)
    W = np.full((TT, p, p), np.nan)
    C = np.full((TT, p, p), np.nan)
    B = np.full((TT, p, p), np.nan)
    H = np.full((TT, p, p), np.nan)
    

    t = 0
    a[t] = G @ m0
    W[t] = (1 - delta) * G @ C0 @ G.T / delta
    R[t] = G @ C0 @ G.T + W[t]
    f[t] = FF[t] @ a[t]
    Q[t] = FF[t].T @ R[t] @ FF[t] + v[t]
    e[t] = y[t] - f[t]
    A[t] = R[t] @ FF[t] * (Q[t])**(-1)
    m[t] = a[t] + A[t] * e[t]
    C[t] = R[t] - np.outer(A[t], A[t]) * Q[t]

    for t in range(1, TT):
        a[t] = G @ m[t-1]
        W[t] = (1 - delta) * G @ C[t-1] @ G.T / delta
        R[t] = G @ C[t-1] @ G.T + W[t]
        f[t] = FF[t] @ a[t]
        Q[t] = FF[t].T @ R[t] @ FF[t] + v[t]
        e[t] = y[t] - f[t]
        A[t] = R[t] @ FF[t]/Q[t]
        m[t] = a[t] + A[t] * e[t]
        C[t] = R[t] - np.outer(A[t], A[t]) * Q[t] 

    theta_sample[TT-1] = multivariate_normal.rvs(a[TT-1], R[TT-1])
    for t in range(TT-2, -1, -1):
        B[t] = C[t] @ G.T @ np.linalg.inv(R[t+1])
        h[t] = m[t] + B[t] @ (theta_sample[t+1] - a[t+1])
        H[t] = C[t] - B[t] @ R[t+1] @ B[t].T
        theta_sample[t] = multivariate_normal.rvs(h[t], H[t])   

    return {'theta_sample': theta_sample}

#out = SV(tmp, J, q, b, w, u, g, GG, c, CC, aa, v0, gamma_sample[i-1], phi_sample[i-1], mu_sample[i-1], v_sample[i-1], x_sample[i-1], K)

def SV_train(y, J, q, b, w, u=1, g=0, GG=1, c=0, CC=1, aa=2, v0=1, gamma0=None, phi0=None, mu0=None, v00=None, x0=None):
    TT = len(y)
    gamma_sample = np.zeros((2, TT))
    x_sample = np.zeros((2, TT + 1))
    mu_sample = np.zeros(2)
    phi_sample = np.zeros(2)
    v_sample  = np.zeros(2)
    mu_sample[0] = mu0
    x_sample[0] = x0
    phi_sample[0] = phi0
    v_sample[0] = v00
    gamma_sample[0] = gamma0

    for i in range(1, 2):
        B = np.sum(x_sample[i-1, :-1]**2)
        bb = np.sum(x_sample[i-1, 1:] * x_sample[i-1, :-1]) / B
        r = v_sample[i-1] / (v_sample[i-1] + CC * B)
        cp = r * c + (1 - r) * bb
        Cp = r * CC
        phi_sample[i] = truncnorm.rvs(0, 1, loc=cp, scale=np.sqrt(Cp))
#        v_sample[i] = invgamma.rvs((aa + TT) / 2, ((aa * v0 + np.sum((x_sample[i-1, 1:] - phi_sample[i] * x_sample[i-1, :-1])**2)) / 2)**(-1))
        v_sample[i] = invgamma.rvs((aa + TT) / 2, scale=(0.5*(aa * v0 + np.sum((x_sample[i-1, 1:] - phi_sample[i] * x_sample[i-1, :-1])**2))))

        Winv = np.sum(1 / w[gamma_sample[i-1,:].astype(int)])
        W = 1 / Winv
        mu_hat = W * np.sum(1 / w[gamma_sample[i-1,:].astype(int)] * (y - x_sample[i-1, 1:] - b[gamma_sample[i-1,:].astype(int)]))
        r = W / (W + GG)
        mu_sample[i] = np.random.normal((1-r)*mu_hat + r*g, np.sqrt(r*GG))

        m = np.zeros(TT)
        M = np.zeros(TT)
        e = np.zeros(TT)
        a = np.zeros(TT)
        A = np.zeros(TT)
        h = np.zeros(TT)
        t = 0
        h[t] = v_sample[i] + phi_sample[i]**2 * u
        a[t] = 0
        A[t] = h[t] / (h[t] + w[gamma_sample[i-1, t].astype(int)])
        e[t] = y[t] - mu_sample[i] - b[gamma_sample[i-1, t].astype(int)] - a[t]
        m[t] = a[t] + A[t] * e[t]
        M[t] =  w[gamma_sample[i-1, t].astype(int)]*A[t]

        for t in range(1, TT):
            h[t] = v_sample[i] + phi_sample[i]**2 * M[t-1]
            a[t] = phi_sample[i] * m[t-1]
            A[t] = h[t] / (h[t] + w[gamma_sample[i-1, t].astype(int)])
            e[t] = y[t] - mu_sample[i] - b[gamma_sample[i-1, t].astype(int)] - a[t]
            m[t] = a[t] + A[t] * e[t]
            M[t] =  w[gamma_sample[i-1, t].astype(int)]*A[t]

        x_sample[i, -1] = np.random.normal(m[-1], np.sqrt(M[-1]))
        for t in range(TT-2, -1, -1):
            mback = m[t] + phi_sample[i] * M[t] * (x_sample[i, t+2] - a[t+1]) / h[t+1]
            Vback = v_sample[i] * M[t] / h[t+1]
            x_sample[i, t+1] = np.random.normal(mback, np.sqrt(Vback))
        x_sample[i, 0] = np.random.normal(phi_sample[t] * u * (x_sample[i, 1] - a[0]) / h[0], np.sqrt(v_sample[i] * u / h[0]))

        for t in range(TT):
            Pr = np.zeros(J)
            for j in range(J):
                Pr[j] = q[j] * np.exp(-(y[t] - mu_sample[i] - b[j] - x_sample[i, t+1])**2 / (2 * w[j])) / np.sqrt(w[j])
            gamma_sample[i, t] = np.random.choice(np.arange(0, J), p=Pr/np.sum(Pr))

    return {'mu_sample': mu_sample[1],
            'phi_sample': phi_sample[1],
            'v_sample': v_sample[1],
            'gamma_sample': gamma_sample[1],
            'x_sample': x_sample[1]}

def training(r_train, FF, G, m0, C0, delta=0.99, J=None, q=None, b=None, w=None, u=1, g=0, GG=1, c=0, CC=1, aa=2, v0=1, n_burn_in=None, npost=None):
    TT = len(r_train)
    y = np.log(r_train**2) / 2
    p = FF.shape[1]
    mu_sample = np.zeros(npost + n_burn_in)
    phi_sample = np.zeros(npost + n_burn_in)
    v_sample = np.zeros(npost + n_burn_in)
    gamma_sample = np.zeros((npost + n_burn_in, TT))
    x_sample = np.zeros((npost + n_burn_in, TT + 1))
    theta_sample = np.full((npost + n_burn_in, TT, p), np.nan)

    mu_sample[0] = np.random.normal(g, np.sqrt(GG))
    phi_sample[0] = truncnorm.rvs(0, 1, loc=c, scale=np.sqrt(CC))
#    v_sample[0] = invgamma.rvs(aa/2, scale=(aa*v0/2)**(-1))
    v_sample[0] = invgamma.rvs(aa/2, scale=(aa*v0/2))
    x_sample[0, 0] = np.random.normal(0, np.sqrt(u))
    for t in range(TT):
        x_sample[0, t+1] = np.random.normal(phi_sample[0] * x_sample[0, t], np.sqrt(v_sample[0]))

    i = 0
    for t in range(TT):
        Pr = np.zeros(J)
        for j in range(J):
            Pr[j] = q[j] * np.exp(-(y[t] - mu_sample[i] - b[j] - x_sample[i, t])**2 / (2 * w[j])) / np.sqrt(w[j])
        gamma_sample[i, t] = np.random.choice(np.arange(0, J), p=Pr/np.sum(Pr))
    
    out = DLM_train(y - mu_sample[i] - x_sample[i, 1:] - b[gamma_sample[i,:].astype(int)],
                    b[gamma_sample[i,:].astype(int)], w[gamma_sample[i,:].astype(int)], FF, G, m0, C0, delta)
    theta_sample[0] = out['theta_sample']

    for i in range(1, npost + n_burn_in):
        tmp = y.copy()
        for t in range(TT):
            tmp[t] = y[t] - theta_sample[i-1, t] @ FF[t]
        out = SV_train(tmp, J, q, b, w, u, g, GG, c, CC, aa, v0, gamma_sample[i-1], phi_sample[i-1], mu_sample[i-1], v_sample[i-1], x_sample[i-1])
        mu_sample[i] = out['mu_sample']
        phi_sample[i] = out['phi_sample']
        v_sample[i] = out['v_sample']
        gamma_sample[i] = out['gamma_sample']
        x_sample[i] = out['x_sample']

        out = DLM_train(y - mu_sample[i] - x_sample[i, 1:] - b[gamma_sample[i,:].astype(int)],
                        b[gamma_sample[i,:].astype(int)], w[gamma_sample[i,:].astype(int)], FF, G, m0, C0, delta)
        theta_sample[i] = out['theta_sample']
        if i % 20 == 0:
            print(i)

    return {'mu_sample': mu_sample[n_burn_in:],
            'phi_sample': phi_sample[n_burn_in:],
            'v_sample': v_sample[n_burn_in:]}



def DLM(y, b, v, FF, G, m0, C0, delta, K):
    p = FF.shape[1]
    theta_sample = np.zeros((len(y) + K, p))
    TT = len(y)
    f = np.zeros(TT + K)
    Q = np.zeros(TT + K)
    e = np.zeros(TT + K)
    a = np.zeros((TT + K, p))
    #a_back = np.zeros((TT + K, p))
    m = np.zeros((TT + K, p))
    A = np.zeros((TT + K, p))
    h = np.zeros((TT + K, p))
    R = np.full((TT + K, p, p), np.nan)
    #R_back = np.full((TT + K, p, p), np.nan)
    W = np.full((TT + K, p, p), np.nan)
    C = np.full((TT + K, p, p), np.nan)
    B = np.full((TT + K, p, p), np.nan)
    H = np.full((TT + K, p, p), np.nan)
    

    t = 0
    a[t] = G @ m0
    W[t] = (1 - delta) * G @ C0 @ G.T / delta
    R[t] = G @ C0 @ G.T + W[t]
    f[t] = FF[t] @ a[t]
    Q[t] = FF[t].T @ R[t] @ FF[t] + v[t]
    e[t] = y[t] - f[t]
    A[t] = R[t] @ FF[t] * (Q[t])**(-1)
    m[t] = a[t] + A[t] * e[t]
    C[t] = R[t] - np.outer(A[t], A[t]) * Q[t]

    for t in range(1, TT):
        a[t] = G @ m[t-1]
        W[t] = (1 - delta) * G @ C[t-1] @ G.T / delta
        R[t] = G @ C[t-1] @ G.T + W[t]
        f[t] = FF[t] @ a[t]
        Q[t] = FF[t].T @ R[t] @ FF[t] + v[t]
        e[t] = y[t] - f[t]
        A[t] = R[t] @ FF[t]/Q[t]
        m[t] = a[t] + A[t] * e[t]
        C[t] = R[t] - np.outer(A[t], A[t]) * Q[t] 

    theta_sample[TT-1] = multivariate_normal.rvs(a[TT-1], R[TT-1])
    for t in range(TT-2, -1, -1):
        B[t] = C[t] @ G.T @ np.linalg.inv(R[t+1])
        h[t] = m[t] + B[t] @ (theta_sample[t+1] - a[t+1])
        H[t] = C[t] - B[t] @ R[t+1] @ B[t].T
        theta_sample[t] = multivariate_normal.rvs(h[t], H[t])

    y_forecast = np.zeros(K)
    for t in range(TT, TT+K):
        a[t] = G @ m[t-1]
        W[t] = (1 - delta) * G @ C[t-1] @ G.T / delta
        theta_sample[t] = G @ theta_sample[t-1] + multivariate_normal.rvs(np.zeros(p), W[t])
        R[t] = G @ C[t-1] @ G.T + W[t]
        f[t] = FF[t] @ a[t]
        Q[t] = FF[t].T @ R[t] @ FF[t] + v[t]
        y_forecast[t-TT] = np.random.normal(f[t], np.sqrt(Q[t])) + b[t]
        e[t] = y_forecast[t-TT] - f[t]
        A[t] = R[t] @ FF[t] * (Q[t])**(-1)
        m[t] = a[t] + A[t] * e[t]
        C[t] = R[t] - np.outer(A[t], A[t]) * Q[t]

    return {'theta_sample': theta_sample, 'y_forecast': y_forecast}

#out = SV(tmp, J, q, b, w, u, g, GG, c, CC, aa, v0, gamma_sample[i-1], phi_sample[i-1], mu_sample[i-1], v_sample[i-1], x_sample[i-1], K)

def SV(y, J, q, b, w, u=1, g=0, GG=1, c=0, CC=1, aa=2, v0=1, gamma0=None, phi0=None, mu0=None, v00=None, x0=None, K=None):
    TT = len(y)
    gamma_sample = np.zeros((2, TT + K))
    x_sample = np.zeros((2, TT + 1))
    mu_sample = np.zeros(2)
    phi_sample = np.zeros(2)
    v_sample  = np.zeros(2)
    mu_sample[0] = mu0
    x_sample[0] = x0
    phi_sample[0] = phi0
    v_sample[0] = v00
    gamma_sample[0] = gamma0

    for i in range(1, 2):
        B = np.sum(x_sample[i-1, :-1]**2)
        bb = np.sum(x_sample[i-1, 1:] * x_sample[i-1, :-1]) / B
        r = v_sample[i-1] / (v_sample[i-1] + CC * B)
        cp = r * c + (1 - r) * bb
        Cp = r * CC
        phi_sample[i] = truncnorm.rvs(0, 1, loc=cp, scale=np.sqrt(Cp))
        v_sample[i] = invgamma.rvs((aa + TT) / 2, ((aa * v0 + np.sum((x_sample[i-1, 1:] - phi_sample[i] * x_sample[i-1, :-1])**2)) / 2)**(-1))
        Winv = np.sum(1 / w[gamma_sample[i-1, :-K].astype(int)])
        W = 1 / Winv
        mu_hat = W * np.sum(1 / w[gamma_sample[i-1, :-K].astype(int)] * (y - x_sample[i-1, 1:] - b[gamma_sample[i-1, :-K].astype(int)]))
        r = W / (W + GG)
        mu_sample[i] = np.random.normal((1-r)*mu_hat + r*g, np.sqrt(r*GG))

        m = np.zeros(TT)
        M = np.zeros(TT)
        e = np.zeros(TT)
        a = np.zeros(TT)
        A = np.zeros(TT)
        h = np.zeros(TT)
        t = 0
        h[t] = v_sample[i] + phi_sample[i]**2 * u
        a[t] = 0
        A[t] = h[t] / (h[t] + w[gamma_sample[i-1, t].astype(int)])
        e[t] = y[t] - mu_sample[i] - b[gamma_sample[i-1, t].astype(int)] - a[t]
        m[t] = a[t] + A[t] * e[t]
        M[t] =  w[gamma_sample[i-1, t].astype(int)]*A[t]

        for t in range(1, TT):
            h[t] = v_sample[i] + phi_sample[i]**2 * M[t-1]
            a[t] = phi_sample[i] * m[t-1]
            A[t] = h[t] / (h[t] + w[gamma_sample[i-1, t].astype(int)])
            e[t] = y[t] - mu_sample[i] - b[gamma_sample[i-1, t].astype(int)] - a[t]
            m[t] = a[t] + A[t] * e[t]
            M[t] =  w[gamma_sample[i-1, t].astype(int)]*A[t]

        x_sample[i, -1] = np.random.normal(m[-1], np.sqrt(M[-1]))
        for t in range(TT-2, -1, -1):
            mback = m[t] + phi_sample[i] * M[t] * (x_sample[i, t+2] - a[t+1]) / h[t+1]
            Vback = v_sample[i] * M[t] / h[t+1]
            x_sample[i, t+1] = np.random.normal(mback, np.sqrt(Vback))
        x_sample[i, 0] = np.random.normal(phi_sample[t] * u * (x_sample[i, 1] - a[0]) / h[0], np.sqrt(v_sample[i] * u / h[0]))

        for t in range(TT):
            Pr = np.zeros(J)
            for j in range(J):
                Pr[j] = q[j] * np.exp(-(y[t] - mu_sample[i] - b[j] - x_sample[i, t+1])**2 / (2 * w[j])) / np.sqrt(w[j])
            gamma_sample[i, t] = np.random.choice(np.arange(0, J), p=Pr/np.sum(Pr))
        gamma_sample[i, TT:] = np.random.choice(np.arange(0, J), size=K, replace=True, p=q)

    return {'mu_sample': mu_sample[1],
            'phi_sample': phi_sample[1],
            'v_sample': v_sample[1],
            'gamma_sample': gamma_sample[1],
            'x_sample': x_sample[1]}

def SV_forecast(r, FF, G, m0, C0, delta=0.99, J=None, q=None, b=None, w=None, u=1, g=0, GG=1, c=0, CC=1, aa=2, v0=1, n_burn_in=None, npost=None, K=None):
    TT = len(r)
    y = np.log(r**2) / 2
    p = FF.shape[1]
    mu_sample = np.zeros(npost + n_burn_in)
    phi_sample = np.zeros(npost + n_burn_in)
    v_sample = np.zeros(npost + n_burn_in)
    gamma_sample = np.zeros((npost + n_burn_in, TT + K))
    x_sample = np.zeros((npost + n_burn_in, TT + 1))
    theta_sample = np.full((npost + n_burn_in, TT + K, p), np.nan)
    y_forecast = np.zeros((npost + n_burn_in, K))
    sigma_forecast = np.zeros((npost + n_burn_in, K))

    mu_sample[0] = np.random.normal(g, np.sqrt(GG))
    phi_sample[0] = truncnorm.rvs(0, 1, loc=c, scale=np.sqrt(CC))
    v_sample[0] = invgamma.rvs(aa/2, (aa*v0/2)**(-1))
    x_sample[0, 0] = np.random.normal(0, np.sqrt(u))
    for t in range(TT):
        x_sample[0, t+1] = np.random.normal(phi_sample[0] * x_sample[0, t], np.sqrt(v_sample[0]))

    i = 0
    for t in range(TT):
        Pr = np.zeros(J)
        for j in range(J):
            Pr[j] = q[j] * np.exp(-(y[t] - mu_sample[i] - b[j] - x_sample[i, t])**2 / (2 * w[j])) / np.sqrt(w[j])
        gamma_sample[i, t] = np.random.choice(np.arange(0, J), p=Pr/np.sum(Pr))
    gamma_sample[i, TT:] = np.random.choice(np.arange(0, J), size=K, replace=True, p=q)

    out = DLM(y - mu_sample[0] - x_sample[0, 1:] - b[gamma_sample[i, :-K].astype(int)],
              b[gamma_sample[i,:].astype(int)],
              w[gamma_sample[i, :].astype(int)],
              FF, G, m0, C0, delta, K)
    theta_sample[0] = out['theta_sample']
    y_forecast[0] = out['y_forecast']

    for i in range(1, npost + n_burn_in):
        tmp = y.copy()
        for t in range(TT):
            tmp[t] = y[t] - theta_sample[i-1, t] @ FF[t]
        out = SV(tmp, J, q, b, w, u, g, GG, c, CC, aa, v0, gamma_sample[i-1], phi_sample[i-1], mu_sample[i-1], v_sample[i-1], x_sample[i-1], K)
        mu_sample[i] = out['mu_sample']
        phi_sample[i] = out['phi_sample']
        v_sample[i] = out['v_sample']
        gamma_sample[i] = out['gamma_sample']
        x_sample[i] = out['x_sample']

        out = DLM(y - mu_sample[i] - x_sample[i, 1:] - b[gamma_sample[i, :-K].astype(int)],
                  b[gamma_sample[i, :].astype(int)],
                  w[gamma_sample[i, :].astype(int)],
                  FF, G, m0, C0, delta, K)
        theta_sample[i] = out['theta_sample']
        y_forecast[i] = out['y_forecast']

        x_forecast = np.zeros(K)
        k = 0
        x_forecast[k] = phi_sample[i] * x_sample[i, -1] + np.random.normal(0, np.sqrt(v_sample[i]))
        sigma_forecast[i, k] = np.exp(mu_sample[i] + x_forecast[k] + theta_sample[i, TT+k] @ FF[TT+k])
        for k in range(1, K):
            x_forecast[k] = phi_sample[i] * x_forecast[k-1] + np.random.normal(0, np.sqrt(v_sample[i]))
            sigma_forecast[i, k] = np.exp(mu_sample[i] + x_forecast[k] + theta_sample[i, TT+k] @ FF[TT+k])
        y_forecast[i] = y_forecast[i] + mu_sample[i] + x_forecast
        if i % 20 == 0:
            print(i)

    return {'mu_sample': mu_sample[n_burn_in:],
            'phi_sample': phi_sample[n_burn_in:],
            'v_sample': v_sample[n_burn_in:],
            'gamma_sample': gamma_sample[n_burn_in:],
            'x_sample': x_sample[n_burn_in:],
            'theta_sample': theta_sample[n_burn_in:],
            'y_forecast': y_forecast[n_burn_in:],
            'sigma_forecast': sigma_forecast[n_burn_in:]}

