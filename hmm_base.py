
import numpy as np
import sklearn.hmm as hmm
import math

def hmm_encode_obs(obs1, obs2):
        obs = []
        for i in range(0, len(obs1)):
                obs.append(obs1[i] + 2 * obs2[i])
        return obs

def hmm_model_fit(obs):
        model = hmm.MultinomialHMM(n_components = 2)
        model.fit([obs])
        return model

def hmm_model_fit_k(obs1, obs2, k):
        obs = hmm_encode_obs(obs1, obs2)
        score = -999999
        d1o = 0.0
        d2o = 0.0
        model = -1
        if 2 not in obs or 3 not in obs:
                return model
        if 0 not in obs:
                obs.append(0)
        if 1 not in obs:
                obs.append(1)
        for i in range(0, k):
                model1 = hmm_model_fit(obs)
                score1 = model1.score(obs)
                # if model1.emissionprob_.shape[1] < 4:
                        # continue
                d1 = model1.emissionprob_[0][1] - model1.emissionprob_[1][1]
                d2 = model1.emissionprob_[0][3] - model1.emissionprob_[1][3]
                if score1 > score and d1 >= 0.20 and d2 >= 0.10:
                           model = model1
                           score = score1
                           d1o = d1
                           d2o = d2
        return model

def max_star(v1, v2):
        return max(v1, v2) + math.log(1 + math.exp(-math.fabs(v1 - v2)))

def log_param(model):
        a = []
        for i in range(0, model.transmat_.shape[0]):
                a.append([])
                for j in range(0, model.transmat_.shape[1]):
                        a[i].append(math.log(model.transmat_[i][j]))
        b = []
        for i in range(0, model.emissionprob_.shape[0]):
                b.append([])
                for j in range(0, model.emissionprob_.shape[1]):
                        b[i].append(math.log(model.emissionprob_[i][j]))
        pi = []
        for i in range(0, len(model.startprob_)):
                pi.append(math.log(model.startprob_[i]))
        ta = model.transmat_
        q0 = ta[1][0] / (ta[1][0] + 1.0 - ta[0][0])
        q1 = 1.0 - q0
        return (a, b, pi, [math.log(q0), math.log(q1)])

def log_param_reduced(model):
        a = []
        for i in range(0, model.transmat_.shape[0]):
                a.append([])
                for j in range(0, model.transmat_.shape[1]):
                        a[i].append(math.log(model.transmat_[i][j]))
        b = []
        for i in range(0, model.emissionprob_.shape[0]):
                b.append([])
                p0 = model.emissionprob_[i][0] + model.emissionprob_[i][2]
                p1 = model.emissionprob_[i][1] + model.emissionprob_[i][3]
                b[i].append(math.log(p0))
                b[i].append(math.log(p1))
                b[i].append(math.log(p0))
                b[i].append(math.log(p1))
        pi = []
        for i in range(0, len(model.startprob_)):
                pi.append(math.log(model.startprob_[i]))
        return (a, b, pi)

def calc_alpha_seq(param, obs, ext):
        alpha = []
        a = param[0]
        b = param[1]
        pi = param[2]
        alpha.append([ pi[0] + b[0][obs[0]], pi[1] + b[1][obs[0]] ])
        for i in range(1, len(obs)):
                a0 = alpha[i - 1][0] + a[0][0]
                a0 = max_star(a0, alpha[i - 1][1] + a[1][0])
                a0 = a0 + b[0][obs[i]] + 0.5 * ext[i - 1][0]
                a1 = alpha[i - 1][0] + a[0][1]
                a1 = max_star(a1, alpha[i - 1][1] + a[1][1])
                a1 = a1 + b[1][obs[i]] - 0.5 * ext[i - 1][0]
                m = max(a0, a1)
                a0 = a0 - m
                a1 = a1 - m
                alpha.append([a0, a1])
        return alpha

def calc_beta_seq(param, obs, ext):
        beta = [[0, 0]]
        a = param[0]
        b = param[1]
        for i in range(len(obs) - 1, 0, -1):
                b0 = a[0][0] + b[0][obs[i]] + beta[-1][0]
                b0 = max_star(b0, a[0][1] + b[1][obs[i]] + beta[-1][1])
                b0 = b0 + 0.5 * ext[i - 1][1]
                b1 = a[1][0] + b[0][obs[i]] + beta[-1][0]
                b1 = max_star(b1, a[1][1] + b[1][obs[i]] + beta[-1][1])
                b1 = b1 - 0.5 * ext[i - 1][1]
                m = max(b0, b1)
                b0 = b0 - m
                b1 = b1 - m
                beta.append([b0, b1])
        beta.reverse()
        return beta

def calc_llr_seq(param, alpha, beta, obs):
        llr = []
        a = param[0]
        b = param[1]
        for i in range(0, len(obs) - 1):
                s00 = alpha[i][0] + a[0][0] + b[0][obs[i + 1]] + beta[i + 1][0]
                s01 = alpha[i][0] + a[0][1] + b[1][obs[i + 1]] + beta[i + 1][1]
                s10 = alpha[i][1] + a[1][0] + b[0][obs[i + 1]] + beta[i + 1][0]
                s11 = alpha[i][1] + a[1][1] + b[1][obs[i + 1]] + beta[i + 1][1]
                fllr = s00 + s10 - s01 - s11
                bllr = s00 + s01 - s10 - s11
                # if fllr > 20.0:
                        # fllr = 20.0
                # elif fllr < -20.0:
                        # fllr = -20.0
                # if bllr > 20.0:
                        # bllr = 20.0
                # elif bllr < -20.0:
                        # bllr = -20.0
                llr.append((fllr, bllr))
        return llr

def calc_ext_seq(llr, llr1):
        ext = []
        for i in range(0, len(llr)):
                ext.append((0.5 * (llr[i][0] - llr1[i][0]), \
                            0.5 * (llr[i][1] - llr1[i][1])))
        return ext

def map_forecast(param, alpha):
        a = param[0]
        b = param[1]
        pi = param[2]
        a0 = alpha[-1][0] + a[0][0]
        a0 = max_star(a0, alpha[-1][1] + a[1][0])
        a1 = alpha[-1][0] + a[0][1]
        a1 = max_star(a1, alpha[-1][1] + a[1][1])
        p0 = a0 + b[0][0]
        p0 = max_star(p0, a0 + b[0][2])
        p0 = max_star(p0, a1 + b[1][0])
        p0 = max_star(p0, a1 + b[1][2])
        p1 = a0 + b[0][1]
        p1 = max_star(p1, a0 + b[0][3])
        p1 = max_star(p1, a1 + b[1][1])
        p1 = max_star(p1, a1 + b[1][3])
        return (p0, p1)

def ml_forecast(param, alpha):
        a = param[0]
        b = param[1]
        q = param[3]
        pp = map_forecast(param, alpha)
        p0 = q[0] + b[0][0]
        p0 = max_star(p0, q[0] + b[0][2])
        p0 = max_star(p0, q[1] + b[1][0])
        p0 = max_star(p0, q[1] + b[1][2])
        p1 = q[0] + b[0][1]
        p1 = max_star(p1, q[0] + b[0][3])
        p1 = max_star(p1, q[1] + b[1][1])
        p1 = max_star(p1, q[1] + b[1][3])
        return (pp[0] - p0, pp[1] - p1)

def alpha_forecast(param, alpha):
        a = param[0]
        b = param[1]
        a0 = alpha[-1][0] + a[0][0]
        a0 = max_star(a0, alpha[-1][1] + a[1][0])
        a1 = alpha[-1][0] + a[0][1]
        a1 = max_star(a1, alpha[-1][1] + a[1][1])
        return (a0, a1)

def hmm_base_predict(param, obs, forecast, corr):
        rlt = []
        ext = [(0.0, 0.0)] * (len(obs) - 1)
        for i in range(0, len(obs) - corr):
                alpha = calc_alpha_seq(param, obs[i:(i + corr)], ext)
                pp = forecast(param, alpha)
                if pp[0] - pp[1] > -math.fabs(pp[0] - pp[1]) * 0.25:
                        rlt.append(0)
                else:
                        rlt.append(1)
        return rlt

def hmm_turbo_predict(model1, model2, obs1, obs2, forecast, corr):
        rlt = []
        # load 2 model parameters
        param1 = log_param(model1)
        param1r = log_param_reduced(model1)
        param2 = log_param(model2)
        param2r = log_param_reduced(model2)
        for i in range(0, len(obs1) - corr):
                ext = [(0.0, 0.0)] * (len(obs1) - 1)
                oo1 = obs1[i:(i + corr)]
                oo2 = obs2[i:(i + corr)]
                # reduction components
                alpha = calc_alpha_seq(param1r, oo1, ext)
                beta = calc_beta_seq(param1r, oo1, ext)
                llr1r = calc_llr_seq(param1r, alpha, beta, oo1)
                alpha = calc_alpha_seq(param2r, oo2, ext)
                beta = calc_beta_seq(param2r, oo2, ext)
                llr2r = calc_llr_seq(param2r, alpha, beta, oo2)
                # iteration
                for j in range(0, 16):
                        # first sub-decoder
                        alpha1 = calc_alpha_seq(param1, oo1, ext)
                        beta = calc_beta_seq(param1, oo1, ext)
                        llr = calc_llr_seq(param1, alpha1, beta, oo1)
                        ext = calc_ext_seq(llr, llr1r)
                        # second sub-decoder
                        alpha2 = calc_alpha_seq(param2, oo2, ext)
                        beta = calc_beta_seq(param2, oo2, ext)
                        llr = calc_llr_seq(param2, alpha2, beta, oo2)
                        ext = calc_ext_seq(llr, llr2r)
                pp1 = forecast(param1, alpha1)
                pp2 = forecast(param2, alpha2)
                # if pp1[0] > pp1[1]:
                if pp1[0] + pp2[0] - pp1[1] - pp2[1] > -math.fabs(pp1[0] + pp2[0] - pp1[1] - pp2[1]) * 0.25:
                        rlt.append(0)
                else:
                        rlt.append(1)
        return rlt




