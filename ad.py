
import os, sys
import data2graphs as dg
import proximity as px
import hmm_base as hb

sys.path.append(os.getcwd())

k = 3 - 1
core = range(0, 155)
t1 = 0.08
t2 = 0.04
corr = range(0, 420, 30)

print 'build graphs ...'
graphs = dg.data2graphs('enron-cleaned.edges.txt', 978313080, 86400, 1500000000)

print 'build k graphs ...'
kgraphs = dg.kgraph(graphs, k + 1)

print 'accounting user frequency ...'
freq = dg.count_pair_freq(graphs[0:160], core)

print 'build target user pairs ...'
targets = [] 
for f in freq:
        if f[2] < 15:
                break
        user1 = f[0]
        user2 = f[1]
        # user1 = 41
        # user2 = 96
        code = dg.get_code_seq(graphs, user1, user2)
        cn = px.measure_across_time(kgraphs, px.common_neighbor, \
                        user1, user2, core)
        # print cn
        for i in range(0, len(cn)):
                if 1.0 * cn[i][0] / cn[i][1] <= t1:
                        cn[i] = 1
                else:
                        cn[i] = 0
        # print cn
        pa = px.measure_across_time(kgraphs, px.prefer_attach, \
                        user1, user2, core)
        for i in range(0, len(pa)):
                if 1.0 * pa[i][0] / pa[i][1] <= t2:
                        pa[i] = 1
                else:
                        pa[i] = 0
        model1 = hb.hmm_model_fit_k(code[k:120 + k], cn[0:120], 100)
        if model1 == -1:
                print 'skip user pair ' + str(user1) + ' ' + str(user2)
                continue
        # print model1.transmat_
        # print model1.emissionprob_
        param1 = hb.log_param(model1)
        obs1 = hb.hmm_encode_obs(code[k:], cn)
        model2 = hb.hmm_model_fit_k(code[k:120 + k], pa[0:120], 100)
        if model2 == -1:
                print 'skip user pair ' + str(user1) + ' ' + str(user2)
                continue
        # print model2.transmat_
        # print model2.emissionprob_
        param2 = hb.log_param(model2)
        obs2 = hb.hmm_encode_obs(code[k:], pa)
        targets.append([user1, user2, code, cn, pa, model1, param1, obs1, \
                model2, param2, obs2])
        print 'user pair ' + str(user1) + ' ' + str(user2)

output = open('ad.csv', 'a')
output.write('cn,pa\n')

print 'anomaly detection'
for m in corr:
        score1 = 0
        score2 = 0
        num = 0
        for t in targets:
                user1 = t[0]
                user2 = t[1]
                model1 = t[5]
                obs1 = t[7]
                model2 = t[8]
                obs2 = t[10]
                print 'corr ' + str(m) + ' user ' + \
                                str(user1) + ' ' + str(user2)
                score1 = score1 + model1.score(obs1[m:(m + 30)])
                score2 = score2 + model2.score(obs2[m:(m + 30)])
                num = num + 1
        output.write(str(score1 / num) + ',' + str(score2 / num) + '\n')

output.close()

