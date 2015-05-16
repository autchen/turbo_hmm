
import os, sys
import data2graphs as dg
import proximity as px
import hmm_base as hb

sys.path.append(os.getcwd())

k = 3 - 1
core = range(0, 155)
t1 = 0.08
t2 = 0.04
corr = range(120, 130, 10)

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
        obs1 = hb.hmm_encode_obs(code[k:150 + k], cn[0:150])
        model2 = hb.hmm_model_fit_k(code[k:120 + k], pa[0:120], 100)
        if model2 == -1:
                print 'skip user pair ' + str(user1) + ' ' + str(user2)
                continue
        # print model2.transmat_
        # print model2.emissionprob_
        param2 = hb.log_param(model2)
        obs2 = hb.hmm_encode_obs(code[k:150 + k], pa[0:150])
        targets.append([user1, user2, code, cn, pa, model1, param1, obs1, \
                model2, param2, obs2])
        print 'user pair ' + str(user1) + ' ' + str(user2)
        print model1.transmat_, model1.emissionprob_
        print model2.transmat_, model2.emissionprob_

output = open('map.csv', 'a')
output.write('corr,recall,precision,ber\n')
output.write(',turbo,,,,hmm1,,,,hmm2,,,,prox1,,,,prox2\n')

print 'predict map'
for m in corr:
        acc = [([0] * 6), [0] * 6, [0] * 6, [0] * 6, [0] * 6]
        for t in targets:
                rlt = []
                user1 = t[0]
                user2 = t[1]
                code = t[2]
                cn = t[3]
                pa = t[4]
                model1 = t[5]
                param1 = t[6]
                obs1 = t[7]
                model2 = t[8]
                param2 = t[9]
                obs2 = t[10]
                print 'corr ' + str(m) + ' user ' + \
                                str(user1) + ' ' + str(user2)
                pred = hb.hmm_turbo_predict(model1, model2, \
                                obs1[(120 - m):150], obs2[(120 - m):150], \
                                hb.map_forecast, m)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                pred = hb.hmm_base_predict(param1, obs1[(120 - m):150], \
                                hb.map_forecast, m)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                pred = hb.hmm_base_predict(param2, obs2[(120 - m):150], \
                                hb.map_forecast, m)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                pred = px.proximity_predict(cn[119:149], 1)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                pred = px.proximity_predict(pa[119:149], 1)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                print rlt
                for i in range(0, 5):
                        for j in range(0, 6):
                                acc[i][j] = acc[i][j] + rlt[i][j]
        output.write(str(m) + ',')
        print acc
        for i in range(0, 5):
                if acc[i][1] != 0:
                        r = 1.0 * acc[i][0] / acc[i][1]
                else:
                        r = 0
                if acc[i][3] != 0:
                        p = 1.0 * acc[i][0] / acc[i][3]
                else:
                        p = 0
                if acc[i][5] != 0:
                        b = 1.0 * acc[i][4] / acc[i][5]
                else:
                        b = 0
                output.write(str(r) + ',' + str(p) + ',' + str(b) + ',,')
        output.write('\n')

output.close()

output = open('ml.csv', 'a')
output.write('corr,recall,precision,ber\n')
output.write(',turbo,,,,hmm1,,,,hmm2,,,,prox1,,,,prox2\n')

print 'predict ml'
for m in corr:
        acc = [([0] * 6), [0] * 6, [0] * 6, [0] * 6, [0] * 6]
        for t in targets:
                rlt = []
                user1 = t[0]
                user2 = t[1]
                code = t[2]
                cn = t[3]
                pa = t[4]
                model1 = t[5]
                param1 = t[6]
                obs1 = t[7]
                model2 = t[8]
                param2 = t[9]
                obs2 = t[10]
                print 'corr ' + str(m) + ' user ' + \
                                str(user1) + ' ' + str(user2)
                pred = hb.hmm_turbo_predict(model1, model2, \
                                obs1[(120 - m):150], obs2[(120 - m):150], \
                                hb.ml_forecast, m)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                pred = hb.hmm_base_predict(param1, obs1[(120 - m):150], \
                                hb.ml_forecast, m)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                pred = hb.hmm_base_predict(param2, obs2[(120 - m):150], \
                                hb.ml_forecast, m)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                pred = px.proximity_predict(cn[119:149], 1)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                pred = px.proximity_predict(pa[119:149], 1)
                rlt.append(dg.account_result(code[120 + k:150 + k], pred))
                print rlt
                for i in range(0, 5):
                        for j in range(0, 6):
                                acc[i][j] = acc[i][j] + rlt[i][j]
        output.write(str(m) + ',')
        print acc
        for i in range(0, 5):
                if acc[i][1] != 0:
                        r = 1.0 * acc[i][0] / acc[i][1]
                else:
                        r = 0
                if acc[i][3] != 0:
                        p = 1.0 * acc[i][0] / acc[i][3]
                else:
                        p = 0
                if acc[i][5] != 0:
                        b = 1.0 * acc[i][4] / acc[i][5]
                else:
                        b = 0
                output.write(str(r) + ',' + str(p) + ',' + str(b) + ',,')
        output.write('\n')

output.close()

