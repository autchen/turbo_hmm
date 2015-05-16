
import networkx as nx

def common_neighbor(graph, user1, user2):
        if graph.has_node(user1) and graph.has_node(user2):
                n1 = graph.neighbors(user1);
                n2 = graph.neighbors(user2);
                return len(set(n1).intersection(n2))
        return 0

def jacard(graph, user1, user2):
        if graph.has_node(user1) and graph.has_node(user2):
                n1 = graph.neighbors(user1);
                n2 = graph.neighbors(user2);
                len1 = len(set(n1).intersection(n2))
                len2 = len(set(n1).union(n2))
                return 1.0 * len1 / len2
        return 0

def prefer_attach(graph, user1, user2):
        if graph.has_node(user1) and graph.has_node(user2):
                n1 = graph.neighbors(user1);
                n2 = graph.neighbors(user2);
                return len(n1) * len(n2)
        return 0

def rank(graph, measure, user1, user2, core):
        mk = measure(graph, user1, user2)
        if mk == 0:
                return [1, 1]
        rk = 0
        num = 0
        for i in range(0, len(core)):
                for j in range(i + 1, len(core)):
                        m = measure(graph, core[i], core[j])
                        if m > mk:
                                rk = rk + 1
                        if m > 0:
                                num = num + 1
        return [rk, num]

def measure_across_time(graphs, measure, user1, user2, core):
        mm = []
        for gc in graphs:
                m = rank(gc[0], measure, user1, user2, core)
                mm.append(m)
        return mm

def proximity_predict(measure, thresh):
        predict = []
        for i in range(0, len(measure)):
                if measure[i] >= thresh:
                        predict.append(1)
                else:
                        predict.append(0)
        return predict


import data2graphs as dg

def rank_predict(graphs, kgraphs, measure, targets, core, thresh):
        for tt in targets:
                mm = measure_across_time(kgraphs, measure, tt[0], tt[1], core)
                cc = dg.get_code_seq(graphs, tt[0], tt[1])
                cc = cc[(len(cc) - len(mm) + 1):len(cc)]
                pd = []
                for i in range(0, len(mm) - 1):
                        if mm[i] >= thresh:
                                pd.append(1)
                        else:
                                pd.append(0)
                rlt = dg.account_result(cc, pd)
                print rlt


