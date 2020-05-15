import sys
import os
import random
import math
import time
import json
from pyspark import SparkContext, SparkConf

input_path = sys.argv[1]
n_cluster = int(sys.argv[2])
result_file = sys.argv[3]
intermediate_file = sys.argv[4]

alpha = 4
Header = 'round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained'


def get_dist(point1, point2):
    '''
    :param point1: one point
    :param point2: another point
    :return: Euclidean distance of the two points
    '''

    if point1[0] == point2[0]:
        return 0
    dist = 0
    for i in range(0, len(point1)):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)


def get_centroids(data_points, k):
    '''
    :param data_points: the data set of points
    :param k: the number of clusters
    :return: the centroids of the k clusters
    '''

    current_point = data_points[random.randint(0, len(data_points) - 1)]
    centroids = []
    dist_matrix = []
    for j in range(k):
        max_dist = []
        cur_point_dist = []
        for i in range(len(data_points)):
            dist = get_dist(current_point[1:], data_points[i][1:])
            cur_point_dist.append(dist)
            min_dist = dist
            for point_dist in dist_matrix:
                min_dist = min(min_dist, point_dist[i])
            max_dist.append([i, min_dist])
        dist_matrix.append(cur_point_dist)
        current_point = data_points[sorted(max_dist, key=lambda x: x[1], reverse=True)[0][0]]
        centroids.append(tuple(current_point[1:]))

    return list(set(centroids))


def choose_centroid(point, centroids):
    '''
    used for the point assignment in the kmeans of MapReduce
    :param point: a single point
    :param centroids: a list of centroids from different clusters
    :return: (assigned centroid, (point vector, 1, a list with this point))
    '''

    min_dist = []
    for centroid in centroids:
        dist = get_dist(point[1:], centroid)
        min_dist.append([centroid, dist])
    assigned_centroid = sorted(min_dist, key=lambda x: x[1], reverse=False)[0][0]
    return (assigned_centroid, [point[1:], 1, [point]])


def list_add(list1, list2):
    '''
    used for list addition
    :param list1: the first list
    :param list2: the second list
    :return: the new list after connecting the two lists
    '''

    new_list = []
    for i in range(len(list1)):
        new_list.append(list1[i] + list2[i])
    return new_list


def kmeans_MapReduce(data_points, centroids, sc):
    '''
    :param data_points: a set of points
    :param centroids: some initially assigned centroids
    :param sc: SparkContext, used for MapReduce
    :return: the clusters after kmeans
    '''

    new_centroids = centroids
    centroids = []
    assignment = dict()
    k = 0
    while set(new_centroids) != set(centroids) and k < 50:
        centroids = new_centroids
        assignment = sc.parallelize(data_points) \
            .map(lambda s: choose_centroid(s, centroids)) \
            .reduceByKey(lambda u, v: (list_add(u[0], v[0]), u[1] + v[1], u[2] + v[2])) \
            .map(lambda s: (tuple(map(lambda x: x / s[1][1], s[1][0])), s[1][2])) \
            .collectAsMap()
        new_centroids = list(assignment.keys())
        k += 1
    return assignment


def summarize(data_points):
    '''
    :param data_points: the data set of points
    :param return: the summary of a cluster
    '''

    N = len(data_points)
    SUM = [0] * (len(data_points[0]) - 1)
    SUMSQ = [0] * (len(data_points[0]) - 1)
    for point in data_points:
        for i in range(1, len(point)):
            SUM[i - 1] += point[i]
            SUMSQ[i - 1] += point[i] ** 2
    return [N, SUM, SUMSQ]


def get_DS(DS_points, n_cluster, sc):
    '''
    :param DS_points: the data set of points for DS
    :param n_cluster: the number of clusters for DS
    :param sc: SparkContect, used for MapReduce
    :return: the summary of DS (S, SUM, SUMSQ), and the record of assignment of points
    '''

    centroids = get_centroids(DS_points, n_cluster)
    clusters = kmeans_MapReduce(DS_points, centroids, sc)
    DS = []
    DS_record = []
    for centroid, points in clusters.items():
        comment = summarize(points)
        DS.append(comment)

        DS_record.append(list(map(lambda x: x[0], points)))

    return DS, DS_record


def init_centroids_sd(summary):
    '''
    :param summary: a piece of statistics
    :return: the centroid and standard deviation of this piece of statistics
    '''

    stat_cents = []
    stat_sd = []
    for stat in summary:
        centroid = list(map(lambda s: s / stat[0], stat[1]))
        stat_cents.append(centroid)
        sd = get_standard_deviation(stat)
        stat_sd.append(sd)
    return stat_cents, stat_sd

def check_set(point, centroids, sd_set, threshold):
    '''
    :param point: the point waiting to be added
    :param centroids: centroids of a set (DS or RS)
    :param sd: standard deviations of a set (DS or RS)
    :param threshold: (hyper-parameter alpha) * sqrt(dimension)
    :return: if yes, the index of target set; else -1
    '''

    candidates = []
    for i in range(len(centroids)):
        dist = get_Mahalanobis(point, centroids[i], sd_set[i])
        if dist < threshold:
            candidates.append([i, dist])
    if len(candidates) > 0:
        target_index = sorted(candidates, key=lambda x: x[1], reverse=False)[0][0]
        return target_index
    else:
        return -1


def update_set(point, index, set, record, centroids, sd_set):
    '''
    :param point: the point which is going to be added into DS or RS
    :param index: the index of target DS/RS which will insert this point
    :param set: the DS/RS waiting to be updated
    :param record: the points record of DS/RS waiting to be updated
    :param centroids: the centroids of DS/RS waiting to be updated
    :param sd_set: the standard deviations of DS/RS waiting be to updated
    :return set, record, centroids, sd_set: the updated set DS/RS
    '''

    # update DS or RS
    target_set = set[index]
    target_set[0] += 1
    for i in range(len(point) - 1):
        target_set[1][i] += point[i + 1]
        target_set[2][i] += point[i + 1] ** 2
    set[index] = target_set

    # update the points record of DS or RS
    record[index].append(point[0])

    # update centroids of DS or RS
    centroid = list(map(lambda s: s / target_set[0], target_set[1]))
    centroids[index] = centroid

    # update standard deviation of DS or RS
    sd = get_standard_deviation(target_set)
    sd_set[index] = sd
    return set, record, centroids, sd_set

def RS2CS(data_points, n_cluster, sc):
    '''
    :param data_points: the set of points
    :param n_cluster: the planned number of clusters
    :param sc: SparkContext, used for MapReduce
    :return: statistics of CS, Retained points list, and points record of CS
    '''

    centroids = get_centroids(data_points, n_cluster)
    clusters = kmeans_MapReduce(data_points, centroids, sc)
    CS = []
    RS = []
    CS_record = []
    for centroid, points in clusters.items():
        if len(points) > 1:
            comment = summarize(points)
            CS.append(comment)

            CS_record.append(list(map(lambda x: x[0], points)))
        elif len(points) == 1:
            RS.append(points[0])
    return CS, RS, CS_record

def get_CSRS(data_points, DS, DS_record, n_cluster, sc):
    '''
    :param data_points: the data set of points for CS and RS
    :param DS: Discard Set of statistics
    :param DS_record：the point record for DS
    :param n_cluster: the number of clusters for CS and RS
    :return: the summary of CS and RS (S, SUM, SUMSQ), and the record of assignment of CS points
    '''

    DS_cents, DS_sd = init_centroids_sd(DS)
    d = len(DS[0][1])
    threshold = alpha * math.sqrt(d)

    rest_points = []
    for point in data_points:
        # check DS
        DS_index = check_set(point, DS_cents, DS_sd, threshold)
        if DS_index > -1:
            DS, DS_record, DS_cents, DS_sd = update_set(point, DS_index, DS, DS_record, DS_cents, DS_sd)
        else:
            rest_points.append(point)

    CS, RS, CS_record = [], [], []
    if len(rest_points) != 0:
        CS, RS, CS_record = RS2CS(rest_points, n_cluster, sc)
    return DS, CS, RS, DS_record, CS_record


def get_standard_deviation(ds):
    '''
    :param ds: the statistics of a DS cluster
    :return: the standard deviation list of this DS cluster
    '''

    N = ds[0]
    SUM = ds[1]
    SUMSQ = ds[2]
    sd = []
    for i in range(len(SUM)):
        sd.append(math.sqrt(SUMSQ[i] / N - (SUM[i] / N) ** 2))
    return sd


def get_Mahalanobis(point, centroid, sd):
    '''
    :param point: a new point
    :param centroid: the centroid of a target cluster
    :param sd: the standard deviation of the target cluster
    :return: the Mahalanobis distance of the new point and this target cluster
    '''

    if len(point) > len(centroid):
        point = point[1:]
    dist = 0
    for i in range(len(point)):
        if sd[i] == 0:
            dist += (point[i] - centroid[i]) ** 2
        else:
            dist += ((point[i] - centroid[i]) / sd[i]) ** 2
    return math.sqrt(dist)


def add_points(data_points, DS, CS, RS, DS_record, CS_record):
    '''
    :param data_points: the data set of points
    :param DS: the DS
    :param CS: the CS
    :param RS: the RS
    :param DS_record: the points record of DS
    :param CS_record: the points record of CS
    :return: the updated DS, CS, RS, DS_record, CS_record
    '''

    DS_cents, DS_sd = init_centroids_sd(DS)
    CS_cents, CS_sd = init_centroids_sd(CS)
    d = len(DS[0][1])
    threshold = alpha * math.sqrt(d)

    for point in data_points:
        # check DS
        DS_index = check_set(point, DS_cents, DS_sd, threshold)
        if DS_index > -1:
            DS, DS_record, DS_cents, DS_sd = update_set(point, DS_index, DS, DS_record, DS_cents, DS_sd)
            continue

        # check CS
        CS_index = check_set(point, CS_cents, CS_sd, threshold)
        if CS_index > -1:
            CS, CS_record, CS_cents, cs_sd = update_set(point, CS_index, CS, CS_record, CS_cents, CS_sd)
            continue

        RS.append(point)

    return DS, CS, RS, DS_record, CS_record


def add_set(set, new_set, record, new_record):
    '''
    :param set: the old set DS/CS
    :param new_set: the new set DS/RS
    :param record: the old points record DS/RS
    :param new_record: the new points record DS/RS
    :return: the updated set and record
    '''

    # update set
    set += new_set
    # update record
    record += new_record
    return set, record


def merge_stat(stat1, stat2):
    '''
    :param stat1: the statistics of cluster 1
    :param stat2: the statistics of cluster 2
    :return: the statistics of the cluster after merging
    '''

    stat1[0] += stat2[0]
    for i in range(len(stat1[1])):
        stat1[1][i] += stat2[1][i]
        stat1[2][i] += stat2[2][i]
    return stat1


def merge_CS(CS, CS_record):
    '''
    :param CS: the CS statistics set
    :param CS_record: the points record of CS
    :return CS: the CS statistics set in which some have merged
    :return CS_record: the points record of CS in which some have merged
    '''

    d = len(CS[0][1])
    threshold = alpha * math.sqrt(d)

    centroids, std = init_centroids_sd(CS)
    abanden_CS = []
    abanden_record = []
    for i in range(len(CS) - 1):
        if CS[i] in abanden_CS:
            continue
        for j in range(i + 1, len(CS)):
            if CS[j] in abanden_CS:
                continue
            dist = get_Mahalanobis(centroids[j], centroids[i], std[i])
            if dist < threshold:
                CS[i] = merge_stat(CS[i], CS[j])
                abanden_CS.append(CS[j])
                CS_record[i] += CS_record[j]
                abanden_record.append(CS_record[j])
                break

    for cs in abanden_CS:
        CS.remove(cs)
    for record in abanden_record:
        CS_record.remove(record)
    return CS, CS_record


def merge_all2DS(DS, CS, RS, DS_record, CS_record):
    '''
    :param DS: the DS statistics set
    :param CS: the CS statistics set
    :param RS: the RS points list
    :param DS_record: the points record of DS
    :param CS_record: the points record of CS
    :return DS: the DS statistics set
    :return DS_record: the final assignment of all points
    :return outliers_CS: statistics of CS which are not close to all the clusters
    :return outliers_RS: points of RS which are not close to all the clusters
    :return outliers: IDs of points which are not close to all the clusters
    '''

    DS_cents, DS_std = init_centroids_sd(DS)
    threshold = alpha * math.sqrt(len(DS_cents[0]))
    outliers_CS = []
    outliers_RS = []
    outliers = []

    for i in range(len(CS)):
        cs = CS[i]
        cs_centroid = list(map(lambda s: s / cs[0], cs[1]))
        min_dist = []
        for j in range(len(DS)):
            dist = get_Mahalanobis(cs_centroid, DS_cents[j], DS_std[j])
            min_dist.append([j, dist])
        (assigned_index, dis) = sorted(min_dist, key=lambda x: x[1], reverse=False)[0]
        if dis < threshold:
            DS[assigned_index][0] += cs[0]
            DS_record[assigned_index] += CS_record[i]
        else:
            outliers_CS.append(cs)
            for point_id in CS_record[i]:
                outliers.append(point_id)

    for rs in RS:
        min_dist = []
        for k in range(len(DS)):
            dist = get_Mahalanobis(rs, DS_cents[k], DS_std[k])
            min_dist.append([k, dist])
        (assigned_index, dis) = sorted(min_dist, key=lambda x: x[1], reverse=False)[0]
        if dis < threshold:
            DS[assigned_index][0] += 1
            DS_record[assigned_index].append(rs[0])
        else:
            outliers_RS.append(rs)
            outliers.append(rs[0])
    return DS, DS_record, outliers_CS, outliers_RS, outliers


def intermediate_report(intermediate_file, file_index, DS, CS, RS):
    '''
    :param fd: file descriptor of the intermediate report file
    :param file_index: the round id
    :param DS: the DS statistics set
    :param CS: the CS statistics set
    :param RS: the RS points list
    '''
    if file_index > 0:
        report = ''
        report += str(file_index) + ','
        report += str(len(DS)) + ','
        report += str(sum(map(lambda x: x[0], DS))) + ','
        report += str(len(CS)) + ','
        report += str(sum(map(lambda x: x[0], CS))) + ','
        report += str(len(RS)) + '\n'
        with open(intermediate_file, 'a') as f:
            f.write(report)


if __name__ == "__main__":
    start_time = time.time()
    file_index = 0
    DS, CS, RS, DS_record, CS_record = [], [], [], [], []
    conf = SparkConf().setAppName('inf553_hw5').setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    with open(intermediate_file, 'w') as f_inter:
        f_inter.write(Header + '\n')
    for file in os.listdir(input_path):
        if not os.path.isdir(file):
            # write the intermediate report
            intermediate_report(intermediate_file, file_index, DS, CS, RS)

            file_index += 1
            if file_index == 1:
                # load data of the first file
                with open(os.path.join(input_path, file), 'r') as f:
                    data_points = [line.split(',') for line in f.readlines()]
                    data_points = [[int(line[0])] + list(map(float, line[1:])) for line in data_points]

                # divide data into DS points and other points
                DS_points = random.sample(data_points, int(0.05 * len(data_points)))
                rest_points = list(map(list, set(map(tuple, data_points)) - set(map(tuple, DS_points))))

                # get DS summary
                DS, DS_record = get_DS(DS_points, n_cluster, sc)

                # get CS and RS summary

                DS, CS, RS, DS_record, CS_record = get_CSRS(rest_points, DS, DS_record, 5 * n_cluster, sc)

            else:
                # load the next file
                with open(os.path.join(input_path, file), 'r') as f:
                    data_points = [line.split(',') for line in f.readlines()]
                    data_points = [[int(line[0])] + list(map(float, line[1:])) for line in data_points]

                DS, CS, RS, DS_record, CS_record = add_points(data_points, DS, CS, RS, DS_record, CS_record)

                if len(RS) != 0:
                    # merge_RS
                    new_CS, RS, new_CS_record = RS2CS(RS, 5 * n_cluster, sc)
                    # update CS
                    CS, CS_record = add_set(CS, new_CS, CS_record, new_CS_record)

                if len(CS) != 0:
                    # merge CS
                    CS, CS_record = merge_CS(CS, CS_record)


    DS, DS_record, outliers_CS, outliers_RS, outliers = merge_all2DS(DS, CS, RS, DS_record, CS_record)
    # write the intermediate report
    intermediate_report(intermediate_file, file_index, DS, outliers_CS, outliers_RS)

    result = []
    for i in range(len(DS_record)):
        for id in DS_record[i]:
            result.append([id, i])
    for point_id in outliers:
        result.append([point_id, -1])
    result.sort(key=lambda x: x[0], reverse=False)
    output = dict(map(lambda x: [str(x[0]), x[1]], result))
    with open(result_file, 'w') as f_res:
        f_res.write(json.dumps(output))

    print('Duration: %f' % (time.time() - start_time))
