"""This implementation of K-Means uses NumPy to accelerate the numerous distance measurements required. It also uses Futures to execute all tests in parallel accross system cores, to reduce test durations. The final printout contains all calculated solutions to HW4.
"""
import concurrent.futures
import numpy as np
import pandas as pd
import time

task1Samples = np.array([[3, 5],
                         [3, 4],
                         [2, 8],
                         [2, 3],
                         [6, 2],
                         [6, 4],
                         [7, 3],
                         [7, 4],
                         [8, 5],
                         [7, 6]], dtype=float)
task1Centroids = np.array([[[4, 6], [5, 4]],
                           [[4, 6], [5, 4]],
                           [[3, 3], [8, 3]],
                           [[3, 2], [4, 8]]], dtype=float)

task3Samples = [[4.7, 3.2],
                [4.9, 3.1],
                [5.0, 3.0],
                [4.6, 2.9],
                [5.9, 3.2],
                [6.7, 3.1],
                [6.0, 3.0],
                [6.2, 2.8]]
task3Y = [0, 0, 0, 0, 1, 1, 1, 1]


def listsAreEqual(a, b):
    return np.array_equal(np.sort(a), np.sort(b))
    # l1.sort()
    # l2.sort()
    # return l1 == l2


def manhattan(a, b):
    return np.sum(np.absolute(np.subtract(a, b)))
    # return sum(abs(e1-e2) for e1, e2 in zip(a, b))


def euclidean(a, b):
    return np.sqrt(np.sum(np.square(np.subtract(a, b))))
    # return math.sqrt(sum((e1-e2)**2 for e1, e2 in zip(a, b)))


def cosine(a, b):
    return 1 - np.divide(np.sum(np.multiply(a, b)),
                         np.multiply(np.sqrt(np.sum(np.square(a))),
                                     np.sqrt(np.sum(np.square(b)))))
    # numerator = sum(e1*e2 for e1, e2 in zip(a, b))
    # denominator1 = sum(e1**2 for e1 in a)
    # denominator2 = sum(e1**2 for e1 in b)
    # return 1 - (numerator / (denominator1 * denominator2))


def jaccard(a, b):
    return 1 - np.divide(np.sum(np.minimum(a, b)),
                         np.sum(np.maximum(a, b)))
    # numerator = sum(min(e1, e2) for e1, e2 in zip(a, b))
    # denominator = sum(max(e1, e2) for e1, e2 in zip(a, b))
    # return 1 - (numerator / denominator)


def SSE(distance_func, X, centroids):
    result = 0
    for centroid in centroids:
        for point in X:
            result += distance_func(centroid, point)**2
    return result


def accuracy(Y, computed_Y):
    # Initialize a 2D array of scores.
    # The first dimension is the index of the cluster a data point (i) was
    # classified into (computed_Y).
    # The second dimension is a list of the number of ground-truth labels
    # associated with each cluster.
    # The majority label is said to represent this cluster. Everything in the
    # cluster that doesn't have this label is considered an incorrect clustering.
    cluster_score = []
    for i in range(len(Y)):
        cluster_score.insert(i, [])
        for j in range(len(Y)):
            cluster_score[i].insert(j, 0)
    # Do the scoring.
    # We want to track how many of each label ended up in each cluster.
    for i in range(len(Y)):
        cluster_score[computed_Y[i]][Y[i][0]] += 1
    # Determine which label won for each cluster.
    correct = 0
    total = 0
    for i in range(len(Y)):
        winner = 0
        max_seen = 0
        for j in range(len(Y)):
            if cluster_score[i][j] > max_seen:
                winner = j
                max_seen = cluster_score[i][j]
        # Get the accuracy for this cluster.
        for j in range(len(Y)):
            total += cluster_score[i][j]
            if j == winner:
                correct += cluster_score[i][j]
    return correct / total


def kMeans(distance_func, X, Y=[], K=0, centroids=np.array([]), stoppers=["unchanged"],
           maxIterations=0, task_id=""):
    # All tests report their distance function by name.
    if task_id == "task2stopper":
        # Stoppers also report the stop condition(s).
        ret = str(distance_func.__name__) + "\t" + str(stoppers) + "\n"
    else:
        ret = str(distance_func.__name__) + "\n"
    # The centroid associated with each data point.
    computed_Y = np.full(X.shape[0], 0)
    # We need at least 1 stopping criteria.
    if len(stoppers) < 1 and maxIterations == 0:
        print("Missing stop criteria.")
        return
    # If the centroids list is not empty, use it for initial cluster centroids.
    if centroids.size > 0:
        # If we didn't get a full list of centroids, something went wrong.
        if len(centroids) != K:
            print("Mismatch: Found " + str(centroids) +
                  "initial centroids with K=" + str(K))
    # Otherwise, use random points from the dataset.
    else:
        # Sample K data points without replacement from X,
        # to distance the centroids.
        centroids = X[np.random.choice(X.shape[0], K, replace=False), :]

    # Track how long it takes to run all iterations for a given K-means calculation.
    start = time.time_ns()
    iterations = 0
    while True:
        if task_id == "task1full":
            ret += "Iteration " + str(iterations) + ":\t" \
                + np.array_repr(centroids).replace('\n', '') + "\n"

        # Get a copy of the old centroids so we can check for changes later.
        old_centroids = np.copy(centroids)
        # Increment the iterations count.
        iterations += 1
        # Used to update centroid positions.
        tmp_centroid_sum = np.zeros(centroids.shape)
        tmp_centroid_count = np.zeros(centroids.shape[0])
        # Identify the centroid each point is closest to.
        for point_idx, point in enumerate(X):
            shortest_distance = float('inf')
            # Check each centroid to find the shortest distance.
            for centroid_idx, centroid in enumerate(centroids):
                distance = distance_func(point, centroid)
                if distance < shortest_distance:
                    shortest_distance = distance
                    # Associate this point with this cluster.
                    computed_Y[point_idx] = centroid_idx
            # Consider this point in updated centroid calculations.
            # We will use this later to average and find the new centroid location.
            tmp_centroid_sum[computed_Y[point_idx]] = np.add(
                tmp_centroid_sum[computed_Y[point_idx]], point)
            # Keep track of how many points ended up in this cluster.
            tmp_centroid_count[computed_Y[point_idx]] += 1
        # Recompute the centroid of each cluster.
        for i in range(len(centroids)):
            # Case where the centroid has no data points.
            # This should never happen unless something is very wrong.
            if tmp_centroid_count[i] == 0:
                print("A centroid was found empty at iteration " + str(iterations))
                # Keep the centroid where it is.
                # This is the simplest solution.
                centroids[i] = np.copy(old_centroids[i])
            else:
                # The centroid belongs at the mean of each feature.
                centroids[i] = np.divide(tmp_centroid_sum[i],
                                         np.full(centroids.shape[1], tmp_centroid_count[i]))

        # Our stop conditions.
        # Centroids do not move (converge to a stable solution).
        if "unchanged" in stoppers and listsAreEqual(old_centroids, centroids):
            break
        # SSE increases (no more improvements).
        if "sse" in stoppers and SSE(distance_func, X, centroids) \
                > SSE(distance_func, X, old_centroids):
            # Copy back the centroids from before SSE increased.
            centroids = np.copy(old_centroids)
            break
        # Stop after some upper bound on iterations (prevent running forever).
        # maxIterations == 0 means that the default iteration bounds are used.
        if (maxIterations != 0 and iterations >= maxIterations) \
                or (maxIterations == 0 and iterations >= 500):
            break
    # End timing. Iterations complete.
    end = time.time_ns()

    # Return a string with some performance results.
    # The calculations and format of returned data depend on the problem (task_id).
    if task_id == "task1full":
        # Print the classifications of each data point.
        for point_idx, point in enumerate(X):
            ret += "X" + str(point_idx + 1) + ": " + \
                str(computed_Y[point_idx]) + "\n"
    if task_id == "task2":
        ret += "SSE=" + str(SSE(distance_func, X, centroids)) + "\n"
        ret += "Predictive accuracy=" + str(accuracy(Y, computed_Y))
    if task_id == "task2stopper":
        ret += str(iterations) + "\t" + str(SSE(distance_func, X, centroids)) \
            + "\t" + str(end - start) + " nanoseconds"
    return ret


# Task 3 is specialized, so it gets its own function.
def doTask3():
    # Find the farthest distance.
    farthest = float(0)
    # Find the nearest distance.
    nearest = float('inf')
    # Find the average distance.
    total = 0
    count = 0
    for i in range(len(task3Samples)):
        for j in range(i, len(task3Samples)):
            # Do not consider the distances betwen points in the same cluster.
            if task3Y[i] == task3Y[j]:
                continue
            distance = euclidean(task3Samples[i], task3Samples[j])
            if distance > farthest:
                farthest = distance
            if distance < nearest:
                nearest = distance
            total += distance
            count += 1
    average = total / count
    # Report final results.
    ret = "A: Farthest=" + str(round(farthest, 4)) + "\n" \
        + "B: Closest=" + str(round(nearest, 4)) + "\n" \
        + "C: Average=" + str(round(average, 4)) + "\n" \
        + "C is the most robust to noise, as the others can be radically changed by outliers.\n"
    return ret


def main():
    # Import csv files into Pandas dataframes.
    # These are used in task 2.
    X = pd.read_csv("./data.csv")
    Y = pd.read_csv("./label.csv")

    # Preprocess data.
    X = X.to_numpy(dtype=float)
    Y = Y.to_numpy(dtype=int)

    # Run each test in concurrent processes.
    # This helps scale our computation to finish more quickly.
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        futures = []
        # Task 1
        futures.append(executor.submit(kMeans, manhattan, task1Samples, K=2,
                                       centroids=task1Centroids[0], task_id="task1full"))
        futures.append(executor.submit(kMeans, euclidean, task1Samples, K=2,
                                       centroids=task1Centroids[1], task_id="task1full"))
        futures.append(executor.submit(kMeans, manhattan, task1Samples, K=2,
                                       centroids=task1Centroids[2], task_id="task1full"))
        futures.append(executor.submit(kMeans, manhattan, task1Samples, K=2,
                                       centroids=task1Centroids[3], task_id="task1full"))

        # Task 2
        futures.append(executor.submit(
            kMeans, euclidean, X, Y=Y, K=10, task_id="task2"))
        futures.append(executor.submit(
            kMeans, cosine, X, Y=Y, K=10, task_id="task2"))
        futures.append(executor.submit(
            kMeans, jaccard, X, Y=Y, K=10, task_id="task2"))

        futures.append(executor.submit(kMeans, euclidean, X, Y=Y,
                       K=10, maxIterations=100, stoppers=["unchanged", "sse"], task_id="task2stopper"))
        futures.append(executor.submit(kMeans, cosine, X, Y=Y,
                       K=10, maxIterations=100, stoppers=["unchanged", "sse"], task_id="task2stopper"))
        futures.append(executor.submit(kMeans, jaccard, X, Y=Y,
                       K=10, maxIterations=100, stoppers=["unchanged", "sse"], task_id="task2stopper"))

        futures.append(executor.submit(kMeans, euclidean, X, Y=Y,
                       K=10, stoppers=["unchanged"], task_id="task2stopper"))
        futures.append(executor.submit(kMeans, cosine, X, Y=Y,
                       K=10, stoppers=["unchanged"], task_id="task2stopper"))
        futures.append(executor.submit(kMeans, jaccard, X, Y=Y,
                       K=10, stoppers=["unchanged"], task_id="task2stopper"))
        futures.append(executor.submit(kMeans, euclidean, X, Y=Y,
                       K=10, stoppers=["sse"], task_id="task2stopper"))
        futures.append(executor.submit(kMeans, cosine, X, Y=Y,
                       K=10, stoppers=["sse"], task_id="task2stopper"))
        futures.append(executor.submit(kMeans, jaccard, X, Y=Y,
                       K=10, stoppers=["sse"], task_id="task2stopper"))
        futures.append(executor.submit(kMeans, euclidean, X, Y=Y,
                       K=10, stoppers=[], maxIterations=100, task_id="task2stopper"))
        futures.append(executor.submit(kMeans, cosine, X, Y=Y,
                       K=10, stoppers=[], maxIterations=100, task_id="task2stopper"))
        futures.append(executor.submit(kMeans, jaccard, X, Y=Y,
                       K=10, stoppers=[], maxIterations=100, task_id="task2stopper"))

        # Task 3
        futures.append(executor.submit(doTask3))

        # DEBUG
        # Print as we get results, just to keep an eye on things.
        # for future in concurrent.futures.as_completed(futures):
        #     print(future.result())

        # Now wait for the futures to complete and print the report.
        # By waiting for results in this way,
        # we ensure that print ordering is deterministic.

        # An iterator, for easy iteration over each result.
        iter_futures = iter(futures)

        # Task 1
        print("Task 1:")
        print("1.1")
        print(next(iter_futures).result())
        print("1.2")
        print(next(iter_futures).result())
        print("1.3")
        print(next(iter_futures).result())
        print("1.4")
        print(next(iter_futures).result())

        # Task 2
        print("\nTask 2")

        print("2.1 and 2.2")
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())

        print("2.3")
        print("Iterations\tSSE\tTime")
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())

        print("2.4")
        print("Iterations\tSSE\tTime")
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())

        # Task 3
        print("\nTask 3")
        print(next(iter_futures).result())


if __name__ == "__main__":
    main()
