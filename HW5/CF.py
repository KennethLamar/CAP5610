import concurrent.futures
from itertools import product
from statistics import mean

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import KNNBasic

# Whether or not cross-validation should be verbose.
verbose = False

# All combinations of sim options we want to test.
sim_options = {'name': ['msd', 'cosine', 'pearson'],
               'user_based': [True, False, 'PMF']}

# Prepare a reader.
print('Preparing reader.')
reader = Reader(line_format='user item rating timestamp',
                sep=',', rating_scale=(1, 5), skip_lines=1)

# Load the dataset.
print('Loading dataset.')
# data = Dataset.load_from_file('./ratings.csv', reader)
data = Dataset.load_from_file('./ratings_small.csv', reader)

# Run a test and return results.
def get_results(data, sim_options, k=None):
    result = {}

    if sim_options['user_based'] == 'PMF':
        # SVD without biases is PMF.
        algo = SVD(biased=False)
        result['CF'] = 'PMF'
        result['sim'] = 'None'
        result['K'] = 0
    else:
        if k == None:
            algo = KNNBasic(k=40, min_k=1, sim_options=sim_options)
            result['K'] = 0
        else:
            algo = KNNBasic(k=k, min_k=k, sim_options=sim_options)
            result['K'] = k
        if sim_options['user_based']:
            result['CF'] = 'user-user'
        else:
            result['CF'] = 'item-item'
        result['sim'] = sim_options['name']

    cv = cross_validate(algo, data, measures=[
                        'RMSE', 'MAE'], cv=5, verbose=False)

    result['MAE'] = mean(cv['test_mae'])
    result['RMSE'] = mean(cv['test_rmse'])
    result['fit_time'] = mean(cv['fit_time'])
    result['test_time'] = mean(cv['test_time'])

    return result


print('Starting main tests.')
# Run each test in a concurrent process.
# This helps scale our computation to finish more quickly.
with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
    futures = []
    print('Starting main tests.')
    for index, prod in enumerate((dict(zip(sim_options, x)) for
                                  x in product(*sim_options.values()))):
        # Most PMF tests are redundant. Skip them.
        if prod['user_based'] == 'PMF' and prod['name'] != 'msd':
            continue
        # Run 5-fold cross-validation.
        futures.append(executor.submit(get_results, data, prod))

    print('Starting varied K tests.')
    for user_based in [True, False]:
        for k in range(1, 200+1, 1):
            # Run 5-fold cross-validation.
            futures.append(executor.submit(get_results, data,
                           sim_options={'name': 'msd', 'user_based': user_based}, k=k))

    print('Writing output. Waiting for tests to complete.')
    with open('output.csv', 'w') as f:
        f.write('CF,sim,K,MAE,RMSE,fit_time,test_time\n')
        for future in futures:
            result = future.result()
            result_str = '{CF},{sim},{K},{MAE},{RMSE},{fit_time},{test_time}'.format_map(
                result)
            print(result_str)
            f.write(result_str + '\n')
