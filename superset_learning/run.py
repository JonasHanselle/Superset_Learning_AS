import logging
import sys
import configparser
import multiprocessing as mp
import database_utils
from evaluation import evaluate_scenario
from approaches.single_best_solver import SingleBestSolver
from approaches.oracle import Oracle
# from baselines.linear_regressor import LinearRegressor
from baselines.mlp_regressor import MLPRegressor
# from approaches.survival_forests.surrogate import SurrogateSurvivalForest
# from approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.sunny import SUNNY
from baselines.snnap import SNNAP
from baselines.isac import ISAC
from baselines.satzilla11 import SATzilla11
from baselines.satzilla07 import SATzilla07
from sklearn.linear_model import Ridge
from par_10_metric import Par10Metric
from number_unsolved_instances import NumberUnsolvedInstances


logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())


def initialize_logging():
    logging.basicConfig(filename='logs/log_file.log', filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/test.cfg'))
    return config


def print_config(config: configparser.ConfigParser):
    for section in config.sections():
        logger.info(str(section) + ": " + str(dict(config[section])))


def log_result(result):
    logger.info("Finished experiements for scenario: " + result)


def create_approach(approach_names):
    approaches = list()
    for approach_name in approach_names:
        # if approach_name == 'sbs':
        #     approaches.append(SingleBestSolver())
        # if approach_name == 'superset_mlp':
        #     approaches.append(MLPRegressor(loss_function="superset"))
        # if approach_name == 'superset_linear':
        #     approaches.append(LinearRegressor(loss_function="superset"))
        # if approach_name == 'l2_linear':
        #     approaches.append(LinearRegressor(loss_function="l2"))
        # if approach_name == 'imputed_l2_linear':
        #     approaches.append(LinearRegressor(
        #         impute_censored=True, loss_function="l2"))
        if approach_name == 'imputed_l2_mlp':
            approaches.append(MLPRegressor(
                impute_censored=True, loss_function="l2"))

    return approaches


#######################
#         MAIN        #
#######################

if __name__ == '__main__':

    initialize_logging()
    config = load_configuration()
    logger.info("Running experiments with config:")
    print_config(config)

    db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(
        config)
    database_utils.create_table_if_not_exists(db_handle, table_name)

    amount_of_cpus_to_use = int(config['EXPERIMENTS']['amount_of_cpus'])
    pool = mp.Pool(amount_of_cpus_to_use)

    scenarios = config["EXPERIMENTS"]["scenarios"].split(",")
    approach_names = config["EXPERIMENTS"]["approaches"].split(",")
    amount_of_scenario_training_instances = int(
        config["EXPERIMENTS"]["amount_of_training_scenario_instances"])
    tune_hyperparameters = bool(
        int(config["EXPERIMENTS"]["tune_hyperparameters"]))

    for fold in range(1, 11):

        for scenario in scenarios:
            approaches = create_approach(approach_names)

            if len(approaches) < 1:
                logger.error("No approaches recognized!")
            for approach in approaches:
                metrics = list()
                metrics.append(Par10Metric())
                if approach.get_name() != 'oracle':
                    metrics.append(NumberUnsolvedInstances(False))
                    metrics.append(NumberUnsolvedInstances(True))
                logger.info("Submitted pool task for approach \"" +
                            str(approach.get_name()) + "\" on scenario: " + scenario)
                pool.apply_async(evaluate_scenario, args=(scenario, approach, metrics,
                                                          amount_of_scenario_training_instances, fold, config, tune_hyperparameters), callback=log_result)

                # evaluate_scenario(scenario, approach, metrics,
                #                 amount_of_scenario_training_instances, fold, config, tune_hyperparameters)
                print(
                    f'Finished evaluation of fold {fold} for approach {approach.get_name()} on scenario {scenario}')
    pool.close()
    print("Closed")
    pool.join()
    print("Joined")
    logger.info("Finished all experiments.")
