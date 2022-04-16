from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange, LogUniformParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation import GridSearch, RandomSearch
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.job import LocalClearmlJob

from clearml import Task
from clearml.automation.optuna.optuna import OptunaObjective

from multiprocessing import Pool

estimators = {
    'SVC': [
        DiscreteParameterRange('General/estimator', ['SVC']),
        UniformIntegerParameterRange('General/SVC/C', min_value=1, max_value=1000),
        UniformParameterRange('General/SVC/gamma', min_value=0.0001, max_value=0.01),
        DiscreteParameterRange('General/SVC/kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    ],
    'RF': [
        DiscreteParameterRange('General/estimator', ['RF']),
        UniformIntegerParameterRange('General/RF/n_estimators', min_value=1, max_value=10),
        UniformIntegerParameterRange('General/RF/max_depth', min_value=1, max_value=10),
        UniformIntegerParameterRange('General/RF/max_features', min_value=1, max_value=10)
    ],
    'ADA': [
        DiscreteParameterRange('General/estimator', ['ADA']),
        UniformIntegerParameterRange('General/ADA/n_estimators', min_value=1, max_value=10),
        UniformParameterRange('General/ADA/learning_rate', min_value=0.1, max_value=1),
    ],
    'MLP': [
        DiscreteParameterRange('General/estimator', ['MLP']),
        DiscreteParameterRange('General/MLP/hidden_layer_sizes', [(10, 10), (10,), (25, 25), (25,), (50, 50), (50,)]),
        UniformParameterRange('General/MLP/learning_rate_init', min_value=0.0001, max_value=0.1),
    ]
}


def run_optimizer(args):
    estimator, parameters = args
    task = Task.init(project_name='hpo_blogpost_random',
                     task_name=f"optimizer_{estimator}",
                     task_type=Task.TaskTypes.optimizer,
                     reuse_last_task_id=False)

    optimizer = HyperParameterOptimizer(
        # specifying the task to be optimized, task must be in system already so it can be cloned
        base_task_id='9f217f78b8874122a575540153a6838d',
        # setting the hyper-parameters to optimize
        hyper_parameters=parameters,
        # setting the objective metric we want to maximize/minimize
        objective_metric_title=estimator,
        # objective_metric_series='validation: epoch_sparse_categorical_accuracy',
        objective_metric_series='ROC AUC Test',
        objective_metric_sign='max',

        # setting optimizer
        optimizer_class=RandomSearch,

        # configuring optimization parameters
        execution_queue='default',
        # max_number_of_concurrent_tasks=20,
        compute_time_limit=None,
        total_max_jobs=50,
        min_iteration_per_job=None,
        max_iteration_per_job=None,
        save_top_k_tasks_only=5
    )

    # report every 12 seconds, this is way too often, but we are testing here J
    optimizer.set_report_period(0.1)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    optimizer.start_locally()
    # set the time limit for the optimization process (2 hours)
    # optimizer.set_time_limit(in_minutes=120.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()
    # optimization is completed, print the top performing experiments id
    # top_exp = optimizer.get_top_experiments(top_k=3)
    # print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()

with Pool() as p:
    print(estimators.items())
    p.map(run_optimizer, estimators.items())