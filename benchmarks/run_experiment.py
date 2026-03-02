import sys

import hydra

import benchmarks.cartpole as cartpole
import benchmarks.cartpole_ilqr as cartpole_ilqr
import benchmarks.nonlinear_double_integrator as nonlinear_double_integrator


import matplotlib
matplotlib.use('pdf')


@hydra.main(config_path='configs', config_name='cartpole.yaml')
def main_run_cartpole(cfg):
    cartpole.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole.yaml')
def main_run_cartpole_ilqr(cfg):
    cartpole_ilqr.run(cfg)


@hydra.main(config_path='configs', config_name='nonlinear_double_integrator.yaml')
def main_run_nonlinear_double_integrator(cfg):
    nonlinear_double_integrator.run(cfg)


# @hydra.main(config_path='configs', config_name='power_converter.yaml')
# def main_run_power_converter(cfg):
#     power_converter.run(cfg)


# @hydra.main(config_path='configs', config_name='box_qp.yaml')
# def main_run_box_qp(cfg):
#     example = 'box_qp'
#     box_qp.run(cfg)


# @hydra.main(config_path='configs', config_name='box_qp_radii.yaml')
# def main_run_box_qp_radii(cfg):
#     box_qp.run_radius_experiment(cfg)



# @hydra.main(config_path='configs', config_name='box_qp_radii.yaml')
# def main_run_box_qp_radii(cfg):
#     box_qp.run_radius_experiment(cfg)


# @hydra.main(config_path='configs', config_name='hybrid_vehicle.yaml')
# def main_run_hybrid_vehicle(cfg):
#     hybrid_vehicle.run(cfg)


# @hydra.main(config_path='configs', config_name='phase_retrieval.yaml')
# def main_run_phase_retrieval(cfg):
#     phase_retrieval.run(cfg)


# @hydra.main(config_path='configs', config_name='knapsack.yaml')
# def main_run_knapsack(cfg):
#     knapsack.run(cfg)


# @hydra.main(config_path='configs', config_name='network_utility.yaml')
# def main_run_network_utility(cfg):
#     network_utility.run(cfg)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/sambhar9/lah_robust/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'nonlinear_double_integrator':
        sys.argv[1] = base + 'nonlinear_double_integrator/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_nonlinear_double_integrator()
    elif sys.argv[1] == 'cartpole':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for train_outputs not outputs
        # sys.argv[1] = 'hydra.run.dir=outputs/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv[1] = base + 'cartpole/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole()
    elif sys.argv[1] == 'cartpole_ilqr':
        sys.argv[1] = base + 'cartpole_ilqr/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_ilqr()
    elif sys.argv[1] == 'sparse_coding':
        sys.argv[1] = base + 'sparse_coding/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_sparse_coding()
    elif sys.argv[1] == 'phase_retrieval':
        sys.argv[1] = base + 'phase_retrieval/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_phase_retrieval()
    elif sys.argv[1] == 'power_converter':
        sys.argv[1] = base + 'power_converter/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_power_converter()
    elif sys.argv[1] == 'knapsack':
        sys.argv[1] = base + 'knapsack/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_knapsack()
    elif sys.argv[1] == 'network_utility':
        sys.argv[1] = base + 'network_utility/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_network_utility()
    elif sys.argv[1] == 'hybrid_vehicle':
        sys.argv[1] = base + 'hybrid_vehicle/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_hybrid_vehicle()
