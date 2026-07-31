import sys

import hydra

import benchmarks.cartpole as cartpole
import benchmarks.cartpole_ilqr as cartpole_ilqr
import benchmarks.cartpole_constrained as cartpole_constrained
import benchmarks.cartpole_ilqr_cycle as cartpole_ilqr_cycle
import benchmarks.cartpole_recursive_feas as cartpole_recursive_feas
import benchmarks.nonlinear_double_integrator as nonlinear_double_integrator
import benchmarks.bilinear_recursive_feas as bilinear_recursive_feas
import benchmarks.bilinear_closed_loop as bilinear_closed_loop
import benchmarks.bilinear_closed_loop2 as bilinear_closed_loop2
import benchmarks.bilinear_closed_loop_ilqr as bilinear_closed_loop_ilqr
import benchmarks.cartpole_scp as cartpole_scp
import benchmarks.cartpole_regularize as cartpole_regularize
import benchmarks.nonlinear_double_int_closed_loop as nonlinear_double_int_closed_loop
import benchmarks.cartpole_closed_loop_ilqr as cartpole_closed_loop_ilqr
import benchmarks.stir_tank_subopt as stir_tank_subopt
import benchmarks.spring_masses as spring_masses
import benchmarks.cartpole_constrained_stability_scp as cartpole_constrained_stability_scp
import benchmarks.four_tank as four_tank


import matplotlib
matplotlib.use('pdf')


@hydra.main(config_path='configs', config_name='cartpole.yaml')
def main_run_cartpole(cfg):
    cartpole.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole.yaml')
def main_run_cartpole_ilqr(cfg):
    cartpole_ilqr.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole_constrained.yaml')
def main_run_cartpole_constrained(cfg):
    cartpole_constrained.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole_ilqr_cycle.yaml')
def main_run_cartpole_ilqr_cycle(cfg):
    cartpole_ilqr_cycle.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole_recursive_feas.yaml')
def main_run_cartpole_recursive_feas(cfg):
    cartpole_recursive_feas.run(cfg)


@hydra.main(config_path='configs', config_name='nonlinear_double_integrator.yaml')
def main_run_nonlinear_double_integrator(cfg):
    nonlinear_double_integrator.run(cfg)


@hydra.main(config_path='configs', config_name='bilinear_recursive_feas.yaml')
def main_run_bilinear_recursive_feas(cfg):
    bilinear_recursive_feas.run(cfg)


@hydra.main(config_path='configs', config_name='bilinear_closed_loop.yaml')
def main_run_bilinear_closed_loop(cfg):
    bilinear_closed_loop.run(cfg)
    

@hydra.main(config_path='configs', config_name='bilinear_closed_loop2.yaml')
def main_run_bilinear_closed_loop2(cfg):
    bilinear_closed_loop2.run(cfg)


@hydra.main(config_path='configs', config_name='bilinear_closed_loop_ilqr.yaml')
def main_run_bilinear_closed_loop_ilqr(cfg):
    bilinear_closed_loop_ilqr.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole_scp.yaml')
def main_run_cartpole_scp(cfg):
    cartpole_scp.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole_regularize.yaml')
def main_run_cartpole_regularize(cfg):
    cartpole_regularize.run(cfg)


@hydra.main(config_path='configs', config_name='nonlinear_double_int_closed_loop.yaml')
def main_run_nonlinear_double_int_closed_loop(cfg):
    nonlinear_double_int_closed_loop.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole_closed_loop_ilqr.yaml')
def main_run_cartpole_closed_loop_ilqr(cfg):
    cartpole_closed_loop_ilqr.run(cfg)


@hydra.main(config_path='configs', config_name='stir_tank_subopt.yaml')
def main_run_stir_tank_subopt(cfg):
    stir_tank_subopt.run(cfg)


@hydra.main(config_path='configs', config_name='spring_masses.yaml')
def main_run_spring_masses(cfg):
    spring_masses.run(cfg)


@hydra.main(config_path='configs', config_name='cartpole_constrained_stability_scp.yaml')
def main_run_cartpole_constrained_stability_scp(cfg):
    cartpole_constrained_stability_scp.run(cfg)


@hydra.main(config_path='configs', config_name='four_tank.yaml')
def main_run_four_tank(cfg):
    four_tank.run(cfg)


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
    elif sys.argv[1] == 'cartpole_constrained':
        sys.argv[1] = base + 'cartpole_constrained/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_constrained()
    elif sys.argv[1] == 'cartpole_ilqr_cycle':
        sys.argv[1] = base + 'cartpole_ilqr_cycle/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_ilqr_cycle()
    elif sys.argv[1] == 'cartpole_recursive_feas':
        sys.argv[1] = base + 'cartpole_recursive_feas/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_recursive_feas()
    elif sys.argv[1] == 'bilinear_recursive_feas':
        sys.argv[1] = base + 'bilinear_recursive_feas/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_bilinear_recursive_feas()
    elif sys.argv[1] == 'bilinear_closed_loop':
        sys.argv[1] = base + 'bilinear_closed_loop/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_bilinear_closed_loop()
    elif sys.argv[1] == 'bilinear_closed_loop2':
        sys.argv[1] = base + 'bilinear_closed_loop2/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_bilinear_closed_loop2()
    elif sys.argv[1] == 'bilinear_closed_loop_ilqr':
        sys.argv[1] = base + 'bilinear_closed_loop_ilqr/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_bilinear_closed_loop_ilqr()
    elif sys.argv[1] == 'cartpole_scp':
        sys.argv[1] = base + 'cartpole_scp/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_scp()
    elif sys.argv[1] == 'cartpole_regularize':
        sys.argv[1] = base + 'cartpole_regularize/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_regularize()
    elif sys.argv[1] == 'nonlinear_double_int_closed_loop':
        sys.argv[1] = base + 'nonlinear_double_int_closed_loop/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_nonlinear_double_int_closed_loop()
    elif sys.argv[1] == 'cartpole_closed_loop_ilqr':
        sys.argv[1] = base + 'cartpole_closed_loop_ilqr/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_closed_loop_ilqr()
    elif sys.argv[1] == 'stir_tank_subopt':
        sys.argv[1] = base + 'stir_tank_subopt/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_stir_tank_subopt()
    elif sys.argv[1] == 'spring_masses':
        sys.argv[1] = base + 'spring_masses/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_spring_masses()
    elif sys.argv[1] == 'cartpole_constrained_stability_scp':
        sys.argv[1] = base + 'cartpole_constrained_stability_scp/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_constrained_stability_scp()
    elif sys.argv[1] == 'four_tank':
        sys.argv[1] = base + 'four_tank/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_four_tank()
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
