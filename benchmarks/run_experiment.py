import sys

import hydra

import benchmarks.cartpole_ilqr as cartpole_ilqr
import benchmarks.nonlinear_double_integrator as nonlinear_double_integrator
import benchmarks.bilinear_closed_loop2 as bilinear_closed_loop2
import benchmarks.stir_tank_subopt as stir_tank_subopt
import benchmarks.two_tank as two_tank
import benchmarks.two_tank_constrained as two_tank_constrained


import matplotlib
matplotlib.use('pdf')


@hydra.main(config_path='configs', config_name='cartpole.yaml')
def main_run_cartpole_ilqr(cfg):
    cartpole_ilqr.run(cfg)


@hydra.main(config_path='configs', config_name='nonlinear_double_integrator.yaml')
def main_run_nonlinear_double_integrator(cfg):
    nonlinear_double_integrator.run(cfg)


@hydra.main(config_path='configs', config_name='bilinear_closed_loop2.yaml')
def main_run_bilinear_closed_loop2(cfg):
    bilinear_closed_loop2.run(cfg)


@hydra.main(config_path='configs', config_name='stir_tank_subopt.yaml')
def main_run_stir_tank_subopt(cfg):
    stir_tank_subopt.run(cfg)


@hydra.main(config_path='configs', config_name='two_tank.yaml')
def main_run_two_tank(cfg):
    two_tank.run(cfg)


@hydra.main(config_path='configs', config_name='two_tank_constrained.yaml')
def main_run_two_tank_constrained(cfg):
    two_tank_constrained.run(cfg)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/sambhar9/lah_robust/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'nonlinear_double_integrator':
        sys.argv[1] = base + 'nonlinear_double_integrator/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_nonlinear_double_integrator()
    elif sys.argv[1] == 'cartpole_ilqr':
        sys.argv[1] = base + 'cartpole_ilqr/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_cartpole_ilqr()
    elif sys.argv[1] == 'bilinear_closed_loop2':
        sys.argv[1] = base + 'bilinear_closed_loop2/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_bilinear_closed_loop2()
    elif sys.argv[1] == 'stir_tank_subopt':
        sys.argv[1] = base + 'stir_tank_subopt/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_stir_tank_subopt()
    elif sys.argv[1] == 'two_tank':
        sys.argv[1] = base + 'two_tank/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_two_tank()
    elif sys.argv[1] == 'two_tank_constrained':
        sys.argv[1] = base + 'two_tank_constrained/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_two_tank_constrained()
