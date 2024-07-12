import hoomd
import numpy as np


class StdOutLogger(hoomd.custom.Action):
    def __init__(self, n_steps, sim):
        self.n_steps = int(n_steps)
        self.sim = sim
        self.starting_step = sim.timestep

    def act(self, timestep):
        if timestep != 0:
            tps = np.round(self.sim.tps, 2)
            current_step = self.sim.timestep - self.starting_step
            eta = np.round((self.n_steps - current_step) / (60 * tps), 1)
            if eta <= 60.0:
                print(
                    f"Step {current_step} of {self.n_steps}; TPS: {tps}; ETA: "
                    f"{eta} minutes"
                )
            else:
                eta_hour = eta // 60
                eta_min = np.round(eta % 60, 0)
                print(
                    f"Step {current_step} of {self.n_steps}; TPS: {tps}; ETA: "
                    f"{eta_hour} hours, {eta_min} minutes"
                )


class PullParticles(hoomd.custom.Action):
    def __init__(self, shift_by, axis, neg_filter, pos_filter):
        self.shift_by = shift_by
        self.axis = axis
        self.neg_filter = neg_filter
        self.pos_filter = pos_filter

    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            neg_filter = snap.particles.rtag[self.neg_filter.tags]
            pos_filter = snap.particles.rtag[self.pos_filter.tags]
            snap.particles.position[neg_filter] -= self.shift_by * self.axis
            snap.particles.position[pos_filter] += self.shift_by * self.axis


class StressStrainLogger(hoomd.custom.Action):
    def __init__(self, sim, pull_axis):
        self.sim = sim
        self.axis = pull_axis
        self.log_freq = sim.log_write_freq
        tensor_index_map = {0: 0, 1: 3, 2: 5}
        self.tensor_log_axis = tensor_index_map[int(self.axis[0])]

    def act(self, timestep):
        strain = np.round(self.sim.strain[0], 6)
        stress = self.sim.operations.computes[0].pressure_tensor
        array_index = (timestep - self.sim._initial_timestep) // self.log_freq
        self.sim._run_strain[array_index] = strain
        self.sim._run_stress[array_index] = stress[self.tensor_log_axis]


class UpdateWalls(hoomd.custom.Action):
    def __init__(self, sim):
        self.sim = sim

    def act(self, timestep):
        self.update_walls()

    def update_walls(self):
        for wall_axis in self.sim._wall_forces:
            wall_force = self.sim._wall_forces[wall_axis][0]
            wall_kwargs = self.sim._wall_forces[wall_axis][1]
            self.sim.remove_force(wall_force)
            self.sim.add_walls(wall_axis, **wall_kwargs)


class ScaleEpsilon(hoomd.custom.Action):
    def __init__(self, sim, scale_factor):
        self.scale_factor = scale_factor
        self.sim = sim

    def act(self, timestep):
        self.sim.adjust_epsilon(shift_by=self.scale_factor)


class ScaleSigma(hoomd.custom.Action):
    def __init__(self, sim, scale_factor):
        self.scale_factor = scale_factor
        self.sim = sim

    def act(self, timestep):
        self.sim.adjust_sigma(shift_by=self.scale_factor)
        self.sim._lj_force()
