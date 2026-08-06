#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import uproot
from opengate.tests import utility
import opengate as gate
from opengate.contrib.beamlines.ionbeamline import BeamlineModel
from opengate.contrib.tps.ionbeamtherapy import spots_info_from_txt, TreatmentPlanSource
from scipy.spatial.transform import Rotation
import itk
import matplotlib.pyplot as plt
import os
import numpy as np


def calculate_mean_unc(edep_arr, unc_arr, edep_thresh_rel=0.7):
    edep_max = np.amax(edep_arr)
    mask = edep_arr > edep_max * edep_thresh_rel
    unc_used = unc_arr[mask]
    unc_mean = np.mean(unc_used)

    return unc_mean


class sim_tps_dose:
    def __init__(self,flat:bool=False,nSim=40000):
        paths = utility.get_default_test_paths(
            __file__, "gate_test044_pbs", output_folder="test059"
        )
        output_path = paths.output
        ref_path = paths.output_ref
        label="flat" if flat else "weighted"
        d0name=f"{label}_edep0"
        d1name=f"{label}_edep1"

        # create the simulation
        sim = gate.Simulation()

        # main options
        sim.g4_verbose = False
        sim.g4_verbose_level = 1
        sim.visu = False
        sim.random_seed = 123654789
        sim.random_engine = "MersenneTwister"
        sim.number_of_threads = 1
        sim.output_dir = output_path

        # units
        km = gate.g4_units.km
        cm = gate.g4_units.cm
        mm = gate.g4_units.mm
        um = gate.g4_units.um
        MeV = gate.g4_units.MeV
        Bq = gate.g4_units.Bq
        nm = gate.g4_units.nm
        deg = gate.g4_units.deg
        mrad = gate.g4_units.mrad
        s = gate.g4_units.s

        # add a material database
        sim.volume_manager.add_material_database(paths.data / "GateMaterials.db")

        #  change world size
        world = sim.world
        world.size = [600 * cm, 500 * cm, 500 * cm]

        ## FIRST DETECTOR ##
        # box
        # translation and rotation like in the Gate macro
        box = sim.add_volume("Box", "box")
        box.size = [200 * mm, 200 * mm, 1050 * mm]
        # box.translation = [0 * cm, 0 * cm, 52.5 * cm]
        # box.rotation = Rotation.from_euler('y',-90,degrees=True).as_matrix()
        box.translation = [0 * cm, 0 * cm, 52.5 * cm]
        box.material = "Vacuum"
        box.color = [0, 0, 1, 1]

        # phantoms
        m = Rotation.identity().as_matrix()

        phantom_s1 = sim.add_volume("Box", "phantom_s1")
        phantom_s1.mother = "box"
        phantom_s1.size = [100 * mm, 100 * mm, 50 * mm]
        phantom_s1.translation = [-50 * mm, 0 * mm, -500 * mm]
        phantom_s1.rotation = m
        phantom_s1.material = "G4_WATER"
        phantom_s1.color = [1, 0, 1, 1]

        phantom_s2 = sim.add_volume("Box", "phantom_s2")
        phantom_s2.mother = "box"
        phantom_s2.size = [100 * mm, 100 * mm, 50 * mm]
        phantom_s2.translation = [50 * mm, 0 * mm, -500 * mm]
        phantom_s2.rotation = m
        phantom_s2.material = "G4_WATER"
        phantom_s2.color = [1, 0, 1, 1]

        # add dose actor
        dose_s1 = sim.add_actor("DoseActor", d0name)
        filename = f"phantom_s1_{label}.mhd"
        dose_s1.output_filename = filename
        dose_s1.attached_to = "phantom_s1"
        dose_s1.size = [25, 25, 1]
        dose_s1.spacing = [4.0, 4.0, 50.0]
        dose_s1.hit_type = "random"
        dose_s1.edep_uncertainty.active = True
        # dose_s1.ste_of_mean = True

        # add dose actor
        dose_s2 = sim.add_actor("DoseActor", d1name)
        filename = f"phantom_s2_{label}.mhd"
        dose_s2.output_filename = filename
        dose_s2.attached_to = "phantom_s2"
        dose_s2.size = [25, 25, 1]
        dose_s2.spacing = [4.0, 4.0, 50.0]
        dose_s2.hit_type = "random"
        dose_s2.edep_uncertainty.active = True

        ## TPS SOURCE ##
        # beamline model
        beamline = BeamlineModel()
        beamline.name = None
        beamline.radiation_types = "proton"

        # polinomial coefficients
        beamline.energy_mean_coeffs = [1, 0]
        beamline.energy_spread_coeffs = [0.4417036946562556]
        beamline.sigma_x_coeffs = [2.3335754]
        beamline.theta_x_coeffs = [2.3335754e-3]
        beamline.epsilon_x_coeffs = [0.00078728e-3]
        beamline.sigma_y_coeffs = [1.96433431]
        beamline.theta_y_coeffs = [0.00079118e-3]
        beamline.epsilon_y_coeffs = [0.00249161e-3]

        # tps
        print("--- Flat spots distribution ---")
        tps = sim.add_source("TreatmentPlanPBSource", label)
        tps.number_of_primaries = nSim
        tps.end_time = 0.5 * s
        tps.flat_generation = flat # from function argument
        tps.beam_model = beamline
        tps.plan_path = ref_path / "TreatmentPlan2Spots_flat_gen_test.txt"
        tps.beam_nr = 1
        tps.gantry_rot_axis = "x"
        tps.particle = "proton"
        tps.position.translation = [0 * cm, 0 * cm, 0 * cm]

        # add stat actor
        self.stats = sim.add_actor("SimulationStatisticsActor", "Stats")
        self.stats.track_types_flag = True

        # physics
        sim.physics_manager.physics_list_name = "FTFP_INCLXX_EMZ"
        sim.physics_manager.set_production_cut("world", "all", 1000 * km)
        # sim.set_user_limits("phantom_a_2","max_step_size",1,['proton'])

        # create output dir, if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # start simulation
        sim.run(start_new_process=True)
        print(self.stats)

        # ----------------------------------------------------------------------------------------------------------------
        # tests
        self.d_names = [d0name,d1name]
        self.edep_paths = [output_path / sim.get_actor(d).get_output_path("edep") for d in self.d_names]
        self.edep_unc_paths = [output_path / sim.get_actor(d).get_output_path("edep_uncertainty") for d in self.d_names]

if __name__ == "__main__":

    flatsim = sim_tps_dose(flat=True)
    wsim = sim_tps_dose(flat=False)
    test = True

    # with and without flat generation, the result in terms of dose should be identical

    # check that the dose output is the same
    for spot in [0,1]:
        print(f"--- Dose image spot {spot} ---")
        test = (
            utility.assert_images(
                flatsim.edep_paths[spot],
                wsim.edep_paths[spot],
                flatsim.stats,
                tolerance=70,
                ignore_value_data2=0,
            )
            and test
        )

    # check that output with flat distribution has better statistics for the spot with less particles
    unc_vec = []
    all_edep_paths = flatsim.edep_paths + wsim.edep_paths
    all_edep_unc_paths = flatsim.edep_unc_paths + wsim.edep_unc_paths
    all_names = flatsim.d_names + wsim.d_names
    for d, du, name in zip(all_edep_paths, all_edep_unc_paths, all_names):
        edep_img = itk.imread(d)
        edep_arr = itk.GetArrayViewFromImage(edep_img)
        unc_img = itk.imread(du)
        unc_arr = itk.GetArrayFromImage(unc_img)
        # fig = utility.plot2D(unc_arr[0,:,:], d_actor, show=True)
        mean_unc = calculate_mean_unc(edep_arr, unc_arr, edep_thresh_rel=0.2)
        unc_vec.append(mean_unc)
        print(f"mean uncertainty {name}: {mean_unc}")

    test = unc_vec[1] < unc_vec[3] and test

    utility.test_ok(test)
