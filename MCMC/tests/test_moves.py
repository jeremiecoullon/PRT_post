# -*- coding: utf-8 -*-

import unittest
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from MCMC.moves.lwr_moves.del_Cast_move import DelCastMove
from MCMC.moves import gaussian
from MCMC.gp.ou_process import OUProcess
from MCMC.moves.lwr_moves import utils
from MCMC.moves.nonparametric_moves.utils import transform_section_dict, create_N_y

class TestMHMove(unittest.TestCase):

    def test_gaussian_cov_1(self):
        param_info = {'x': 2, 'y': -1}
        move = gaussian.GaussianMove(param_info)
        assert np.array_equal(move.cov, np.eye(len(param_info)))

    def test_gaussian_cov_2(self):
        param_info = {'x': 2, 'y': -1}
        x_rand = np.random.random(size=(2,2))
        cov = np.dot(x_rand.T, x_rand)
        move = gaussian.GaussianMove(param_info, cov)
        assert np.array_equal(move.cov, cov)

    def test_gaussian_move_cov_wrong_dim(self):
        """
        Instantiating a gaussian with a covariance shape that doesn't match the parameters
        should raise ValueError
        """
        ICs = {'x': 0}
        cov = np.array([[2,-1], [-1, 2]])
        self.assertRaises(ValueError, gaussian.GaussianMove, param_info=ICs, cov=cov)

    def test_get_proposal(self):
        param_info = {'x': 2, 'y': -1}
        x_rand = np.random.random(size=(2,2))
        cov = np.dot(x_rand.T, x_rand)
        move = gaussian.GaussianMove(param_info, cov)
        new_samples, log_hastings = move.get_proposal(current_samples={'x': 2, 'y': -1.4})
        assert log_hastings == 0
        assert type(new_samples) == dict
        assert len(new_samples) == 2

class TestJointMove(unittest.TestCase):

    def test_phi_symmetric_outlet(self):
        """
        Test that the function phi is symmetric
        Namely: phi^-1(phi(BC)) != BC
        """
        FD1 = {'rho_j': 442.0, 'u': 3.246, 'w': 0.0443, 'z': 167.5}
        FD2 = {'rho_j': 438.31, 'u': 3.27, 'w': 0.06, 'z': 169.013}
        phi = utils.build_phi_del_Cast(FD_1=FD1, FD_2=FD2, t_crit=37, BC_type="BC_outlet", dataset="Sim")
        phi_inv = utils.build_phi_del_Cast(FD_1=FD2, FD_2=FD1, t_crit=37, BC_type="BC_outlet", dataset="Sim")

        BC = np.arange(0,400,1, dtype=float)
        BC_mapped = phi(BC)
        BC_test = phi_inv(BC_mapped)
        assert np.allclose(BC, BC_test)

    def test_phi_symmetric_inlet(self):
        FD1 = {'rho_j': 442.0, 'u': 3.246, 'w': 0.0443, 'z': 167.5}
        FD2 = {'rho_j': 438.31, 'u': 3.27, 'w': 0.06, 'z': 169.013}
        phi = utils.build_phi_del_Cast(FD_1=FD1, FD_2=FD2, t_crit=45, BC_type="BC_inlet", dataset="Sim")
        phi_inv = utils.build_phi_del_Cast(FD_1=FD2, FD_2=FD1, t_crit=45, BC_type="BC_inlet", dataset="Sim")

        BC = np.arange(0,400,1, dtype=float)
        BC_mapped = phi(BC)
        BC_test = phi_inv(BC_mapped)
        assert np.allclose(BC, BC_test)


class TestSamplerMoves(unittest.TestCase):
    """
    Test instantiating moves for both Del Castillo FD
    """

    def setUp(self):
        self.section_dict = {
            'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 90, 'omega': 0.1},
            'section_2': {'param': 'BC_outlet', 'cut1':0, 'cut2': 90, 'omega':0.1},
        }
        self.move_probs = [0.3, 0.3, 0.4]
        self.config_dict = {'my_analysis_dir': '2018/Test_Folder',
                        'run_num': 1,
                        'data_array_dict':
                                # Dataset 1, shorter
                                {'flow': 'data_array_70108_flow_shorter.csv',
                                'density': 'data_array_70108_density_shorter.csv'},
                        'upload_to_S3': False,
                        'save_chain': False,
                        'w_transf_type': 'inv',
                        'comments': "Test MCMC",
                        'ratio_times_BCs': 1,
                              }

    def tearDown(self):
        pass

    def delCastsetup(self):
        self.ICs = OrderedDict([('z', 170), ('rho_j', 460), ('u', 3.1), ('w', 1/0.15),
                ('BC_outlet', np.ones(90)), ('BC_inlet', np.ones(90))])
        self.cov, self.cov_joint = np.eye(4), np.eye(4)
        self.move = DelCastMove(param_info=self.ICs, cov=self.cov, cov_joint=self.cov_joint,
                        section_dict=self.section_dict, move_probs=self.move_probs,
                        config_dict=self.config_dict, w_transf_type='inv')

    def run_move_tests(self, current_samples, FD_type):
        """
        Test FD|BC, BC Gibbs, and joint FD;BC move for either del Castillo
        """
        # create ordered dicts
        if 'z' in current_samples.keys():
            current_FD = OrderedDict({k: current_samples[k] for k in ['z','rho_j','u','w']})
        elif 'alpha' in current_samples.keys():
            current_FD = OrderedDict({k: current_samples[k] for k in ['alpha', 'beta']})
        current_BCs = OrderedDict({k: current_samples[k] for k in ['BC_outlet', 'BC_inlet']})
        new_samples = OrderedDict([(k, v) for k, v in current_samples.items()])

        # propose FD|BC move
        new_move_samples, log_hastings = self.move.propose_FD(new_samples=new_samples, current_FD=current_FD,
                    current_BCs=current_BCs, FD_move=self.move.FD_move, joint_BC_move=False)
        if FD_type == "del_Cast":
            assert list(new_move_samples.keys()) == ['z', 'rho_j', 'u', 'w', 'BC_outlet', 'BC_inlet']
        else:
            raise ValueError("FD_type should be 'del_Cast'")
        assert log_hastings == 0

        # joint move
        if FD_type == "del_Cast":
            new_move_samples, log_hastings = self.move.propose_FD(new_samples=new_samples, current_FD=current_FD,
                        current_BCs=current_BCs, FD_move=self.move.FD_joint_move, joint_BC_move=True)
            assert new_move_samples['BC_outlet'].shape == (90,)
            assert new_move_samples['BC_inlet'].shape == (90,)
            assert (type(log_hastings) == int) or (type(log_hastings) == float)
            # self.assertRaises(ValueError, self.move.propose_FD, new_samples=new_samples, current_FD=current_FD,
            #             current_BCs=current_BCs, FD_move=self.move.FD_joint_move, joint_BC_move=True)

        # BC Gibbs
        log_current_BCs = OrderedDict({k: np.log(current_BCs[k]) for k in ['BC_outlet', 'BC_inlet']})
        log_new_BC_samples, log_hastings = self.move.BC_move.get_proposal(log_current_BCs)
        assert list(log_new_BC_samples) == ['BC_outlet', 'BC_inlet']
        assert (type(log_hastings) == np.float64)


    def test_run_DelCastMove(self):
        """
        Instantiate a del Castillo move and test all 3 moves
        """
        self.delCastsetup()
        current_samples = OrderedDict([('z', 170), ('rho_j', 460), ('u', 3.1), ('w', 1/0.15),
                ('BC_outlet', np.ones(90)), ('BC_inlet', np.ones(90))])

        self.run_move_tests(current_samples=current_samples, FD_type="del_Cast")

    def test_transform_section_dict(self):
        """
        Test transforming section_dict for high resolution BCs
        """
        section_dict_test = {
            'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2':60, 'omega': 0.07},
            'section_2': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 22, 'omega':0.08},
            'section_3': {'param': 'BC_inlet', 'cut1': 20, 'cut2': 37, 'omega':0.13},
            'section_4': {'param': 'BC_inlet', 'cut1': 35, 'cut2': 52, 'omega':0.3},
            'section_5': {'param': 'BC_inlet', 'cut1': 50, 'cut2': 60, 'omega':0.2},
        }
        transf_sd = transform_section_dict(section_dict=section_dict_test, ratio_times=40)
        assert transf_sd == {'section_1': {'cut1': 0, 'cut2': 2361, 'omega': 0.07, 'param': 'BC_inlet'},
                             'section_2': {'cut1': 0, 'cut2': 841, 'omega': 0.08, 'param': 'BC_inlet'},
                             'section_3': {'cut1': 800, 'cut2': 1441, 'omega': 0.13, 'param': 'BC_inlet'},
                             'section_4': {'cut1': 1400, 'cut2': 2041, 'omega': 0.3, 'param': 'BC_inlet'},
                             'section_5': {'cut1': 2000, 'cut2': 2361, 'omega': 0.2, 'param': 'BC_inlet'}}
