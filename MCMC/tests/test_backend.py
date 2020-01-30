# -*- coding: utf-8 -*-


import unittest
from MCMC.backends.backend import Backend

class TestBackend(unittest.TestCase):

    def setUp(self):
        ICs = {'x': 3, 'y': 10}
        self.backend = Backend(ICs)

    def tearDown(self):
        pass

    def test_current_samples(self):
        self.assertDictEqual(self.backend.current_samples, {'x': 3, 'y':10})

    def test_save_step_accept(self):
        new_samples, loss_new, accepted = {'x': -2, 'y': -4}, 1234, True
        self.backend.save_step(new_samples, loss_new, accepted)
        self.assertDictEqual(self.backend.current_samples, {'x': -2, 'y': -4})
        self.assertEqual(self.backend.counter_params_accept, 1)
        self.assertEqual(self.backend.loss_current, 1234)
        self.assertEqual(self.backend.log_post_list, [1234])

    def test_save_step_reject(self):
        new_samples, loss_new, accepted = {'x': -2, 'y': -4}, 1234, False
        self.backend.save_step(new_samples, loss_new, accepted)
        self.assertDictEqual(self.backend.current_samples, {'x': -2, 'y': -4})
        self.assertEqual(self.backend.counter_params_accept, 0)
        self.assertEqual(self.backend.log_post_list, [1234])
