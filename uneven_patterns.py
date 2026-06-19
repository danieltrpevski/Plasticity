
import plasticity_experiment as pe
import parameters as p
from neuron import h
import sys
import random as rnd

class Uneven_Patterns(pe.Plasticity_Experiment):

    def create_training_set(self, size):
        training_set = []

        if self.exptype in ['xor_hom_spillover']:
            training_set.extend(rnd.choices(['rs', 'yb', 'ys', 'rb'], weights = [0.4, 0.4, 0.1, 0.1], k = int(p.training_set_size/2)))
            training_set.extend(rnd.choices(['rs', 'yb', 'ys', 'rb'], weights = [0.1, 0.1, 0.4, 0.4], k = int(p.training_set_size/2)))
            if p.extra_training_inputs > 0:
                training_set = ['rs', 'yb', 'ys', 'rb'] + training_set + ['rs', 'yb', 'ys', 'rb']

        elif self.exptype in ['fbp']:
            training_set.extend(rnd.choices(['rs', 'ys'], weights = [0.8, 0.2], k = int(p.training_set_size/2)))
            training_set.extend(rnd.choices(['rs', 'ys'], weights = [0.2, 0.8], k = int(p.training_set_size/2)))
            if p.extra_training_inputs > 0:
                training_set = ['rs', 'ys'] + training_set + ['rs', 'ys']
        else:
            print("From method create_training_set in %s" % type(self))
            print("Exptype '%s' not supported" % self.exptype)
            sys.exit(-1)

        return training_set
