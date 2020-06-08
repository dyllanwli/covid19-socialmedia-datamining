# Import packages
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os

class FuzzyAccuracy():
    def __init__(self):
        
        x_universe = np.arange(0, 1, 0.01)
        y_universe = np.arange(0, 1, 0.01)
        
        self.x_accuracy, self.y_result = self.fuzz_accuracy_tran(x_universe, y_universe)
        self.x_accuracy.view()
        self.y_result.view()
        
    # https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119994374
    def fuzz_accuracy_sigmf(self, topic_index_list, x_universe, y_universe):
        x_topic = ctrl.Antecedent(topic_index_list, 'topic')
        y_accuracy  = ctrl.Consequent(y_universe,'accuracy')

        y_accuracy['absolutely_wrong'] = fuzz.sigmf(y_accuracy.universe, 0.1, 0.3)
        y_accuracy['understandable_wrong'] = fuzz.dsigmf(y_accuracy.universe, 0.3, 0.2, 0.4, 0.2)
        y_accuracy['reasonable'] = fuzz.dsigmf(y_accuracy.universe, 0.5, 0.2, 0.6, 0.2)
        y_accuracy['good'] = fuzz.dsigmf(y_accuracy.universe, 0.7, 0.2, 0.8, 0.2)
        y_accuracy['absolutely_right'] = fuzz.sigmf(y_accuracy.universe, 0.9, 0.2)

        return x_topic

    def fuzz_accuracy_tran(self, x_universe, y_universe):
        x_accuracy  = ctrl.Antecedent(x_universe,'x_accuracy')
        y_result = ctrl.Consequent(y_universe,'percentage')
        
        l1 = [0, 0, 0.05, 0.3]
        l2 = [0.20, 0.35, 0.5]
        l3 = [0.4, 0.55, 0.7]
        l4 =  [0.6, 0.8, 1, 1]

        x_accuracy['understandable'] = fuzz.trapmf(x_accuracy.universe, [0, 0, 0.05, 0.2])
        x_accuracy['reasonable'] = fuzz.trimf(x_accuracy.universe, [0.10, 0.25, 0.4])
        x_accuracy['good'] = fuzz.trimf(x_accuracy.universe,  [0.3, 0.45, 0.6])
        x_accuracy['absolutely_right'] = fuzz.trapmf(x_accuracy.universe,  [0.5, 0.7, 1, 1])

        y_result['understandable'] = fuzz.trapmf(y_result.universe, [0, 0, 0.1, 0.3])
        y_result['reasonable'] = fuzz.trimf(y_result.universe,  [0.15, 0.35, 0.55])
        y_result['good'] = fuzz.trimf(y_result.universe, [0.35, 0.55, 0.75])
        y_result['absolutely_right'] = fuzz.trapmf(y_result.universe,[0.6, 0.8, 1, 1])
        
        return x_accuracy, y_result

    def setting_control_system(self, x_accuracy, y_result, input_value):
        rule2 = ctrl.Rule(x_accuracy['understandable'], y_result['understandable'])
        rule3 = ctrl.Rule(x_accuracy['reasonable'], y_result['reasonable'])
        rule4 = ctrl.Rule(x_accuracy['good'], y_result['good'])
        rule5 = ctrl.Rule(x_accuracy['absolutely_right'], y_result['absolutely_right'])

        accuracy_ctrl = ctrl.ControlSystem([rule2, rule3, rule4, rule5])
        accuracy_simu = ctrl.ControlSystemSimulation(accuracy_ctrl)
        accuracy_simu.input['x_accuracy'] = input_value
        accuracy_simu.compute()
        return accuracy_simu.output['percentage']
    
    def caculate(self, input_value):
        result = self.setting_control_system(self.x_accuracy, self.y_result, input_value)
        return int(round(result*5))

FA = FuzzyAccuracy()