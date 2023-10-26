import numpy as np

import openmdao.api as om


class CalibratedResponse(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("model_metadata")
        self.options.declare("response")
        self.options.declare("calibrated_response_name")

    def setup(self):
        metadata = self.options['model_metadata']
        response = self.options['response']
        calibrated_response_name = self.options['calibrated_response_name']

        resp_shape = metadata[response]['shape']
        resp_units = metadata[response]['units']

        self.add_input(response, shape=resp_shape, units=resp_units)
        self.add_input(f"{response}_bias", shape=resp_shape, units=resp_units)

        self.add_output(calibrated_response_name,
                        shape=resp_shape, units=resp_units)

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        response = self.options['response']
        calibrated_response_name = self.options['calibrated_response_name']

        outputs[calibrated_response_name] = inputs[response] + \
            inputs[f"{response}_bias"]
