import openmdao.api as om

class NewDesign(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("design_vars",
                             desc="Dictionary of design variables and their metadata")

    def setup(self):
        dvs = self.options['design_vars']
        for dv, meta in dvs.items():
            self.add_input(f"delta_{dv}",
                           units=meta['units'],
                           distributed=meta['distributed'])
            self.add_input(f"{dv}_k",
                           units=meta['units'],
                           distributed=meta['distributed'])
            self.add_output(dv,
                            units=meta['units'],
                            distributed=meta['distributed'])
            
            # self.declare_partials(dv, [f"delta_{dv}", f"{dv}_k"], method='cs')
            self.declare_partials(dv, f"delta_{dv}", method='cs')

    def compute(self, inputs, outputs):
        dvs = self.options['design_vars']
        for dv in dvs.keys():
            outputs[dv] = inputs[f"{dv}_k"] + inputs[f"delta_{dv}"]
