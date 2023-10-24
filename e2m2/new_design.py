import openmdao.api as om


class NewDesign(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("metadata",
                             desc="Dictionary of HF model variables and their metadata")
        self.options.declare("design_vars",
                             desc="Dictionary of design variables and their metadata")

    def setup(self):
        metadata = self.options['metadata']
        dvs = self.options['design_vars']
        for dv, meta in dvs.items():
            shape = metadata[meta['source']]['shape']
            print(shape)
            self.add_input(f"delta_{dv}",
                           val=0,
                           units=meta['units'],
                           distributed=meta['distributed'],
                           shape=shape)
            self.add_input(f"{dv}_k",
                           units=meta['units'],
                           distributed=meta['distributed'],
                           shape=shape)
            self.add_output(dv,
                            units=meta['units'],
                            distributed=meta['distributed'],
                            shape=shape)

            self.declare_partials(dv, [f"delta_{dv}", f"{dv}_k"], method='cs')
            # self.declare_partials(dv, f"delta_{dv}", method='cs')

    def compute(self, inputs, outputs):
        dvs = self.options['design_vars']
        for dv in dvs.keys():
            outputs[dv] = inputs[f"{dv}_k"] + inputs[f"delta_{dv}"]
