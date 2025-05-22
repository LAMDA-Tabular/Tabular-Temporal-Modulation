from model.methods.tabm_temporal import TabM_TemporalMethod


class TabM_ModulatedMethod(TabM_TemporalMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)


    def construct_model(self, model_config = None):
        from model.models.tabm_modulated import TabM_Modulated
        if model_config is None:
            model_config = self.args.config['model']
        self.model = TabM_Modulated(
            n_num_features=self.d_in,
            cat_cardinalities=self.categories,
            n_classes=self.d_out,
            t_mean = self.args.t_mean,
            t_std = self.args.t_std,
            **model_config
        ).to(self.args.device)