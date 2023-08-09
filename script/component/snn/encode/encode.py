"""
本脚本实现自定义脉冲编码构件。
"""

from wuyuan import spaic
import numpy as np

class PoissonEncoding1(spaic.Encoder):
    """
        泊松频率编码，发放脉冲的概率即为刺激强度，刺激强度需被归一化到[0, 1]。
        Generate a poisson spike train.
        time: encoding window ms
        dt: time step
    """
    def __init__(self, shape=None, num=None, dec_target=None, dt=None, coding_method=('poisson', 'spike_counts', '...'),
                 coding_var_name='O', node_type=('excitatory', 'inhibitory', 'pyramidal', '...'), **kwargs):
        super(PoissonEncoding1, self).__init__(shape, num, dec_target, dt, coding_method, coding_var_name, node_type, **kwargs)
        self.unit_conversion = kwargs.get('unit_conversion', 0.1)
        self.run_time = kwargs.get('run_time', 20)
        self.sim_name = None
        self.device = 'cpu'

    def numpy_coding(self, source, device):
        # assert (source >= 0).all(), "Inputs must be non-negative"
        # shape = list(source.shape)
        # spk_shape = [self.time_step] + shape
        # spikes = np.random.rand(*spk_shape).__le__(source * self.dt).astype(float)
        print('PoissonEncoding1 numpy_coding')
        spikes = np.random.rand(int(self.run_time / self.dt), \
            self.source.size).__le__(self.source * self.dt).astype(float)
        return spikes

    def torch_coding(self, source, device):
        # assert (source >= 0).all(), "Inputs must be non-negative"
        # if source.__class__.__name__ == 'ndarray':
        #     source = torch.tensor(source, device=device, dtype=torch.float32)
        # shape = source.shape
        # # source_temp = source.view(shape[0], -1)
        # spk_shape = [self.time_step] + list(shape)
        # spikes = torch.rand(spk_shape, device=device).le(source * self.unit_conversion*self.dt).float()
        print('PoissonEncoding1 torch_coding')
        spikes = np.random.rand(int(self.run_time / self.dt), \
            self.source.size).__le__(self.source * self.dt).astype(float)

        return spikes

spaic.Encoder.register('poisson1', PoissonEncoding1)
