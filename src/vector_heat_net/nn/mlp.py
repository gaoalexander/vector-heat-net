import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, LeakyReLU, Dropout

from .nonlin import BatchNorm1d, VectorNonLin


def MLP(channels, bias=False, nonlin=LeakyReLU(negative_slope=0.2)):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=bias), BatchNorm1d(channels[i]), nonlin)
        for i in range(1, len(channels))
    ])

# def VectorMLP(channels, batchnorm=False, dropout=False):
#     return Seq(*[
#         Seq(Dropout(p=0.5) if (dropout and i > 1) else None, 
#             Lin(channels[i - 1], channels[i], bias=False).to(dtype=torch.cfloat), 
#             VectorNonLin(channels[i], batchnorm=BatchNorm1d(channels[i]) if batchnorm else None))
#         for i in range(1, len(channels))
#     ])

class VectorMLP(nn.Sequential):
    def __init__(self, channels, batchnorm=False, dropout=False, name="VectorMLP"):
        super(VectorMLP, self).__init__()
        
        for i in range(1, len(channels)):
            is_last = (i + 1 == len(channels))

            if dropout and i > 1:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    Dropout(p=0.2)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    channels[i - 1],
                    channels[i],
                    bias=False
                )
#                 ).to(dtype=torch.cfloat),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    VectorNonLin(channels[i], batchnorm=BatchNorm1d(channels[i]) if batchnorm else None)
                )

                
class DropoutComplex(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        # work around unimplemented dropout for complex
        if x.is_complex():
            mask = nn.functional.dropout(torch.ones_like(x.real), p=self.p)
            return x * mask
        else:
            return torch.nn.functional.dropout(x)


class ScalarVectorMLP(torch.nn.Module):
    def __init__(self, channels, nonlin=True, vector_stream=True):
        super(ScalarVectorMLP, self).__init__()
        self.scalar_mlp = MLP(channels, nonlin=LeakyReLU(negative_slope=0.2) if nonlin else torch.nn.Identity())
        self.vector_mlp = None
        if vector_stream:
            self.vector_mlp = VectorMLP(channels)

    def forward(self, x):
        assert self.vector_mlp is None or (self.vector_mlp is not None and type(x) is tuple)

        if type(x) is tuple:
            x, v = x

        x = self.scalar_mlp(x)

        if self.vector_mlp is not None:
            v = self.vector_mlp(v)
            x = (x, v)

        return x

class ScalarVectorIdentity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ScalarVectorIdentity, self).__init__()

    def forward(self, input):
        return input
