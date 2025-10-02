from functools import partial

import flax.linen as nn

from ltc.agents.drl import add_batch_dim


class DLMANetwork(nn.Module):
    num_actions: int = 2

    @nn.compact
    def __call__(self, s):
        dense = partial(nn.Dense, kernel_init=nn.initializers.he_normal())

        s = add_batch_dim(s)
        b, *_ = s.shape
        x = s.reshape(b, -1)

        for _ in range(2):
            x = dense(64)(x)
            x = nn.relu(x)

        for _ in range(2):
            x_res = x
            x = dense(64)(x)
            x = nn.relu(x)
            x = dense(64)(x)
            x = nn.relu(x)
            x = x + x_res

        x = dense(self.num_actions)(x)
        return x
