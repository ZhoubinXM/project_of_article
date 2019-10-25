import torch.nn as nn
import numpy as np
import collections
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
"""
Gumbel Softmax Sampler // Gumbel Softmax采样器
Requires 2D input [batchsize, number of categories]
Does not support sinlge binary category. Use two dimensions with softmax instead.
Gumbel Softmax采样器
需要2D输入[批量，类别数]
不支持sinlge二进制类别。 请改用带有softmax的两个尺寸。
"""
class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1) 从一个这样的分布中进行采样"""
        noise = torch.rand(shape)  # 从0-1中随机采样一个shape大小的tensor
        noise.add_(eps).log_().neg_() # add_() 对于原始的每个数据进行相加的操作 _带有下滑线是直接运算形式 分别加上一个值 取对数 取负
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return Variable(noise).cuda() # gpu可用就加载到gpu中
        else:
            return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_() # clone返回与原tensor相同大小数据类型的tensor uniform用0-1的均匀数字填充tensor
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(-1) #size 返回tensor大小
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)

        gumble_trick_log_prob_samples = logits + gumble_samples_tensor

        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, 1)
        return soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        “”
         来自Gumbel-Softmax分布的样本，也可以离散化。
         ARGS：
         logits：[batch_size，n_class]未标准化的对数概率
         temperature：非负标量
         hard：如果为True，则采用argmax，但要区分w.r.t. 软样本y
         返回值：
         [batch_size时，n_class]从Gumbel-Softmax分布中采样本。
         如果hard = True，则返回的样本将为one-hot一维向量，否则它将为是一个概率分布，各个类别的总和为1
         “”
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if hard:
            _, max_value_indexes = y.data.max(1, keepdim=True) #返回每一行最大值的索引
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y

    # 定义的前向传播
    def forward(self, logits, temp=1, force_hard=False):

        samplesize = logits.size()

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True)


# 创建一个mlp多层感知器 参数是 激活函数 标准化 dropout（防止过拟合）
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    # MLP[128+1024,1024,128] mlp的维度是输入128 输出1024
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):  # zip函数将组成一个元祖一个元祖的形式用来进行循环变量
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

# 随机采样噪声：高斯分布 均匀分布
def get_noise(shape, noise_type, gpu):
    if noise_type == 'gaussian':
        if gpu:
            return torch.randn(*shape).cuda()
        else:
            return torch.randn(*shape).cpu()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


# 创建一个编码器（生成器和对抗器的编码部分）
class Encoder(nn.Module):
    """
    Encoder is part of both TrajectoryGenerator andTrajectoryDiscriminator
    编码器是生成器和判别器的一部分
    """
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0, use_gpu=1
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        self.spatial_embedding = nn.Linear(2, embedding_dim)

        self.use_gpu = use_gpu

    # 做全零初始化 [层数 批次 lstm输出维度]
    def init_hidden(self, batch):
        if self.use_gpu:
            return (
                torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
                torch.zeros(self.num_layers, batch, self.h_dim).cuda()
            )
        else:
            return (
                torch.zeros(self.num_layers, batch, self.h_dim).cpu(),
                torch.zeros(self.num_layers, batch, self.h_dim).cpu()
            )

    # 定义的前向传播的过程
    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        batch = obs_traj.size(1) # 取出batch
        if self.use_gpu:
            obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2)) # view进行tensor的重组
        else:
            # 执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
            obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))

        obs_traj_embedding = obs_traj_embedding.view( -1, batch, self.embedding_dim)

        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)  # state是（h_n,c_n） 详情见公式1
        final_h = state[0] # 输出最终lstm的隐藏状态 将包含轨迹的所有隐藏信息  公式2
        return final_h


# 创建一个解码器（解码器用于生成器中 用于连接编码器的输出）
class Decoder(nn.Module):
    """
    Decoder is part of TrajectoryGenerator
    """
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8, use_gpu=1
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )
            # MLP[128+1024,1024,128]
            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            # 调用make_mlp函数创建一个mlp多层感知器网络
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

        self.use_gpu = use_gpu

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        输入：
         -last_pos：形状的张量（batch，2）
         -last_pos_rel：形状的张量（batch，2）
         -state_tuple：（hh，ch）每个形状的张量（num_layers，batch，h_dim）
         -seq_start_end：在批处理中界定序列的元组列表
         输出：
         -pred_traj：形状的张量（self.seq_len，batch，2）
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]  # 取出网络的隐藏状态

                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)[0]

                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1)

                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


# 池化层用于共享行人信息
class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim  # 64+64=128
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]  # [128,512,1024]

        self.spatial_embedding = nn.Linear(2, embedding_dim)  # 空间嵌入 embedding层
        # 创建池化网络的mlp多层感知器
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

        self.gumbel = GumbleSoftmax()

    '''
    代码实现时，计算相对位置信息时显得比较巧妙，例如在同场景的行人位置信息，
    代码通过两次不同的repeat策略将原有N个人的位置信息重复N次，
    从而形成了[P0, P0, P0, ...] [P1, P1, P1, ...] ... 和 [P0, P1, P2, ...] [P0, P1, P2, ...] ..两个矩阵，
    通过矩阵相减即可得到一个N*N行的矩阵，第i行是第i%N个人相对于第i/N个人的相对位置
    '''
    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        1. Compute relative positions between the pedestrians.
        2. Concatenate relative positions with each pedestrian's hidden state.
        3. Feed concatenation into MLP
        4. Pool elementwise to compute each pedestrian's pooling vector.
        5. The obtained pooling vectors encode interactions between all pairs of pedestrians, there is exactly one
           pooling vector per pedestrian in the scene.

        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start

            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]

            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)  # stack vertically

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)  # stack vertically
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)  # don't stack, repeat in place

            # 得到行人的end_pos间的相对关系，并交给感知机去具体处理。
            # 每个行人与其他行人的相对位置关系由num_ped项，合计有num_ped**2项。
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2  # compute distances between all pairs of pedestrians
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)

            # 拼接H_i和处理过的pos，放入多层感知机，最后经过maxPooling。
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h_raw = self.mlp_pre_pool(mlp_h_input)
            curr_pool_raw_view = curr_pool_h_raw.view(num_ped, num_ped, -1)  # rearrange such that each a row represents a pedestrian

            # Max Pooling: Pool elementwise to compute each pedestrian's pooling vector
            #curr_pool_h, idxs = curr_pool_raw_view.max(1)
            #computePoolingStats = True


            # gumbel pooling
            activationHack = torch.log(curr_pool_raw_view + 10**(-8))
            gumbelSelector = self.gumbel(activationHack, force_hard=True)
            curr_pool_h = torch.sum(gumbelSelector * curr_pool_raw_view, dim=1)
            _, idxs = gumbelSelector.max(1)
            computePoolingStats = True


            """
            # closest instead of max
            # curr_rel_pos is distance matrix
            # compute index of smallest value (distance) in each row
            # in curr_pool_raw use the computed indices instead of .max()

            curr_rel_posGrouped = []
            assert len(curr_rel_pos) % num_ped == 0
            for i in range(0, curr_rel_pos.shape[0], num_ped):
                curr_rel_posGrouped.append(curr_rel_pos[i: i+num_ped])
                
                # compute min euclidian distance of current coordinates to 0
                # index of min is closest pedestrian 
                # 

            # euclidian distances between curr pedestrian and all others
            #distancesToCurrPedestrian = torch.sqrt(torch.sum((curr_end_pos - curr_end_pos[currPedestrianIdx]) ** 2, dim=1))

            # distance of closest pedestrian that is not the current pedestrian himself
            #distOfClosestPed = torch.min(torch.cat([distancesToCurrPedestrian[0:currPedestrianIdx], distancesToCurrPedestrian[currPedestrianIdx + 1:]]))
            """


            # Mean Pooling instead of max
            #curr_pool_h = curr_pool_raw_view.mean(1)  # compute columnwise mean instead of max
            #idxs = []   # no idxs when using mean as all values get "selected"
            #computePoolingStats = False


            """
            # Random Pooling instead of max
            # TODO get index of randomly selected value IF you want to compute pooling statistics
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            curr_pool_h = []
            for matrix in curr_pool_raw_view:
                curr_pool_h.append([random.choice(column) for column in zip(*matrix)])
            curr_pool_h = torch.Tensor(curr_pool_h)  # convert list to tensor
            curr_pool_h = curr_pool_h.to(device)
            idxs = []
            computePoolingStats = False
            """


            # statistics are only computable on CPU, this is fixable by converting the cuda tensors to cpu and then computing the statistics
            if computePoolingStats and not idxs[0].is_cuda:


                # dict that maps % of included pedestrians, to number of pooling vectors with that %
                includedPedestrians = collections.Counter()

                # same as includedPedestrians but maps % of OTHER (not self) pedestrians included, to num of vectors
                includedPedestriansNoSelfIncl = collections.Counter()

                # dict that maps num of times a pedestrian is included in his own pooling vector, to number of pedestrians
                includedSelf = collections.Counter()

                ratioChosenAndClosest = collections.Counter()

                for currPedestrianIdx, currPoolingVectorIdxs in enumerate(idxs):
                    uniqueIndices = set(np.asarray(currPoolingVectorIdxs))  # unique indices in the pooling vector
                    includedPedestrians[len(uniqueIndices) / num_ped] += 1  # % of pedestrians included in pooling vec

                    if (currPedestrianIdx in uniqueIndices):
                        uniqueIndices.remove(currPedestrianIdx)
                    includedPedestriansNoSelfIncl[len(uniqueIndices) / (num_ped - 1)] += 1

                    numOfSelfInclusions = int(sum(currPoolingVectorIdxs == currPedestrianIdx))
                    includedSelf[numOfSelfInclusions / len(currPoolingVectorIdxs)] += 1  # % of self inclusions in vector



                    distancesToCurrPedestrian = torch.sqrt(torch.sum((curr_end_pos - curr_end_pos[currPedestrianIdx]) ** 2, dim=1))

                    distOfClosestPed = torch.min(torch.cat([distancesToCurrPedestrian[0:currPedestrianIdx], distancesToCurrPedestrian[currPedestrianIdx+1:]]))

                    for currPoolValueIdx in currPoolingVectorIdxs:
                        if currPoolValueIdx == currPedestrianIdx:
                            ratioChosenAndClosest[-1] += 1  # value comes from currPedestrian himself (self inclusion)
                        else:
                            ratio = float(distancesToCurrPedestrian[currPoolValueIdx] / distOfClosestPed)
                            assert ratio >= 1
                            ratioChosenAndClosest[ratio] += 1
            else:
                includedPedestrians, includedPedestriansNoSelfIncl, includedSelf, ratioChosenAndClosest = \
                    collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()


            pool_h.append(curr_pool_h)

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h, (includedPedestrians, includedPedestriansNoSelfIncl, includedSelf, ratioChosenAndClosest)


# social lstm 部分（改版的论文中是否可用？）
#TODO:try using this model in the last paper to join the part of the social factor.
class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


# 生成器的代码部分
'''
embeding位于encoder代码下 
embedding->encoder->poolingnet->mlp->decoder->mlp
代码分析博客园
'''
class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=None,
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
        use_gpu=0
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.use_gpu = use_gpu

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_gpu=use_gpu
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
            use_gpu=use_gpu
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim==None:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type, self.use_gpu)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]

            pool_h, poolingStatistics = self.pool_net(final_encoder_h, seq_start_end, end_pos)

            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        if self.use_gpu:
            decoder_c = torch.zeros(
                self.num_layers, batch, self.decoder_h_dim
            ).cuda()
        else:
            decoder_c = torch.zeros(
                self.num_layers, batch, self.decoder_h_dim
            ).cpu()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        if self.pooling_type:
            return pred_traj_fake_rel, poolingStatistics
        else:
            return pred_traj_fake_rel, []


# 判别器部分（用到上面的编码器模块代码）
class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local', use_gpu=1
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_gpu=use_gpu
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            #classifier_input = self.pool_net(final_h.squeeze(), seq_start_end, traj[0])
            # [0] because you changed pooling return
            classifier_input = self.pool_net(final_h.squeeze(), seq_start_end, traj[0])[0]
        scores = self.real_classifier(classifier_input)
        return scores
