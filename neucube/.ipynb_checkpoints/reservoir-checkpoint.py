import torch
from tqdm import tqdm
import math
from .topology import small_world_connectivity
from .utils import print_summary

class Reservoir():
  def __init__(self, cube_shape=(10,10,10), inputs=None, coordinates=None, mapping=None, c=1.2, l=1.6, c_in = 0.9, l_in = 1.2):
    """
    Initializes the reservoir object.

    Parameters:
        cube_shape (tuple): Dimensions of the reservoir as a 3D cube (default: (10,10,10)).
        inputs (int): Number of input features.
        coordinates (torch.Tensor): Coordinates of the neurons in the reservoir.
                                    If not provided, the coordinates are generated based on `cube_shape`.
        mapping (torch.Tensor): Coordinates of the input neurons.
                                If not provided, random connectivity is used.
        c (float): Parameter controlling the connectivity of the reservoir.
        l (float): Parameter controlling the connectivity of the reservoir.
        c_in (float): Parameter controlling the connectivity of the input neurons.
        l_in (float): Parameter controlling the connectivity of the input neurons.
    """
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

    if coordinates is None:
      self.n_neurons = math.prod(cube_shape)
      x, y, z = torch.meshgrid(torch.linspace(0, 1, cube_shape[0]), torch.linspace(0, 1, cube_shape[1]), torch.linspace(0, 1, cube_shape[2]), indexing='xy')
      pos = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    else:
      self.n_neurons = coordinates.shape[0]
      pos = coordinates

    dist = torch.cdist(pos, pos)
    conn_mat = small_world_connectivity(dist, c=c, l=l) / 100
    inh_n = torch.randint(self.n_neurons, size=(int(self.n_neurons*0.2),))
    conn_mat[:, inh_n] = -conn_mat[:, inh_n]

    if mapping is None:
      input_conn = torch.where(torch.rand(self.n_neurons, inputs) > 0.95, torch.ones_like(torch.rand(self.n_neurons, inputs)), torch.zeros(self.n_neurons, inputs)) / 50
    else:
      dist_in = torch.cdist(coordinates, mapping, p=2)
      input_conn = small_world_connectivity(dist_in, c=c_in, l=l_in) / 50

    self.w_latent = conn_mat.to(self.device)
    self.w_in = input_conn.to(self.device)

  def simulate(self, X, mem_thr=0.1, refractory_period=5, train=True, verbose=True):
    """
    Simulates the reservoir activity given input data.

    Parameters:
        X (torch.Tensor): Input data of shape (batch_size, n_time, n_features).
        mem_thr (float): Membrane threshold for spike generation.
        refractory_period (int): Refractory period after a spike.
        train (bool): Flag indicating whether to perform online training of the reservoir.
        verbose (bool): Flag indicating whether to display progress during simulation.

    Returns:
        torch.Tensor: Spike activity of the reservoir neurons over time, of shape (batch_size, n_time, n_neurons).
    """
    self.batch_size, self.n_time, self.n_features = X.shape

    spike_rec = torch.zeros(self.batch_size, self.n_time, self.n_neurons)

    for s in tqdm(range(X.shape[0]), disable = not verbose):

      spike_latent = torch.zeros(self.n_neurons).to(self.device)
      mem_poten = torch.zeros(self.n_neurons).to(self.device)
      refrac = torch.ones(self.n_neurons).to(self.device)
      refrac_count = torch.zeros(self.n_neurons).to(self.device)
      spike_times = torch.zeros(self.n_neurons).to(self.device)

      for k in range(self.n_time):

        spike_in = X[s,k,:]
        spike_in = spike_in.to(self.device)

        refrac[refrac_count < 1] = 1

        I = torch.sum(self.w_in*spike_in, axis=1)+torch.sum(self.w_latent*spike_latent, axis=1)
        mem_poten = mem_poten*torch.exp(torch.tensor(-(1/40)))*(1-spike_latent)+(refrac*I)

        spike_latent[mem_poten >= mem_thr] = 1
        spike_latent[mem_poten < mem_thr] = 0

        refrac[mem_poten >= mem_thr] = 0
        refrac_count[mem_poten >= mem_thr] = refractory_period
        refrac_count = refrac_count-1

        if train == True:
          t_constant = 3
          self.aux = k-spike_times
          self.pre_w = 0.0001*torch.exp(-self.aux/t_constant)*torch.gt(self.aux,0).int()
          self.pos_w = -0.01*torch.exp(-self.aux/t_constant)*torch.gt(self.aux,0).int()
          pre_updates = self.pre_w*torch.gt((self.w_latent.T*spike_latent).T, 0).int()
          pos_updates = self.pos_w*torch.gt(self.w_latent*spike_latent, 0).int()

          self.w_latent += pre_updates
          self.w_latent += pos_updates

        spike_times[mem_poten >= mem_thr] = k
        
        spike_rec[s,k,:] = spike_latent

    return spike_rec

  def summary(self):
    """
    Prints a summary of the reservoir.
    """
    res_info = [["Neurons", str(self.n_neurons)],
                ["Reservoir connections", str(sum(sum(self.w_latent != 0)).item())],
                ["Input connections", str(sum(sum(self.w_in != 0)).item())],
                ["Device", str(self.device)]]

    print_summary(res_info)