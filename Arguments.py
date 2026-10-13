import torch

class Arguments:
    def __init__(self, **kwargs): #task, n_qubits, n_layers, fold):

        n_qubits = kwargs.get('n_qubits', 4)
        n_layers = kwargs.get('n_layers', 4)
        task = kwargs.get('task', 'MNIST')
        fold = kwargs.get('fold', 1)
        backend = kwargs.get('backend', 'tq')

        # Qiskit arguments
        self.use_noise_model_train = False
        self.use_noise_model_inference = True
        self.backend_device = 'CPU'

        self.device     = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.clr        = 0.005
        self.qlr        = 0.01

        self.allowed_gates = ['Identity', 'U3', 'data', 'data+U3'] #['Identity', 'RX', 'RY', 'RZ']
        
        self.task      = task
        self.n_qubits   = n_qubits        
        self.epochs     = 1
        self.batch_size = 256
        self.sampling = 5

        self.n_layers = n_layers      
        self.exploration = [0.001, 0.002, 0.003]
        
        self.backend    = backend      # 'tq' for TorchQuantum; 'qi' for Qiskit Aer Simulator; 'qml' for PennyLane
        self.digits_of_interest = [0, 1, 2, 3]
        self.train_valid_split_ratio = [0.95, 0.05]
        self.center_crop = 24
        self.resize = 28
        self.file_single = 'search_space/search_space_mnist_single'
        self.file_enta   = 'search_space/search_space_mnist_enta'
        self.kernel      = 6
        self.fold        = fold
        self.init_weight = 'init_weight_'+ task
        self.SNR = 0.5
        self.strategy = 'mix'
        self.qiskit_seed=170
        self.shots=10000
        self.name='yorktown'
        self.estimator_job_chunk_size = kwargs.get('estimator_job_chunk_size', None)
        self.n_train_samples = None
        self.n_valid_samples = None
        self.n_test_samples = None

        # Optional grouped-average readout for QNet.
        # Example: n_qubits=12 with group_avg_num_groups=2 -> [6, 6].
        self.enable_group_avg = kwargs.get('enable_group_avg', False)
        self.group_avg_num_groups = kwargs.get('group_avg_num_groups', 2)
        self.qubit_group_sizes = kwargs.get('qubit_group_sizes', None)
        if isinstance(self.qubit_group_sizes, tuple):
            self.qubit_group_sizes = list(self.qubit_group_sizes)


        if task == ('MNIST_10' or 'FASHION_10'):
            self.n_qubits   = n_qubits
            
            self.epochs     = 1
            self.batch_size = 256 
            self.sampling = 5
            self.kernel      = 4
            # self.n_train_samples = 20000
            # self.n_valid_samples = 1000
            # self.n_test_samples = 5000
            self.n_train_samples = None
            self.n_valid_samples = None
            self.n_test_samples = None

            self.n_layers = n_layers
            self.base_code = [self.n_layers, 2, 3, 4, 1]
            self.exploration = [0.001, 0.002, 0.003]

            # self.backend    = 'tq'            
            self.digits_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.file_single = 'search_space/search_space_mnist_half_single'
            self.file_enta   = 'search_space/search_space_mnist_half_enta'
            self.fold        = fold

        if task.startswith('QML'):            
            self.batch_size = 32
            self.task_name = task
            
            # Set path based on task content
            if 'Hidden' in task:
                self.path = 'hidden_manifold'
            elif 'Linear' in task:
                self.path = 'linearly_separable'
            else:
                self.path = None  # or set a default path
            
