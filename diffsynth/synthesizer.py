import torch
import torch.nn as nn
from diffsynth.processor import Gen
import diffsynth.util as util
import math

class Synthesizer(nn.Module):
    """
    defined by a DAG of processors in a similar manner to DDSP
    """

    def __init__(self, dag, name='synth', fixed_params={}, static_params=[]):
        """
        
        Args:
            dag (list of tuples): [(processor, {'param_name':'INPUT_KEY' or 'processor.name'})] 
            ex.)    [   
                    (additive, {'amplitudes':'ADD_AMP', 'harmonic_distribution':'ADD_HARMONIC', 'f0_hz':'ADD_F0'}),
                    (filter, {'input':'additive', 'cutoff':'CUTOFF'}),
                    ...
                    ]
            name (str, optional): Defaults to 'synth'.
            fixed_params: Values of fixed parameters ex.) {'INPUT_KEY': Tensor([(n_frames), param_size])
                          Value=None if the param is added to dict as conditioning later
            static_params: These values use only the last frame of the input_tensor
        """
        super().__init__()
        self.dag = dag
        self.name = name
        self.ext_param_sizes = {}
        self.processor_names = [processor.name for processor, connections in self.dag]
        self.fixed_param_names = list(fixed_params.keys())
        for k, v in fixed_params.items():
            if v is not None:
                self.register_buffer(k, v)
            else:
                setattr(self, k, None)
        self.processors = nn.ModuleList([]) # register modules for .to(device)
        self.dag_summary = {}
        for processor, connections in self.dag:
            self.processors.append(processor)
            # parameters that rely on external input and not outputs of other modules and are not fixed
            ext_params = [k for k, v in connections.items() if v not in self.processor_names+self.fixed_param_names]
            ext_sizes = {connections[k]: desc['size'] for k, desc in processor.param_desc.items() if k in ext_params}
            self.ext_param_sizes.update(ext_sizes)
            # {'ADD_AMP':1, 'ADD_HARMONIC': n_harmonics, 'CUTOFF': ...}
            # summarize dag
            self.dag_summary.update({processor.name +'_'+ input_name: output_name for input_name, output_name in connections.items()})
        self.ext_param_size = sum(self.ext_param_sizes.values())
        self.static_params = static_params


    def fill_params(self, input_tensor, conditioning=None):
        """using network output tensor to fill synth parameter dict
        input_tensor should be 0~1 (ran through sigmoid or tanh)
        scales input according to their type and range

        Args:
            input_tensor (torch.Tensor): Shape [batch, n_frames, input_size]
                if parameters are stationary like a preset, n_frames should be 1
            conditioning: dict of conditions ex.) {'f0_hz': torch.Tensor [batch, n_frames_cond, 1]}
        Returns:
            dag_input: {'amp': torch.Tensor [batch, n_frames, 1], }
        """
        curr_idx = 0
        dag_input = {}
        batch_size = input_tensor.shape[0]
        n_frames = input_tensor.shape[1]
        device = input_tensor.device
        # parameters fed from input_tensor
        for ext_param, param_size in self.ext_param_sizes.items():
            split_input = input_tensor[:, :, curr_idx:curr_idx+param_size]
            if ext_param in self.static_params:
                split_input = split_input[:, -1:, :]
            dag_input[ext_param] = split_input
            curr_idx += param_size
        # Fill fixed_params
        for param_name in self.fixed_param_names:
            param_value = getattr(self, param_name)                
            if param_value is None:
                value = conditioning[param_name] 
            elif len(param_value.shape) == 1:
                if param_name in self.static_params:
                    value = param_value[None, None, :].expand(batch_size, -1, -1).to(device)
                else:
                    value = param_value[None, None, :].expand(batch_size, n_frames, -1).to(device)
            elif len(param_value.shape) == 2:
                value = param_value[None, :, :].expand(batch_size, -1, -1).to(device)
            dag_input[param_name] = value
        return dag_input

    def forward(self, dag_inputs, n_samples):
        """runs input through DAG of processors

        Args:
            dag_inputs (dict): ex. {'INPUT_KEY':Tensor}

        Returns:
            dict: Final output of processor
        """
        outputs = dag_inputs

        for node in self.dag:
            processor, connections = node
            # fixed params are not in 0~1 and do not need to be scaled
            scaled = [k for k in connections if connections[k] in self.fixed_param_names]
            inputs = {key: outputs[connections[key]] for key in connections}
            # list of fixed parameters that
            if n_samples and isinstance(processor, Gen):
                inputs.update({'n_samples': n_samples})
            # Run processor.
            signal = processor.process(scaled_params=scaled, **inputs)

            # Add the outputs of processor for use in subsequent processors
            outputs[processor.name] = signal # audio/control signal output

        #Use the output of final processor as signal
        output_name = self.dag[-1][0].name
        outputs['output'] = outputs[output_name]
        
        return outputs['output'], outputs