import torch
import torch.nn as nn
from typing import Dict, List

class Synthesizer(nn.Module):
    """
    defined by a DAG of processors in a similar manner to DDSP
    """

    def __init__(self, dag, name='synth', conditioned=[]):
        """
        
        Args:
            dag (list of tuples): [(processor, {'param_name':'INPUT_KEY' or 'processor.name'})] 
            ex.)    [   
                    (additive, {'amplitudes':'ADD_AMP', 'harmonic_distribution':'ADD_HARMONIC', 'f0_hz':'ADD_F0'}),
                    (filter, {'input':'additive', 'cutoff':'CUTOFF'}),
                    ...
                    ]
            name (str, optional): Defaults to 'synth'.
            conditioned: parameters like f0 to be provided externally, without being estimated
        """
        super().__init__()
        self.processors = nn.ModuleList([p for p, _c in dag])
        self.connections = tuple(dict(c) for _p, c in dag)
        self.name = name
        self.ext_param_sizes = {}
        self.processor_names = [processor.name for processor in self.processors]
        self.conditioned_params = list(conditioned)
        self.dag_summary = {}
        for processor, connections in zip(self.processors, self.connections):
            # parameters that rely on external input and not outputs of other modules and are not conditioned
            ext_params = [k for k, v in connections.items() if v not in self.processor_names+self.conditioned_params]
            ext_sizes = {connections[k]: size for k, size in processor.param_sizes.items() if k in ext_params}
            self.ext_param_sizes.update(ext_sizes)
            # {'ADD_AMP':1, 'ADD_HARMONIC': n_harmonics, 'CUTOFF': ...}
            # summarize dag
            self.dag_summary.update({processor.name +'_'+ input_name: output_name for input_name, output_name in connections.items()})
        self.ext_param_size = sum(self.ext_param_sizes.values())

    @torch.jit.export
    def fill_params(self, input_tensor: torch.Tensor, conditioning: Dict[str, torch.Tensor]):
        """using network output tensor to fill synth parameter dict
        input_tensor should be 0~1 (ran through sigmoid)
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
            dag_input[ext_param] = split_input
            curr_idx += param_size
        # Fill conditioned_params
        for param_name in self.conditioned_params:
            dag_input[param_name] = conditioning[param_name]
        return dag_input

    def forward(self, dag_inputs: Dict[str, torch.Tensor], n_samples: int):
        """runs input through DAG of processors

        Args:
            dag_inputs (dict): ex. {'INPUT_KEY':Tensor}

        Returns:
            dict: Final output of processor
        """
        outputs = dag_inputs
        for processor, connections in zip(self.processors, self.connections):
            # fixed params are not in 0~1 and do not need to be scaled
            scaled: List[str] = []
            for k in connections:
                if connections[k] in self.conditioned_params:
                    scaled.append(k)
            inputs = {key: outputs[connections[key]] for key in connections}

            signal = processor.process(inputs, n_samples, scaled_params=scaled)

            # Add the outputs of processor for use in subsequent processors
            outputs[processor.name] = signal # audio/control signal output

        #Use the output of final processor as signal
        output_name = self.processors[-1].name
        outputs['output'] = outputs[output_name]
        
        return outputs['output'], outputs