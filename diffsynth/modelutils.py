import hydra
import torch
from diffsynth.synthesizer import Synthesizer

def construct_synth_from_conf(synth_conf):
    dag = []
    for module_name, v in synth_conf.dag.items():
        module = hydra.utils.instantiate(v.config, name=module_name)
        conn = v.connections
        dag.append((module, conn))
    fixed_p = synth_conf.fixed_params
    fixed_p = {} if fixed_p is None else fixed_p
    fixed_p = {k: None if v is None else v*torch.ones(1) for k, v in fixed_p.items()}
    synth = Synthesizer(dag, fixed_params=fixed_p, static_params=synth_conf.static_params)
    return synth