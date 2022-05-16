import hydra
import torch
from diffsynth.synthesizer import Synthesizer

def construct_synth_from_conf(synth_conf):
    dag = []
    for module_name, v in synth_conf.dag.items():
        module = hydra.utils.instantiate(v.config, name=module_name)
        conn = v.connections
        dag.append((module, conn))
    synth = Synthesizer(dag, conditioned=synth_conf.conditioned)
    return synth