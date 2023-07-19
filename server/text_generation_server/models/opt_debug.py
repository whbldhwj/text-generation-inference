import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from text_generation_server.models.custom_modeling.opt_modeling import OPTForCausalLM
from text_generation_server.models.opt import OPTSharded
from text_generation_server.models.custom_modeling.bloom_modeling import BloomForCausalLM
from text_generation_server.models.bloom import BLOOMSharded

process_group, rank, world_size = initialize_torch_distributed()
model_id = "facebook/opt-6.7b"
#model_id = "facebook/opt-1.3b"
#model_id = "bigscience/bloom-560m"
revision = None
device = torch.device(f"cuda:{rank}")
dtype = torch.float16

config = AutoConfig.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=False,
)
#import pdb; pdb.set_trace()
filenames = weight_files(model_id, revision=revision, extension=".safetensors")
weights = Weights(
    filenames, device=device, dtype=dtype, process_group=process_group
)
#model = OPTForCausalLM(config, weights)
model = OPTSharded(model_id)
#model = BloomForCausalLM(config, weights)
#model = BLOOMSharded(model_id)