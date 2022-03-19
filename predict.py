import torch
from catalyst.utils import set_global_seed

from src.config import Config
from src.const import INPUTS
from src.dataset import get_dataloaders
from src.model import Seq2Seq
from src.train_utils import BeamGenerator

config = Config(
    batch_size=1,
    teacher_forcing_ratio=0.0,
    device=torch.device('cpu'),
)
set_global_seed(config.seed)
model = Seq2Seq(config)
model.load_state_dict(torch.load('artifacts/best.pth', map_location='cpu')['model_state_dict'])
model.eval()

beam = BeamGenerator()

# input_string = 'print "hello world"'
# encoded_input = config.intent_vocab.encode(input_string)
# encoded_input = [1] + encoded_input + [2]
# input_tensor = torch.unsqueeze(torch.tensor(encoded_input), 0).T
# len_tensor = torch.tensor([len(encoded_input)])
# input_data = (input_tensor, len_tensor, torch.zeros(12, 1))

train_dataloader, valid_dataloader = get_dataloaders(config)
data_iter = iter(valid_dataloader)
batch = next(data_iter)
batch = next(data_iter)
batch = next(data_iter)
batch = next(data_iter)
batch = next(data_iter)
batch = next(data_iter)

output = model(batch[INPUTS])
# output = model(input_data)

result = beam(
    torch.squeeze(output),
    max_steps_n=output.shape[0],
    beamsize=1000,
    return_hypotheses_n=15,
)
for score, ids in result:
    print(config.snippet_vocab.decode(ids))
print('\n\n')

ids = torch.squeeze(torch.argmax(output, dim=2)).tolist()
print(
    # config.intent_vocab.decode(torch.squeeze(input_data[0]).tolist()),
    config.intent_vocab.decode(torch.squeeze(batch[INPUTS][0]).tolist()),
    '\n\n',
    config.snippet_vocab.decode(torch.squeeze(batch[INPUTS][2]).tolist()),
    '\n\n',
    config.snippet_vocab.decode(ids),
)
# text = 'print current date in python'
# while text[-1] != config.end_sent:
#     token_ids = config.vocab(text)
#     inputs = torch.unsqueeze(torch.tensor(token_ids, dtype=torch.long), dim=0)
#     output = torch.squeeze(torch.argmax(model(inputs), dim=1))
#     text = text + config.vocab.lookup_tokens([output.tolist()[-1]])
#
# print(' '.join(text[1:-1]))
