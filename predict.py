import torch

from src.config import Config
from src.const import INPUTS
from src.dataset import get_dataloaders
from src.model import Seq2Seq


config = Config(
    batch_size=1,
    teacher_forcing_ratio=0.0,
    device=torch.device('cpu'),
)
model = Seq2Seq(config)
model.load_state_dict(torch.load('artifacts/best.pth', map_location='cpu')['model_state_dict'])
model.eval()
train_dataloader, valid_dataloader = get_dataloaders(config)

data_iter = iter(valid_dataloader)
batch = next(data_iter)
output = model(batch[INPUTS])
ids = torch.squeeze(torch.argmax(output, dim=2) + 1).tolist()
print(
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
