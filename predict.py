import torch

from src.config import Config
from src.dataset import get_dataloaders
from src.model import Model


config = Config()
model = Model(config)
model.load_state_dict(torch.load('artifacts/best.pth', map_location='cpu')['model_state_dict'])
train_dataloader, valid_dataloader = get_dataloaders(config)

text = [config.start_sent] + ['neural', 'networks']
while text[-1] != config.end_sent:
    token_ids = config.vocab(text)
    inputs = torch.unsqueeze(torch.tensor(token_ids, dtype=torch.long), dim=0)
    output = torch.squeeze(torch.argmax(model(inputs), dim=1))
    text = text + config.vocab.lookup_tokens([output.tolist()[-1]])

print(' '.join(text[1:-1]))
