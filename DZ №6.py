import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Датасет
sciq_dataset = load_dataset("allenai/sciq")

class SciQDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            question = item["question"]
            correct_answer = item["correct_answer"]
        except KeyError as e:
            logging.warning(f"Missing key in data: {e}")
            return None

        prompt = f"Question: {question}\nAnswer: {correct_answer}\nDistractors:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        inputs["labels"] = inputs.input_ids.clone()
        return {key: val.squeeze(0) for key, val in inputs.items()}

train_data = sciq_dataset["train"]
dataset = SciQDataset(train_data, tokenizer)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# отвлекающие ответы
def generate_distractors(question, correct_answer, num_distractors=3):
    model.eval()
    prompt = f"Question: {question}\nAnswer: {correct_answer}\nDistractors:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=50,
            num_return_sequences=num_distractors,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    distractors = []
    for output in outputs:
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        distractor = decoded_text.split("Distractors:")[-1].strip().split("\n")[0]
        distractors.append(distractor if distractor else "No distractor generated")
    return list(set(distractors))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

EPOCHS = 3
logging.info(f"Starting training for {EPOCHS} epochs")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in data_loader:
        if batch is None:
            continue
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    logging.info(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

model_path = "gpt2_distractors.pth"
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}")

# Генерируем текст
test_question = "What is the powerhouse of the cell?"
test_answer = "Mitochondria"
distractors = generate_distractors(test_question, test_answer)
logging.info("Generated Distractors:")
for idx, distractor in enumerate(distractors):
    logging.info(f"Distractor {idx + 1}: {distractor}")