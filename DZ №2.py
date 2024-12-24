from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch.nn.modules import Module
from torch.nn import Linear
from torch.optim import Adam
from sklearn.preprocessing import OneHotEncoder

class TitanicDataset(Dataset):
    def __init__(self):
        self.titanic_dataframe = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

        # Заполнение пропусков в столбце 'Age' средним значением
        self.titanic_dataframe['Age'] = self.titanic_dataframe['Age'].fillna(self.titanic_dataframe['Age'].median())

        # Заполнение пропусков в столбце 'Fare'
        self.titanic_dataframe['Fare'] = self.titanic_dataframe['Fare'].fillna(self.titanic_dataframe['Fare'].median())

        # Заполнение пропусков в столбце 'Embarked'
        self.titanic_dataframe['Embarked'] = self.titanic_dataframe['Embarked'].fillna(self.titanic_dataframe['Embarked'].mode()[0])

        # Преобразование Pclass в one-hot encoding
        encoder = OneHotEncoder(sparse=False, drop='first')
        pclass_encoded = encoder.fit_transform(self.titanic_dataframe[['Pclass']])
        pclass_df = pd.DataFrame(pclass_encoded, columns=['Pclass_2', 'Pclass_3'])
        self.titanic_dataframe = pd.concat([self.titanic_dataframe, pclass_df], axis=1)

    def __len__(self):
        return self.titanic_dataframe.shape[0]

    def __getitem__(self, idx):
        row = self.titanic_dataframe.iloc[idx]
        survived = torch.tensor(row['Survived'], dtype=torch.float)
        fare = torch.tensor(row['Fare'])
        age = torch.tensor(row['Age'])
        sex = torch.tensor([0.] if row['Sex'] == 'male' else [1.])
        pclass_2 = torch.tensor(row['Pclass_2'], dtype=torch.float)
        pclass_3 = torch.tensor(row['Pclass_3'], dtype=torch.float)

        x = torch.tensor([fare, age, sex.item(), pclass_2, pclass_3], dtype=torch.float)
        y = survived
        return x, y

titanic_dataset = TitanicDataset()
dataloader = DataLoader(dataset=titanic_dataset, batch_size=32, shuffle=True)

class SurvivalPredictionModel(Module):
    def __init__(self, input_size: int, hidden_size: int, out_size):
        super().__init__()
        self.hidden = Linear(input_size, hidden_size)
        self.hidden2 = Linear(hidden_size, 20)
        self.out = Linear(20, out_size)
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        x = self.hidden(input)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.out(x)
        return torch.sigmoid(x)

# Гиперпараметры
input_size = 5
hidden_size = 30
learning_rate = 0.001
num_epochs = 50

# Модель
model = SurvivalPredictionModel(input_size, hidden_size, 1)

# Функция потерь и оптимизатор
loss_function = torch.nn.BCELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Обучение
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        y_pred = model(x).squeeze()
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')

# Сохранение модели
torch.save(model.state_dict(), 'checkpoint.pt')

# Загрузка и тестирование
model.eval()
model.load_state_dict(torch.load('checkpoint.pt'))

# Пример предсказания
example = torch.tensor([50.0, 30.0, 1.0, 0.0, 1.0])  # Пример входных данных
pred = model(example)
print('Survival probability:', pred.item())