## a simple GPU LR

class DatasetLoader_e2e(Dataset):
    
    def __init__(self, features, targets, transform=None):
        self.features = features        
        self.targets = targets
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        instance = torch.from_numpy(self.features[index, :].todense())
        if self.targets is not None:
            target = torch.as_tensor(self.targets[index].todense())
        else:
            target = None
        return (instance,target)

class ffNN(nn.Module):
    def __init__(self, input_size,num_classes):
        super(ffNN, self).__init__()
        self.basic = nn.Linear(input_size, num_classes)
        self.first = nn.Linear(num_classes*2, num_classes)
        self.sigma = nn.Sigmoid()        
    def forward(self, x):
        out = self.basic(x)
        out = self.sigma(out)
        return out
    
def to_one_hot(lbx):
    enc = OneHotEncoder(handle_unknown='ignore')
    return enc.fit_transform(lbx.reshape(-1,1))
        
class e2e_DNN:

    def __init__(self,batch_size=1,num_epochs=25,learning_rate=0.0001,stopping_crit=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.BCELoss(reduction='mean')
        self.stopping_crit = stopping_crit
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def fit(self,features,labels):

        labels = to_one_hot(labels)
        train_dataset = DatasetLoader_e2e(features, labels)
        total_step = len(train_dataset)
        stopping_iteration = 0
        loss = 1
        current_loss = 0
        self.model = ffNN(features.shape[1],labels.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        for epoch in range(self.num_epochs):
            if current_loss != loss:
                current_loss = loss
            else:
                stopping_iteration+=1
            if stopping_iteration > self.stopping_crit:
                break
            losses_per_batch = []
            for i, (features,labels) in enumerate(train_dataset):
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)
                outputs = self.model.forward(features)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_per_batch.append(float(loss))
            mean_loss = np.mean(losses_per_batch)
          #  logging.info("Current loss {}, epoch {}".format(mean_loss,epoch))
            
    def predict(self,features):
        test_dataset = DatasetLoader_e2e(features, None)
        predictions = []
        with torch.no_grad():
            for features,_ in test_dataset:
                features = features.float().to(self.device)
                representation = self.model.forward(features)
                values,indices = torch.max(representation,1)
                indices = indices.cpu().numpy()+1
                for el in indices:
                    predictions.append(el)
        return predictions
