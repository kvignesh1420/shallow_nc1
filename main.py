from tqdm import tqdm
import torch
torch.manual_seed(9)
# - setting random seed to 1 leads to zero accuracy for 1 iter (the data is linearly separable
#   from the get go!). However, after 2,3 iters accuracy increases to 0.5
#   and after 4 iters, it becomes 1. This implies that the weight vectors might be rotating?
# - Setting random seed to 9, leads to accuracy 1 even after 1 epoch. However, observe that the data is linearly separable
#   from the get go!
import matplotlib.pyplot as plt
from shallow_collapse.model import DNNModel

def prepare_data():
    # X = torch.randn(size=(N, 1), requires_grad=False)
    X1 = torch.normal(mean=torch.tensor(-1), std=torch.tensor(0.3), size=(N//2, 1), requires_grad=False)
    X2 = torch.normal(mean=torch.tensor(1), std=torch.tensor(0.3), size=(N//2, 1), requires_grad=False)
    X = torch.concat((X1, X2))
    plt.plot(X.cpu().detach().numpy())
    plt.savefig("data.png")
    plt.clf()
    Sigma_W = torch.trace( X.t() @ X ) / N
    class_means = torch.concat((torch.mean(X1, 0, keepdim=True), torch.mean(X2, 0, keepdim=True)))
    global_mean = torch.mean(X, 0, keepdim=True)
    centered_means = class_means - global_mean
    Sigma_B = torch.trace( centered_means.t() @ centered_means ) / 2
    print("Sigma_W(X): {}, Sigma_B(X): {}".format(Sigma_W, Sigma_B))
    print("NC1: {}".format(Sigma_W/Sigma_B))
    Y = torch.concat((-1*torch.ones(N//2, 1), torch.ones(N//2, 1)))
    return X, Y


def compute_accuracy(pred, Y):
    pred = (pred >= 0).type(torch.float)
    # print(pred, Y)
    acc = torch.mean((pred == Y).type(torch.float))
    return acc

def plot_pred(pred):
    plt.plot(pred)
    plt.savefig("pred.png")
    plt.clf()


def probe_features(name):
    def hook(model, inp, out):
        features[name] = out.detach()
    return hook

def probe_non_linear_features(name):
    def hook(model, inp, out):
        non_linear_features[name] = out.detach()
    return hook


def assign_hooks(model):
    for i in range(len(model.hidden_layers)):
        layer_name = i
        model.hidden_layers[i].register_forward_hook(probe_features(name=layer_name))
    for i in range(len(model.activation_layers)):
        layer_name = i
        model.activation_layers[i].register_forward_hook(probe_non_linear_features(name=layer_name))
    return model

@torch.no_grad()
def metrics(data):
    for layer, H in data.items():
        print("Layer: {} Features shape: {}".format(layer, H.shape))
        H = H[perm_inv]
        H1 = H[:N//2, :]
        H2 = H[N//2:, :]
        print("H1 shape: {} H2 shape: {}".format(H1.shape, H2.shape))
        Sigma_W = torch.trace( H.t() @ H ) / N
        class_means = torch.concat((torch.mean(H1, 0, keepdim=True), torch.mean(H2, 0, keepdim=True)))
        print("class means shape: {}".format(class_means.shape))
        global_mean = torch.mean(H, 0, keepdim=True)
        centered_means = class_means - global_mean
        Sigma_B = torch.trace( centered_means.t() @ centered_means ) / 2
        print("Sigma_W(H): {}, Sigma_B(H): {}".format(Sigma_W, Sigma_B))
        print("NC1: {}".format(Sigma_W/Sigma_B))


def train(model, X, Y):
    X = X[perm]
    Y = Y[perm]
    loss_criterion = torch.nn.MSELoss()
    # print("initial loss: {}".format(loss_criterion(X, Y)))
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.01,
        momentum=0,
        weight_decay=5e-4
    )
    num_batches = N//batch_size
    for iter in tqdm(range(1000)):
        for batch in range(num_batches):
            model.zero_grad()
            optimizer.zero_grad()
            x = X[ batch*(batch_size):(batch+1)*batch_size, : ]
            y = Y[ batch*(batch_size):(batch+1)*batch_size, : ]
            pred = model(x)
            loss = loss_criterion(pred, y)
            loss.backward()
            optimizer.step()
        if iter%10 == 0:
            print("iter: {} loss: {}".format(iter, loss.cpu().detach().numpy()))

    pred=model(X)
    acc = compute_accuracy(pred=pred, Y=Y)
    print("accuracy: {}".format(acc.cpu().detach().numpy()))
    plot_pred(pred[perm_inv].detach().cpu().numpy())


if __name__ == "__main__":
    N = 100
    batch_size = 100
    perm = torch.randperm(n=N)
    perm_inv = torch.argsort(perm)
    X, Y = prepare_data()
    features = {}
    non_linear_features = {}
    print(X.shape, Y.shape)
    args = {
        "L": 2,
        "in_features": 1,
        "hidden_features": 1000,
        "out_features": 1,
        "bias": True
    }
    model = DNNModel(args=args)
    model = assign_hooks(model=model)
    print(model)
    train(model=model, X=X, Y=Y)
    print("\nmetrics of features\n")
    metrics(data=features)
    print("\nmetrics of non linear features\n")
    metrics(data=non_linear_features)
