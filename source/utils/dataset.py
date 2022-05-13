import os
import  torch
from    torchvision import datasets, transforms


def get_dataset_from_code(code, batch_size):
    """ interface to get function object
    Args:
        code(str): specific data type
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    dataset_root = "./assets/data"
    if code == 'mnist':
        train_loader, test_loader = get_mnist_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'mnist-data'))
    elif code == 'cifar10':
        train_loader, test_loader = get_cifar10_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'cifar10-data'))
    elif code == 'fmnist':
        train_loader, test_loader = get_fasionmnist_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'fasionmnist-data'))
    elif code == 'cifar100':
        train_loader, test_loader = get_cifar100_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'cifar100-data'))
    elif code == 'flash':
        train_loader, test_loader = get_flash_data(batch_size=batch_size,
            data_folder_path='')
    else:
        raise ValueError("Unknown data type : [{}] Impulse Exists".format(data_name))

    return train_loader, test_loader


def get_fasionmnist_data(data_folder_path, batch_size=64):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                   #transforms.Normalize((0.2860,), (0.3530,)),
                                 ])
    # Download and load the training data
    trainset = datasets.FashionMNIST(data_folder_path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.FashionMNIST(data_folder_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def get_mnist_data(data_folder_path, batch_size=64):
    """ mnist data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    train_data = datasets.MNIST(data_folder_path, train=True,  download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    test_data  = datasets.MNIST(data_folder_path, train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_cifar10_data(data_folder_path, batch_size=64):
    """ cifar10 data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(data_folder_path, train=True, 
        download=True, transform=transform_train)
    test_data  = datasets.CIFAR10(data_folder_path, train=False, 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_cifar100_data(data_folder_path, batch_size=64):
    """ cifar100 data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_data = datasets.CIFAR100(data_folder_path, train=True, 
        download=True, transform=transform_train)
    test_data  = datasets.CIFAR100(data_folder_path, train=False, 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_flash_data(data_folder_path, batch_size=64):
    ''' FLASH data
    Args:
        data_folder_path: data path
        batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    '''
    selected_paths = detecting_related_file_paths(data_folder_path,args.experiment_catergories,args.experiment_epiosdes)
    
    # Outputs
    print('******************Getting RF data*************************')
    RF_train, RF_val, RF_test = get_data(selected_paths, 'rf', 'rf', args.test_all, args.test_all_path)
    ytrain, _ = custom_label(RF_train, args.strategy, args.transfer_learning, args.experiment_catergories, args.cat_wise_beams)
    ytest, _ = custom_label(RF_test,args.strategy, args.transfer_learning, args.experiment_catergories, args.cat_wise_beams)
    print('RF data shapes on same client', RF_train.shape, RF_val.shape, RF_test.shape)

    # GPS
    print('******************Getting Gps data*************************')
    X_coord_train, X_coord_validation, X_coord_test = get_data(selected_paths,'gps','gps', args.test_all, args.test_all_path)
    print('GPS data shapes',X_coord_train.shape,X_coord_validation.shape,X_coord_test.shape)
    X_coord_train, X_coord_test = X_coord_train/9747, X_coord_test/9747
    X_coord_train, X_coord_test = X_coord_train[:,:,None], X_coord_test[:,:,None]
    
    # Image
    print('******************Getting image data*************************')
    X_img_train, X_img_validation, X_img_test = get_data(selected_paths,'image','img',args.test_all,args.test_all_path)
    X_img_train, X_img_test = X_img_train/255, X_img_test/255
     
    # Lidar
    print('******************Getting lidar data*************************')
    X_lidar_train, X_lidar_validation, X_lidar_test = get_data(selected_paths,'lidar','lidar',args.test_all,args.test_all_path)
    
            
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}
    
    training_set = FlashDataLoader(X_coord_train, X_img_train, X_lidar_train, ytrain)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    
    testing_set = FlashDataLoader(X_coord_test, X_img_test, X_lidar_test, ytest)
    test_generator = torch.utils.data.DataLoader(testing_set, **params)
    return train_loader, test_loader

class FlashDataLoader(object):
    def __init__(self, ds1, ds2, ds3, label):
        self.ds1 = ds1
        self.ds2 = ds2
        self.ds3 = ds3
        self.label = label

    def __getitem__(self, index):
        x1, x2, x3 = self.ds1[index], self.ds2[index],  self.ds3[index]
        label = self.label[index]
        return torch.from_numpy(x1), torch.from_numpy(x2),  torch.from_numpy(x3), torch.from_numpy(label)

    def __len__(self):
        return self.ds1.shape[0]  # assume both datasets have same length
    
def fetch_flash_data(data_paths,modality,key,test_on_all,path_test_all):   # per cat for now, need to add per epside for FL part
    first = True
    for l in tqdm(data_paths):
        randperm = np.load(l+'/ranperm.npy')
        if first == True:
            open_file = open_npz(l+'/'+modality+'.npz',key)
            train_data = open_file[randperm[:int(0.8*len(randperm))]]
            validation_data = open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]
            test_data = open_file[randperm[int(0.9*len(randperm)):]]
            first = False
        else:
            open_file = open_npz(l+'/'+modality+'.npz',key)
            train_data = np.concatenate((train_data, open_file[randperm[:int(0.8*len(randperm))]]),axis = 0)
            validation_data = np.concatenate((validation_data, open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]),axis = 0)
            test_data = np.concatenate((test_data, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)

    if test_on_all:
        test_data = open_npz(path_test_all+'/'+modality+'_'+'all.npz',key)

    print('tr/val/te',train_data.shape,validation_data.shape,test_data.shape)
    return train_data, validation_data, test_data