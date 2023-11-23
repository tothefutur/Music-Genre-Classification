import classifier
#import GTZANDataset2
import load_data
import process_data_simple


    
if __name__ == '__main__':
    size_x,size_y = 1,57
    lr,num_epochs,batch_size = 0.02,120,50
    net = classifier.classifier(conv_arch = ((1,64),(1,128),(2,256)),ratio=4,size_x=size_x,size_y=size_y,in_channels=1,layers=3,size_linear=57)
    net.apply(classifier.weight_init)
    
    #data = GTZANDataset2.GTZANDataset(rootDir=r"..//data//music//images_original",resize=(32,448))
    dataset = process_data_simple.GTZANDataset(r"..\data\music\features_3_sec.csv")
    '''train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[int(len(dataset) * 0.7), len(dataset) - int(len(dataset) * 0.7)],
        generator=torch.Generator().manual_seed(0)
    )'''
    train_dataset = dataset(train="True")
    test_dataset = dataset(train="False")
    #调试代码
    '''x,y = train_dataset[1]
    print(x.shape)''' #到此处，尺寸为16*16
    
    #data_train,data_test = load_data.read_data("..//data//music//processed//MusicGen2.pth")
    train_iter,test_iter = classifier.data_iter(train_dataset,test_dataset,batch_size=batch_size)
    classifier.train(net,train_iter,test_iter,num_epochs,classifier.loss(),classifier.optimize(net,lr),'cuda')
    '''for X,y in enumerate(train_iter):
        print(y)'''