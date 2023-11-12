import classifier
#import GTZANDataset2
import load_data



    
if __name__ == '__main__':
    lr,num_epochs,batch_size = 0.001,50,50
    #net = classifier.classifier(conv_arch = ((1,64),(1,128),(2,256)),ratio=4,size_x=32,size_y=448,in_channels=1,layers=3)
    #net.apply(classifier.weight_init)
    #data = GTZANDataset2.GTZANDataset(rootDir=r"..//data//music//images_original",resize=(32,448))
    data_train,data_test = load_data.read_data("..//data//music//processed//MusicGen2.pth")
    train_iter,test_iter = classifier.data_iter(data_train,data_test,batch_size=batch_size)
    #classifier.train(net,train_iter,test_iter,num_epochs,classifier.loss(),classifier.optimize(net,lr),'cuda')
    for X,y in enumerate(train_iter):
        print(y)