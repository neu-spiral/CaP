"""
@author: debashri
This file contains different neural network models used in non-image-fusion-main.py
CoordNet: CNN model for Coordinate
ImgNet: CNN model for Image
LidarNet: CNN model for LiDAR
FeatureFusion: CNN model to fuse two modalities
InfoFusionThree: MLP model to fuse three modalities (Infocom Implementation)
IncrementalFusionThreeMLP: CNN model to fuse three modalities in incremental fashion
MultiLevelDeepFusion: MLP based model for 2nd level deep fusion between all non-image unimodal and one fusion network
MultiLevelQuadraticFusion: MLP based model for 2nd level quadratic fusion between all non-image unimodal and one fusion network
"""

import numpy as np
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures


# CNN based model for Coordinate modality - Infocom version: 9287640 params included all
class CoordNet(nn.Module):
    def __init__(self, input_dim, output_dim, fusion ='penultimate'):
        super(CoordNet, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 20, kernel_size=2,  padding="same")
        self.conv2 = nn.Conv1d(20, 20, kernel_size=2, padding="same")
        self.pool = nn.MaxPool1d(2, padding=1)

        self.hidden1 = nn.Linear(20, 1024)
        self.hidden2 = nn.Linear(1024, 512)
        self.hidden3 = nn.Linear(512, 256)
        self.hidden4 = nn.Linear(256, 64)
        self.out = nn.Linear(256, output_dim)  # 128
        #######################
        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.fusion = fusion


    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        # print("Shape in coord: ", x.shape)
        # FOR CNN BASED IMPLEMENTATION
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print("shape", x.shape)
        x= self.relu(self.hidden1(x))
        x = self.drop(x)
        x = self.relu(self.hidden2(x))
        x = self.drop(x)
        x = self.relu(self.hidden3(x))
        x = self.drop(x)
        if self.fusion == 'penultimate':
            x = self.tanh(self.hidden4(x))
        else:

            # x = self.softmax(self.out(x))
            x = self.relu(self.out(x)) # no softmax: CrossEntropyLoss()
        return x


# CNN based model for Image modality - Infocom version
class CameraNet(nn.Module):
            def __init__(self, input_dim, output_dim, fusion='penultimate'):
                super(CameraNet, self).__init__()
                dropProb = 0.25
                channel = 32
                self.conv1 = nn.Conv2d(input_dim, channel, kernel_size=(7,7), padding="same")
                self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding="same")
                self.pool1 = nn.MaxPool2d((2,2))
                self.pool2 = nn.MaxPool2d((3, 3), padding=1)

                self.hidden1 = nn.Linear(864, 512)
                self.hidden2 = nn.Linear(512, 256)
                self.hidden3 = nn.Linear(256, 256)
                self.out = nn.Linear(256, output_dim)  # 128
                #######################
                self.drop = nn.Dropout(dropProb)
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                self.softmax = nn.Softmax()
                self.fusion = fusion

            def forward(self, x):

                # FOR CNN BASED IMPLEMENTATION
                # print("test1: ", x.shape)
                x = self.relu(self.conv1(x))
                # print("test2: ", x.shape)
                b = x = self.relu(self.conv2(x))
                # print("test3: ", x.shape)
                x = self.relu(self.conv2(x))
                # print("test4: ", x.shape)
                x = self.relu(self.conv2(x))
                # print("test5: ", x.shape)
                x = torch.add(x, b)
                # print("test6: ", x.shape)
                x = self.pool1(x)
                # print("test7: ", x.shape)
                c = x = self.drop(x)
                # print("test8: ", x.shape)

                x = self.relu(self.conv2(x))
                # print("test9: ", x.shape)
                x = self.relu(self.conv2(x))
                # print("test10: ", x.shape)
                x = torch.add(x, c)
                # print("test11: ", x.shape)
                x = self.pool2(x)
                # print("test12: ", x.shape)
                x = self.drop(x)
                # print("test13: ", x.shape)
                x = x.view(x.size(0), -1)
                # print("shape", x.shape)
                x = self.relu(self.hidden1(x))
                x = self.drop(x)
                x = self.relu(self.hidden2(x))
                x = self.drop(x)

                if self.fusion == 'penultimate':
                    x = self.tanh(self.hidden3(x))
                else:
                    # x = self.softmax(self.out(x))
                    x = self.out(x)  # no softmax: CrossEntropyLoss()
                return x

# CNN based model for LiDAR modality - Infocom version
class LidarNet(nn.Module):
            def __init__(self, input_dim, output_dim, fusion='penultimate'):
                super(LidarNet, self).__init__()
                dropProb1 = 0.3
                dropProb2 = 0.2
                channel = 32
                self.conv1 = nn.Conv2d(input_dim, channel, kernel_size=(3, 3), padding='same')
                self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding='same')
                self.pool1 = nn.MaxPool2d((2,2))
                self.pool2 = nn.MaxPool2d((1, 2))

                self.hidden1 = nn.Linear(320, 1024)
                self.hidden2 = nn.Linear(1024, 512)
                self.hidden3 = nn.Linear(1024, 256)
                self.out = nn.Linear(256, output_dim)  # 128
                #######################
                self.drop1 = nn.Dropout(dropProb1)
                self.drop2 = nn.Dropout(dropProb2)
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                self.softmax = nn.Softmax()
                self.fusion = fusion

            def forward(self, x):
                # FOR CNN BASED IMPLEMENTATION
                # x = F.pad(x, (0, 0, 2, 1))
                a = x = self.relu(self.conv1(x))
                # x = F.pad(x, (0, 0, 2, 1))
                x = self.relu(self.conv2(x))
                # x = F.pad(x, (0, 0, 2, 1))
                x = self.relu(self.conv2(x))
                # print("Shapes: ", x.shape, a.shape)
                x = torch.add(x, a)
                x = self.pool1(x)
                b = x = self.drop1(x)

                x = self.relu(self.conv2(x))
                x = self.relu(self.conv2(x))
                x = torch.add(x, b)
                x = self.pool1(x)
                c = x = self.drop1(x)

                x = self.relu(self.conv2(x))
                x = self.relu(self.conv2(x))
                x = torch.add(x, c)
                x = self.pool2(x)
                d = x = self.drop1(x)

                x = self.relu(self.conv2(x))
                x = self.relu(self.conv2(x))
                x = torch.add(x, d)
                x = x.view(x.size(0), -1)
                # print("shape", x.shape)
                x = self.relu(self.hidden1(x))
                x = self.drop2(x)

                if self.fusion == 'penultimate':
                    x = self.relu(self.hidden2(x))
                    x = self.drop2(x)
                else:
                    x = self.relu(self.hidden3(x))
                    x = self.drop2(x)
                    # x = self.softmax(self.out(x))
                    x = self.out(x)  # no softmax: CrossEntropyLoss()
                return x

# CNN BASED FUSION CLASS - COORD + Image (not used here)
class FeatureFusionCI(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=5, fusion = 'ultimate'):
        super(FeatureFusionCI, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fusion = fusion
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.drop = nn.Dropout(0.25)

        self.conv1 = nn.Conv1d(32, 30, kernel_size=7, padding='same')
        self.conv2 = nn.Conv1d(30, 30, kernel_size=7, padding='same')
        self.conv3 = nn.Conv1d(30, 30, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(2)

        self.classifier = nn.Linear(2*nb_classes, nb_classes) # change

        #WHEN TRAINING FROM SCRATCH
        self.hidden1 = nn.Linear(60, 4*nb_classes)
        self.hidden2 = nn.Linear(4 * nb_classes, 3 * nb_classes)
        self.hidden3 = nn.Linear(3*nb_classes, 2*nb_classes)
        self.out = nn.Linear(2*nb_classes, nb_classes)

    # x1: coord; x2: image;
    def forward(self, x1, x2):
        hidden_layers = []
        x1 = self.modelA(x1)  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x2)
        x2 = x2.view(x2.size(0), -1)

        hidden_layers.append(x1)
        hidden_layers.append(x2)
        x = torch.cat((x1, x2), dim=1)
        print("Shape1: ", x.shape)
        if self.fusion == 'penultimate':
            x = torch.reshape(x, (32, 32, 10)) # (batch_size, 32, 10) ((2, 64) in infocom paper)
            # x = x.permute((2, 1)) # used in infocom paper - commented here (not working)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv3(x))
            x = self.pool1(x)

            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool1(x)

            x = x.view(x.size(0), -1)
            # print("Shape: ", x.shape)

            x = self.relu(self.hidden1(x))
            x = self.drop(x)
            x = self.relu(self.hidden2(x))
            x = self.drop(x)
            x = self.relu(self.hidden3(x))
            x = self.drop(x)
            hidden_layers.append(x)
            x = self.out(x)   # no softmax: CrossEntropyLoss()
            return x, hidden_layers
        # x = self.softmax(self.classifier(x))
        hidden_layers.append(x)
        x = self.classifier(x)  # no softmax: CrossEntropyLoss()
        return x, hidden_layers

# CNN BASED FUSION CLASS - LiDAR + Image
class FeatureFusion(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=5, fusion = 'ultimate'):
        super(FeatureFusion, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fusion = fusion
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.drop = nn.Dropout(0.25)

        self.classifier = nn.Linear(2*nb_classes, nb_classes) # change

        #WHEN TRAINING FROM SCRATCH
        self.hidden1 = nn.Linear(320, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.hidden2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.hidden3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, nb_classes)

    # x1: lidar; x2: image;
    def forward(self, x1, x2):
        hidden_layers = []
        x1 = self.modelA(x1)  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x2)
        x2 = x2.view(x2.size(0), -1)

        hidden_layers.append(x1)
        hidden_layers.append(x2)

        x = torch.cat((x1, x2), dim=1)
        print("Shape1: ", x.shape)
        if self.fusion == 'penultimate':
            x = self.relu(self.hidden1(x))
            x = self.bn1(x)
            x = self.relu(self.hidden2(x))
            x = self.bn2(x)
            x = self.relu(self.hidden3(x))
            x = self.bn3(x)
            hidden_layers.append(x)
            x = self.out(x)   # no softmax: CrossEntropyLoss()
            return x, hidden_layers
        # x = self.softmax(self.classifier(x))
        hidden_layers.append(x)
        x = self.classifier(x)  # no softmax: CrossEntropyLoss()
        return x, hidden_layers


# CNN BASED FUSION CLASS - THREE MODALITIES - Infocom version
class InfoFusionThree(nn.Module):
    def __init__(self, modelA, modelB, modelC, nb_classes=5, fusion = 'penultimate'):
        super(InfoFusionThree, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.fusion = fusion
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


        #self.classifier = nn.Linear(3*nb_classes, nb_classes) # change

        #ORIGINAL ARCHITECTURE (INFOCOM)
        # self.hidden1 = nn.Linear(832, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.hidden2 = nn.Linear(1024, 512)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.hidden3 = nn.Linear(512, 256)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.hidden4 = nn.Linear(256, 128)
        # self.bn4 = nn.BatchNorm1d(128)
        # self.out = nn.Linear(128, nb_classes)

        self.hidden0 = nn.Linear(832, 2048)
        self.bn0 = nn.BatchNorm1d(2048)
        #self.hidden01 = nn.Linear(2048, 1024)
        #self.bn01 = nn.BatchNorm1d(1024)

        self.hidden1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.hidden2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.hidden3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.hidden4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, nb_classes)


    # x1: acoustic; x2: radar; x3: seismic
    def forward(self, x1, x2, x3):
        x1 = self.modelA(x1)  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        
        x2 = self.modelB(x2)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.modelC(x3)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)

        if self.fusion == 'penultimate':
            # x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
            x = x.view(x.size(0), -1)
            # print("Shape: ", x.shape)

            x = self.relu(self.hidden0(x))
            x = self.bn0(x)
            # x = self.relu(self.hidden01(x))
            # x = self.bn01(x)

            x = self.relu(self.hidden1(x))
            x = self.bn1(x)
            x = self.relu(self.hidden2(x))
            x = self.bn2(x)
            x = self.relu(self.hidden3(x))
            x = self.bn3(x)
            x = self.relu(self.hidden4(x))
            x = self.bn4(x)
            x = self.out(x)  # no softmax: CrossEntropyLoss()
            
            return x
        x = self.classifier(x) # no softmax: CrossEntropyLoss()
        return x


# CNN BASED INCREMENTAL FUSION CLASS - THREE MODALITIES
class IncrementalFusionThree(nn.Module):
    def __init__(self, tempModel, modelC, nb_classes=5, fusion = 'ultimate'):
        super(IncrementalFusionThree, self).__init__()
        self.tempModel = tempModel
        self.modelC = modelC
        self.fusion = fusion
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(0.5)

        self.hidden1 = nn.Linear(128, 2048) # after conv layers: 256, only FC layers: 128
        self.bn1 = nn.BatchNorm1d(2048)
        self.hidden2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.hidden3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.hidden4 = nn.Linear(256, nb_classes)

        self.classifier = nn.Linear(2*nb_classes, nb_classes)

        self.conv1 = nn.Conv1d(8, 256, kernel_size=4, stride=2)  # 256 #32 # padding # 7
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1)  # 256 #32 # 5
        self.pool1 = nn.MaxPool1d(5)  # 2
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2)  # 256 #32 # padding # 7

    # x1: acoustic; x2: radar; x3: seismic
    def forward(self, x1, x2, x3):

        x1 = self.modelC(x1) # Coordinate
        x1 = x1.view(x1.size(0), -1)

        x3 = self.tempModel(x3, x2)
        x3 = x3.view(x3.size(0), -1)

        x = torch.cat((x3, x1), dim=1)


        if self.fusion == 'penultimate':
            # print("Test1: ", x.shape)
            # x = torch.reshape(x, (x.shape[0], 8, 16))
            # print("Test2: ", x.shape)
            # x = self.relu(self.conv1(x))
            #
            # x = self.relu(self.conv2(x))
            #
            # x = self.relu(self.conv3(x))
            # x = self.relu(self.conv2(x))
            #
            # x = x.view(x.size(0), -1)
            print("Test3: ", x.shape)
            x = self.relu(self.hidden1(x))
            x = self.bn1(x)
            x = self.relu(self.hidden2(x))
            x = self.bn2(x)
            x = self.relu(self.hidden3(x))
            x = self.bn3(x)
            x = self.relu(self.hidden4(x))
            return x

        x = self.classifier(F.sigmoid(x)) # use relu
        return x

# MULTI LEVEL DEEP FUSION CLASS - WITH MLP LAYERS
class MultiLevelDeepFusion(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelAll, nb_classes=5):
        super(MultiLevelDeepFusion, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelAll = modelAll
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.2)

        self.classifier = nn.Linear(4*nb_classes, nb_classes)


        #### MLP layers ################
        self.hidden1 = nn.Linear(4*nb_classes, 4*nb_classes) #20
        self.hidden2 = nn.Linear(4*nb_classes, 3*nb_classes)
        self.hidden3 = nn.Linear(3*nb_classes, 2*nb_classes)
        self.hidden4 = nn.Linear(2*nb_classes, nb_classes)
        ### End of MLP layers ###########


    # x1: acoustic; x2: radar; x3: seismic
    def forward(self, x1, x2, x3):
        ## EACH MODELS
        out_x1 = self.modelA(x1)
        out_x2 = self.modelB(x2)
        out_x3 = self.modelC(x3)
        out_all = self.modelAll(x1, x2, x3)

        # # TRANSFORMING TO SIGMOID- acoustic and seismic
        # out_x1 = self.sigmoid(out_x1)
        # out_x3 = self.sigmoid(out_x3)

        x = torch.cat((out_x1, out_x2, out_x3, out_all), dim=1)

        print("Test1: ", x.shape)
        ## used for deep fusion ####
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)

        # uncomment this when you want to use a single layer
        # x = self.classifier(x)
        return x


# MULTI LEVEL QUADRATIC FUSION CLASS - WITH MLP LAYERS
class MultiLevelQuadraticFusion(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelAll, nb_classes=5): # sequence: acoustic, radar, seismic, aggregated_fusion
        super(MultiLevelQuadraticFusion, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelAll = modelAll
        self.sigmoid = nn.Sigmoid()
        self.poly = PolynomialFeatures(2, interaction_only=False)  # using only both quadratic and interaction features
        self.drop = nn.Dropout(0.3)

        self.classifier = nn.Linear(100, nb_classes)

        #THE MLP BASED FUSION MODEL - WHILE TRAINING FROM SCRATCH
        self.hidden1 = nn.Linear(231, 100)
        self.hidden2 = nn.Linear(100, 50)
        self.hidden3 = nn.Linear(50, 30)
        self.hidden4 = nn.Linear(30, 15)
        self.hidden5 = nn.Linear(15, nb_classes)

    def quadratic(self, input):
        #print("shape of input in quadratic: ", input.shape)
        output = input.repeat(1, input.shape[1])
        for i in range(input.shape[1]):
            output[:, i*input.shape[1]:(i+1)*input.shape[1]] = input[:, :input.shape[1]] * torch.reshape(input[:, i], (-1,1))
        return output

    # x1: acoustic; x2: radar; x3: seismic
    def forward(self, x1, x2, x3):

        ## EACH MODELS
        out_x1 = self.modelA(x1)
        out_x2 = self.modelB(x2)
        out_x3 = self.modelC(x3)
        out_all = self.modelAll(x1, x2, x3)

        # TRANSFORMING TO SIGMOID
        # out_x1 = self.sigmoid(out_x1)
        # out_x3 = self.sigmoid(out_x3)

        ## QUADRATIC FEATURES (YUANYUAN)
        x1 = self.quadratic(out_x1)
        x2 = self.quadratic(out_x2)
        x3 = self.quadratic(out_x3)
        x4 = self.quadratic(out_all)
        ## QUADRATIC FEATURES END

        x = torch.cat((x1, x2, x3, x4), dim=1)

       ## QUADRATIC FEATURES (DEBASHRI)
        # x = x.cpu().detach().numpy()
        # x = self.poly.fit_transform(x)
        # x = torch.from_numpy(x).to(torch.device("cuda:0"))
        ## QUADRATIC FEATURES END

        print("Shape1: ", x.shape)

        x = self.classifier(x)

        return x