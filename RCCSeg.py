# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# October 2022
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import os
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.ops as ops
import SimpleITK as sitk
from ImageBatcher import ImageBatcher
from roc import ComputeROC
from rcc_common import LoadImage, LoadMask, LoadMaskNoRelabel, SaveImage, ComputeLabelWeights, ShowWarnings, ExtractTumorDetections, CleanUpMask
from UNetLeaky import UNet
#from UNet import UNet
from Deterministic import NotDeterministic

class RCCSeg:
    def __init__(self, numClasses=4):
        self.device = "cpu"
        self.multipleOf = 16
        self.numClasses=numClasses
        self.net = UNet(in_channels=4,out_channels=self.numClasses)
        self.dataRoot = None
        self.saveSteps = 10
        self.valSteps = 5*self.saveSteps
        #self.valSteps = 1
        self.dilateUnknown = False

    def SetDevice(self, device):
        self.device = device
        self.net = self.net.to(device)

    def GetDevice(self):
        return self.device

    def SetDataRoot(self, dataRoot):
        self.dataRoot = dataRoot

    def GetDataRoot(self):
        return self.dataRoot

    def SaveModel(self, fileName):
        torch.save(self.net.state_dict(), fileName)

    def LoadModel(self, fileName):
        self.net.load_state_dict(torch.load(fileName, map_location=self.GetDevice()))

    def RunOne(self,patientId):
        sitkVolumes = [None]*4
        npVolumes = [None]*len(sitkVolumes)

        for i in range(len(npVolumes)):
            maskFileName = "normalized_aligned.nii.gz" if i == 0 else f"normalized{i+1}_aligned.nii.gz"
            volumePath = os.path.join(self.dataRoot, "Images", patientId, maskFileName)

            sitkVolumes[i] = LoadImage(volumePath)

            if sitkVolumes[i] is None:
                return None

            npVolumes[i] = sitk.GetArrayViewFromImage(sitkVolumes[i])

            if npVolumes[0].shape != npVolumes[i].shape:
                raise RuntimeError("Error: Dimension mismatch between volumes ({npVolumes[0].shape} != {i}: {npVolumes[i].shape}).")


        halfXDim = int(npVolumes[0].shape[2]/2)

        npVolumesRight = [None]*len(npVolumes)
        npVolumesLeft = [None]*len(npVolumes)

        for i in range(len(npVolumes)):
            npVolumesRight[i] = npVolumes[i][:,:,:halfXDim]
            npVolumesLeft[i] = npVolumes[i][:,:,halfXDim:]

            rightRemainderY = (npVolumesRight[i].shape[1] % self.multipleOf)
            rightRemainderX = (npVolumesRight[i].shape[2] % self.multipleOf)

            leftRemainderY = (npVolumesLeft[i].shape[1] % self.multipleOf)
            leftRemainderX = (npVolumesLeft[i].shape[2] % self.multipleOf)

            if rightRemainderY != 0:
                begin = int(rightRemainderY/2)
                end = begin + npVolumesRight[i].shape[1] - rightRemainderY
                npVolumesRight[i] = npVolumesRight[i][:, begin:end, :]

            if rightRemainderX != 0:
                begin = int(rightRemainderX/2)
                end = begin + npVolumesRight[i].shape[2] - rightRemainderX
                npVolumesRight[i] = npVolumesRight[i][:, :, begin:end]

            if leftRemainderY != 0:
                begin = int(leftRemainderY/2)
                end = begin + npVolumesLeft[i].shape[1] - leftRemainderY
                npVolumesLeft[i] = npVolumesLeft[i][:, begin:end, :]

            if leftRemainderX != 0:
                begin = int(leftRemainderX/2)
                end = begin + npVolumesLeft[i].shape[2] - leftRemainderX
                npVolumesLeft[i] = npVolumesLeft[i][:, :, begin:end]


        batchLeft = np.zeros([npVolumesLeft[0].shape[0], len(npVolumes), npVolumesLeft[0].shape[1], npVolumesLeft[0].shape[2]], dtype=np.float32)
        batchRight = np.zeros([npVolumesRight[0].shape[0], len(npVolumes), npVolumesRight[0].shape[1], npVolumesRight[0].shape[2]], dtype=np.float32)

        for i in range(len(npVolumes)):
            batchLeft[:,i,:,:] = npVolumesLeft[i][:,:,:]
            batchRight[:,i,:,:] = npVolumesRight[i][:,:,:]

        batchLeft = torch.from_numpy(batchLeft).to(self.GetDevice())
        batchRight = torch.from_numpy(batchRight).to(self.GetDevice())

        softmax = nn.Softmax(dim=1).to(self.GetDevice())

        with torch.no_grad():
            self.net.eval()

            probLeft = softmax(self.net(batchLeft))
            probRight = softmax(self.net(batchRight))

            self.net.train()

            #print(probLeft.shape)
            #print(probRight.shape)

            prob = torch.zeros([probLeft.shape[0], probLeft.shape[1], npVolumes[0].shape[1], npVolumes[0].shape[2]])

            #print(prob.shape)

            rightBeginX = int(rightRemainderX/2)
            rightEndX = rightBeginX + probRight.shape[3]

            leftBeginX = halfXDim + int(leftRemainderX/2)
            leftEndX = leftBeginX + probLeft.shape[3]

            #print(f"{leftEndX-leftBeginX}, {probLeft.shape[3]}")

            rightBeginY = int(rightRemainderY/2)
            rightEndY = rightBeginY + probRight.shape[2]

            leftBeginY = rightBeginY
            leftEndY = rightEndY

            #print(f"{leftBeginY}:{leftEndY} , {leftBeginX}:{leftEndX}")
            #print(f"{rightBeginY}:{rightEndY} , {rightBeginX}:{rightEndX}")

            prob[:,:,rightBeginY:rightEndY,rightBeginX:rightEndX] = probRight[:,:,:,:]
            prob[:,:,leftBeginY:leftEndY,leftBeginX:leftEndX] = probLeft[:,:,:,:]

            prob = prob.transpose(1,3).transpose(1,2)

            labelMap = prob.argmax(dim=3).type(torch.int16)

            sitkProb = sitk.GetImageFromArray(prob.numpy())
            sitkLabelMap = sitk.GetImageFromArray(labelMap.numpy())

            sitkProb.SetSpacing(sitkVolumes[0].GetSpacing())
            sitkProb.SetDirection(sitkVolumes[0].GetDirection())
            sitkProb.SetOrigin(sitkVolumes[0].GetOrigin())

            sitkLabelMap.SetSpacing(sitkVolumes[0].GetSpacing())
            sitkLabelMap.SetDirection(sitkVolumes[0].GetDirection())
            sitkLabelMap.SetOrigin(sitkVolumes[0].GetOrigin())

        return sitkProb, sitkLabelMap

    def Test(self,valList):
        if isinstance(valList, str):
            with open(valList, "rt", newline='') as f:
                patientIds = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]
        else:
            patientIds = valList

        allScores = None
        allLabels = None

        allDices = dict()
        npDices = np.zeros([len(patientIds), self.numClasses])

        i = 0
        for patientId in patientIds:
            print(f"Info: Running '{patientId}' ...")

            maskFile = os.path.join(self.GetDataRoot(), "Masks", patientId, "mask_aligned.nii.gz")

            # First DICE!
            gtMask = LoadMask(maskFile, self.numClasses, dilateUnknown=False)
            npGtMask = sitk.GetArrayFromImage(gtMask)

            probMap, labelMap = self.RunOne(patientId)

            npProbMap = sitk.GetArrayFromImage(probMap)
            npProbMap = npProbMap[:,:,:,-1]

            if self.numClasses == 4:
                labelMap = CleanUpMask(labelMap)
                npLabelMap = sitk.GetArrayViewFromImage(labelMap) # XXX: Very slow without making it numpy
                npProbMap[npLabelMap == 0] = 0

            npLabelMap = sitk.GetArrayFromImage(labelMap)

            npLabelMap[npGtMask == 255] = 0
            npGtMask[npGtMask == 255] = 0

            halfX = int(npGtMask.shape[2]/2)

            npRightGtMask = npGtMask[:,:,:halfX]
            npLeftGtMask = npGtMask[:,:,halfX:]

            npRightLabelMap = npLabelMap[:,:,:halfX]
            npLeftLabelMap = npLabelMap[:,:,halfX:]

            for label in range(1,self.numClasses):
                AintB, A, B = 0.0, 0.0, 0.0

                if npRightGtMask.max() > 0:
                    AintB += np.sum(np.logical_and((npRightGtMask == label), (npRightLabelMap == label)))
                    A += np.sum(npRightGtMask == label)
                    B += np.sum(npRightLabelMap == label)

                if npLeftGtMask.max() > 0:
                    AintB += np.sum(np.logical_and((npLeftGtMask == label), (npLeftLabelMap == label)))
                    A += np.sum(npLeftGtMask == label)
                    B += np.sum(npLeftLabelMap == label)

                dice = 1.0 if A+B <= 0.0 else 2.0 * AintB / ( A + B )

                #if AintB == 0.0:
                #    print(f"A = {A}, B = {B}")

                if patientId not in allDices:
                    allDices[patientId] = [ -1 ]*self.numClasses

                allDices[patientId][label] = dice
                npDices[i, label] = dice

                print(f"{label}: dice = {dice}")

            i += 1

            # Now AUC
            gtMask, labelDict = LoadMaskNoRelabel(maskFile)
            npGtMask = sitk.GetArrayFromImage(gtMask)

            npGtMask[npGtMask == 255] = 0

            npRightGtMask = npGtMask[:,:,:halfX]
            npLeftGtMask = npGtMask[:,:,halfX:]

            npRightProbMap = npProbMap[:,:,:halfX]
            npLeftProbMap = npProbMap[:,:,halfX:]

            if npRightGtMask.max() > 0:
                rightProbMap = sitk.GetImageFromArray(npRightProbMap)
                rightProbMap.SetSpacing(probMap.GetSpacing())

                rightGtMask = sitk.GetImageFromArray(npRightGtMask)
                rightGtMask.SetSpacing(probMap.GetSpacing())

                scores, labels = ExtractTumorDetections(rightProbMap, rightGtMask, labelDict)

                if allScores is None:
                    allScores = scores
                    allLabels = labels
                else:
                    allScores = np.concatenate((allScores, scores), axis=0)
                    allLabels = np.concatenate((allLabels, labels), axis=0)

            if npLeftGtMask.max() > 0:
                leftProbMap = sitk.GetImageFromArray(npLeftProbMap)
                leftProbMap.SetSpacing(probMap.GetSpacing())

                leftGtMask = sitk.GetImageFromArray(npLeftGtMask)
                leftGtMask.SetSpacing(probMap.GetSpacing())

                scores, labels = ExtractTumorDetections(leftProbMap, leftGtMask, labelDict)

                if allScores is None:
                    allScores = scores
                    allLabels = labels
                else:
                    allScores = np.concatenate((allScores, scores), axis=0)
                    allLabels = np.concatenate((allLabels, labels), axis=0)

        avgDice = [-1]*self.numClasses
        stdDice = [-1]*self.numClasses
        medDice = [-1]*self.numClasses

        for label in range(1,self.numClasses):
            npMask = (npDices[:,label] >= 0)

            if not npMask.any():
                continue

            avgDice[label] = npDices[npMask, label].mean()
            stdDice[label] = npDices[npMask, label].std()
            medDice[label] = np.median(npDices[npMask, label])

        roc = ComputeROC(torch.from_numpy(allScores), torch.from_numpy(allLabels))

        return (avgDice, stdDice, medDice), allDices, roc

    def Train(self,trainList,valPerc=0.0,snapshotRoot="snapshots", seed=6112):
        batchSize=16
        labelWeights = torch.Tensor([1.0]*self.numClasses)
        historyLength = 30
        learningRate = 1e-3
        ShowWarnings(False)
        numEpochs=1000

        print(f"Info: numClasses = {self.numClasses}, dilateUnknown = {self.dilateUnknown}")

        with open(trainList, mode="rt", newline="") as f:
            patientIds = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]

        valList = None
        if valPerc > 0.0:
            mid = max(1, int(valPerc*len(patientIds)))
            valList = patientIds[:mid]
            trainList = patientIds[mid:]
        else:
            trainList = patientIds

        imageBatcher = ImageBatcher(self.GetDataRoot(), trainList, batchSize, numClasses=self.numClasses, seed=seed, dilateUnknown=self.dilateUnknown)

        imageBatcher.start()

        criterion = nn.CrossEntropyLoss(ignore_index=-1,weight = labelWeights).to(self.GetDevice())

        optimizer = optim.Adam(self.net.parameters(), lr = learningRate)

        trainLosses = np.ones([numEpochs])*1000.0
        valAUCs = np.zeros([numEpochs])

        if not os.path.exists(snapshotRoot):
            os.makedirs(snapshotRoot)

        lossHistory = []
        prevMeanLoss = 100000.0

        for e in range(numEpochs):

            currentLoss = 0.0
            count = 0

            for xbatch, ybatch, digest in imageBatcher:
                #print(xbatch.shape)

                h = hashlib.md5()
                h.update(xbatch.numpy().tobytes())
                h.update(ybatch.numpy().tobytes())

                assert h.hexdigest() == digest

                xbatch = xbatch.to(self.GetDevice())
                ybatch = ybatch.to(self.GetDevice())

                optimizer.zero_grad()            

                outputs = self.net(xbatch)

                with NotDeterministic():
                    loss = criterion(outputs, ybatch)

                    loss.backward()

                optimizer.step()

                currentLoss += loss.item()
                count += 1

            if count > 0:
                currentLoss /= count

            lossHistory.append(currentLoss)
            meanLoss = np.mean(lossHistory[-historyLength:])

            if len(lossHistory) >= historyLength:
                lossHistory.pop(0)

                if False and learningRate > 1e-6 and meanLoss >= (1-0.995)*prevMeanLoss:
                    print("Info: Updating learning rate.")

                    learningRate /= 5.0

                    lossHistory.clear()
                    for g in optimizer.param_groups:
                        g["lr"] = learningRate

            prevMeanLoss = meanLoss

            snapshotFile=os.path.join(snapshotRoot, f"epoch_{e}.pt")
            rocFile=os.path.join(snapshotRoot, f"validation_roc_{e}.txt")
            diceFile=os.path.join(snapshotRoot, f"dice_stats_{e}.txt")

            if ((e+1) % self.saveSteps) == 0:
                print(f"Info: Saving {snapshotFile} ...", flush=True)
                self.SaveModel(snapshotFile)
            else:
                print(f"Info: Skipping saving {snapshotFile}.", flush=True)

            # For debugging
            #self.LoadModel(snapshotFile)

            trainLosses[e] = currentLoss

            print(f"Info: Epoch = {e}, training loss = {currentLoss}, mean loss = {meanLoss}, learning rate = {learningRate}", flush=True)

            if valList is None:
                print(f"Info: Epoch = {e}, training loss = {currentLoss}, mean loss = {meanLoss}, learning rate = {learningRate}", flush=True)
            elif ((e+1) % self.valSteps) == 0: 
                diceStats, allDices, roc = self.Test(valList)
                print(f"Info: Epoch = {e}, training loss = {currentLoss}, mean loss = {meanLoss}, learning rate = {learningRate}, validation AUC = {roc[3]}, validation dices = {diceStats[0]} +/- {diceStats[1]}", flush=True)

                valAUCs[e] = roc[3]

                with open(rocFile, mode="wt", newline="") as f:
                    f.write("# Threshold\tFPR\tTPR\n")

                    for threshold, fpr, tpr in zip(roc[0], roc[1], roc[2]):
                        f.write(f"{threshold}\t{fpr}\t{tpr}\n")

                    f.write(f"# AUC = {roc[3]}\n")

                with open(diceFile, mode="wt", newline="") as f:
                    for patientId in allDices:
                        f.write(f"{patientId}: {allDices[patientId]}\n")

                    f.write(f"\nDice stats: {diceStats[0]} +/- {diceStats[1]}\n")
 
        imageBatcher.stop()

        return trainLosses, valAUCs

