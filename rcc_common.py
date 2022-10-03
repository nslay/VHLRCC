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
import sys
import re
import operator
import functools
import numpy as np
import SimpleITK as sitk
import functools
import operator

_show_warnings = True

def ShowWarnings(value : bool):
    global _show_warnings
    _show_warnings = value

def SaveImage(image, path, compress=True):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.SetUseCompression(compress)

    try:
        writer.Execute(image)
    except:
        return False

    return True

def LoadImage(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)

    try:
        return reader.Execute()
    except:
        return None

def CleanUpHalfMask(mask, kidneyLabel=1, cystLabel=2, tumorLabel=3):
    npMask = sitk.GetArrayFromImage(mask)

    if not (npMask == kidneyLabel).any():
        newMask = sitk.Image(mask.GetSize(), mask.GetPixelID())
        newMask.CopyInformation(mask)
        return newMask

    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    #ccMask = ccFilter.Execute(mask == kidneyLabel)
    ccMask = ccFilter.Execute(mask)

    ccCount = ccFilter.GetObjectCount();
    ccLargestLabel = 0
    ccLargestSize = 0

    npCCMask = sitk.GetArrayViewFromImage(ccMask)

    #print(f"ccCount = {ccCount} ...")

    for ccLabel in range(1,ccCount+1):
        #ccSize = sum(ccMask == ccLabel)
        ccSize = (npCCMask == ccLabel).sum()
        #print(f"{ccLabel}: ccSize = {ccSize}")
        if ccSize > ccLargestSize:
            ccLargestSize = ccSize
            ccLargestLabel = ccLabel

    for ccLabel in range(1,ccCount+1):
        if ccLabel != ccLargestLabel:
            #mask[ccMask == ccLabel] = 0
            npMask[npCCMask == ccLabel] = 0

    """
    # Now find tumor components not connected to kidney
    ccMask = ccFilter.Execute(mask)
    ccCount = ccFilter.GetObjectCount();

    ccWithKidney = set()

    for ccLabel in range(1,ccCount+1):
        if any(mask[ccMask == ccLabel] == kidneyLabel):
            ccWithKidney.add(ccLabel)

    ccTumorMask = ccFilter.Execute(mask == cystLabel or mask == tumorLabel)
    ccTumorCount = ccFilter.GetObjectCount()

    for ccTumorLabel in range(1,ccTumorCount+1):
        ccLabel = max(ccMask[ccTumorMask == ccTumorLabel])
        if ccLabel not in ccWithKidney:
            mask[ccTumorMask == ccTumorLabel] = 0
    """

    newMask = sitk.GetImageFromArray(npMask)
    newMask.CopyInformation(mask)

    return newMask
 

def CleanUpMask(mask, kidneyLabel=1, cystLabel=2, tumorLabel=3):
    halfX = int(mask.GetWidth()/2)

    npMask = sitk.GetArrayFromImage(mask)

    rightMask = sitk.GetImageFromArray(npMask[:,:,:halfX])
    rightMask.SetSpacing(mask.GetSpacing())

    leftMask = sitk.GetImageFromArray(npMask[:,:,halfX:])
    leftMask.SetSpacing(mask.GetSpacing())

    rightMask = CleanUpHalfMask(rightMask, kidneyLabel=kidneyLabel, cystLabel=cystLabel, tumorLabel=tumorLabel)
    leftMask = CleanUpHalfMask(leftMask, kidneyLabel=kidneyLabel, cystLabel=cystLabel, tumorLabel=tumorLabel)

    npRightMask = sitk.GetArrayViewFromImage(rightMask)
    npLeftMask = sitk.GetArrayViewFromImage(leftMask)

    npMask[:,:,:halfX] = npRightMask[:,:,:]
    npMask[:,:,halfX:] = npLeftMask[:,:,:]

    cleanMask = sitk.GetImageFromArray(npMask)
    cleanMask.CopyInformation(mask)

    return cleanMask

def LoadMask(path, numClasses=4, dilateUnknown=False):
    global _show_warnings

    if numClasses != 2 and numClasses != 4:
        return None

    mask = LoadImage(path)

    if mask is None:
        return None

    textFile=os.path.splitext(os.path.basename(path))[0]

    if (textFile.lower().endswith(".nii")):
        textFile = textFile[:-4]


    textFile = os.path.join(os.path.dirname(path), textFile + ".txt")

    # Some text files misspell labels
    kidneyPattern = re.compile(".*kid.*")
    cystPattern = re.compile(".*cy.*")
    unknownPattern = re.compile(".*un.*")
    labelPattern = re.compile("label [0-9]+")

    if os.path.exists(textFile):
        npMask = sitk.GetArrayViewFromImage(mask)
        npNewMask = np.zeros(npMask.shape, dtype=npMask.dtype)

        labelHist=np.histogram(npMask.ravel(), range=[0,npMask.max()], bins=npMask.max()+1)[0]

        with open(textFile, "rt", newline="") as f:
            for line in f:
                line = line.strip().split("#")[0]
                tokens = [ token for token in line.split(" ") if len(token) > 0 ]

                if len(tokens) < 8: # What?
                    continue

                # Lame hack to work around labels with spaces in the name!
                for i in range(8,len(tokens)):
                    tokens[7] += ' '
                    tokens[7] += tokens[i]

                label = int(tokens[0])
                labelName = tokens[7]

                if labelName.startswith('"'):
                    labelName = labelName[1:-1]

                mapLabel = 0

                #if labelPattern.match(labelName.lower()) is not None:
                #    print(labelName)

                #if labelName.lower().endswith("kidney"):
                if kidneyPattern.match(labelName.lower()) is not None:
                    mapLabel = 1
                #elif labelName.lower() == "cyst":
                elif cystPattern.match(labelName.lower()) is not None:
                    mapLabel = 2
                #elif unknownPattern.match(labelName.lower()) is not None or labelName.lower().startswith("rk") or labelName.lower().startswith("lk"):
                elif labelName.lower().startswith("rk") or labelName.lower().startswith("lk"):
                    mapLabel = 3
                elif unknownPattern.match(labelName.lower()) is not None:
                    mapLabel = 255 
                elif labelPattern.match(labelName.lower()) is not None and np.any(npMask == label):
                    #raise RuntimeError(f"Error: Unnamed label '{labelName}' with label value {label}: {path}")
                    print(f"Error: Unnamed label '{labelName}' with label value {label}: {path}")
                    mapLabel = 255
                    
                if label < labelHist.size:
                    labelHist[label] = 0

                npNewMask[npMask == label] = mapLabel

        if np.any(labelHist):
            #raise RuntimeError(f"Error: Undescribed labels found in: {path}")
            print(f"Error: Undescribed labels found in: {path}")
            for label in range(labelHist.size):
                if labelHist[label] > 0:
                    npNewMask[npMask == label] = 255

        if _show_warnings:
            if not np.any(npNewMask == 1):
                print("Warning: No kidneys annotated.", file=sys.stderr)

            if not np.any(npNewMask == 2):
                print("Warning: No cysts annotated.", file=sys.stderr)

            if not np.any(npNewMask == 3):
                print("Warning: No tumors annotated.", file=sys.stderr)

            if np.any(npNewMask == 255):
                print("Warning: Unknown tumors masked out.", file=sys.stderr)

        """
        voxelVolume = functools.reduce(operator.mul, mask.GetSpacing())
        kidneyVolume = (npNewMask == 1).sum()*voxelVolume

        if kidneyVolume < 42000:
            print(f"Warning: Tiny kidney for {path}.", file=sys.stderr)
        """

        if numClasses == 2:
            npNewMask[npNewMask == 1] = 0
            npNewMask[npNewMask == 2] = 0
            npNewMask[npNewMask == 3] = 1

        if dilateUnknown:
            dilateMM = 3.0
            dilateRadius = tuple((int(x) for x in (dilateMM/np.array(mask.GetSpacing()) + 0.5)))
            #print(dilateRadius)

            npIgnoreMask = np.zeros(npNewMask.shape, dtype=npNewMask.dtype)
            npIgnoreMask[npNewMask == 255] = 1

            #dilateRadius = [ int(dilateMM/mask.GetSpacing()[0] + 0.5), int(dilateMM/mask.GetSpacing()[1] + 0.5), int(dilateMM/mask.GetSpacing()[2] + 0.5) ]
            dilateFilter = sitk.BinaryDilateImageFilter()

            dilateFilter.SetKernelType(sitk.sitkBall)
            dilateFilter.SetKernelRadius(dilateRadius)
            dilateFilter.SetBackgroundValue(0)
            dilateFilter.SetForegroundValue(1)

            #print(f"Before {(npIgnoreMask != 0).sum()}")

            ignoreMask = sitk.GetImageFromArray(npIgnoreMask)
            ignoreMask.CopyInformation(mask)

            ignoreMask = dilateFilter.Execute(ignoreMask)
            npIgnoreMask = sitk.GetArrayViewFromImage(ignoreMask)

            #print(f"After {(npIgnoreMask != 0).sum()}")

            npNewMask[npIgnoreMask != 0] = 255


        newMask = sitk.GetImageFromArray(npNewMask)
        newMask.CopyInformation(mask)

        mask = newMask

    return mask

def LoadMaskNoRelabel(path, textFile=None):
    mask = LoadImage(path)

    if mask is None:
        return None

    if textFile is None:
        textFile=os.path.splitext(os.path.basename(path))[0]

        if (textFile.lower().endswith(".nii")):
            textFile = textFile[:-4]

        textFile = os.path.join(os.path.dirname(path), textFile + ".txt")

    labelDict = None

    if os.path.exists(textFile):
        with open(textFile, "rt", newline="") as f:
            labelDict = dict()
        
            for line in f:
                line = line.strip().split("#")[0]
                tokens = [ token for token in line.split(" ") if len(token) > 0 ]

                if len(tokens) < 8: # What?
                    continue

                # Lame hack to work around labels with spaces in the name!
                for i in range(8,len(tokens)):
                    tokens[7] += ' '
                    tokens[7] += tokens[i]

                label = int(tokens[0])
                labelName = tokens[7]

                if labelName.startswith('"'):
                    labelName = labelName[1:-1]

                labelDict[labelName] = label

    return mask, labelDict

def ComputeLabelWeights(dataRoot, listFile, numClasses=4):
    with open(listFile, mode="rt", newline="") as f:
        patientIds = [ line.strip() for line in f if len(line.strip()) > 0 ]

    labelWeights = np.zeros([numClasses], dtype=np.float32)

    for patientId in patientIds:
        maskPath=os.path.join(dataRoot, "Masks", patientId, "mask_aligned.nii.gz")

        mask = LoadMask(maskPath)
        if mask is None:
            print(f"Error: Could not load '{masjPath}'", file=sys.stderr)
            return None

        npMask = sitk.GetArrayViewFromImage(mask)

        for label in range(numClasses):
            labelWeights[label] += (npMask == label).sum()


    if np.any(labelWeights <= 0.0):
        print(f"Error: Empty label count: {labelWeights}.", file=sys.stderr)
        return None

    labelWeights = labelWeights.min() / labelWeights

    print(f"Info: Label weights: {labelWeights}")

    #labelWeights = 1.0 / (4.0 * labelWeights)
    #labelWeights /= labelWeights.max()

    return list(labelWeights)

def ExtractTumorDetections(probMap, mask, labelDict, perc = 0.9):
    smallTumorMM = 5.0
    dilateMM = 2.5
    npProbMap = sitk.GetArrayViewFromImage(probMap)
    npMask = sitk.GetArrayViewFromImage(mask)

    if probMap.GetSize() != mask.GetSize():
        raise RuntimeError("Dimension mismatch between probability map and mask.")

    allScores = []
    allLabels = []

    for labelName in labelDict: 
        if not labelName.lower().startswith("lk") and not labelName.lower().startswith("rk"):
            continue

        label = labelDict[labelName]

        npTumorProbs = npProbMap[npMask == label]

        if npTumorProbs.size == 0:
            continue

        score = np.percentile(npTumorProbs, [perc*100.0])[0]

        allScores.append(score)
        allLabels.append(1)

    # Now dilate tumor/cyst mask

    kidneyPattern = re.compile(".*kid.*")
    unknownPattern = re.compile(".*un.*")
    cystPattern = re.compile(".*cy.*")

    kidneyLabel = 0

    npIgnoreMask = np.ones(npMask.shape, dtype=npMask.dtype)

    for labelName in labelDict:
        if kidneyPattern.match(labelName.lower()) or cystPattern.match(labelName.lower()):
            label = labelDict[labelName]
            npIgnoreMask[npMask == label] = 0

    #dilateRadius = tuple((dilateMM/np.array(mask.GetSpacing()) + 0.5).astype(np.int32))
    """
    dilateRadius = tuple((int(x) for x in (dilateMM/np.array(mask.GetSpacing()) + 0.5)))

    if max(dilateRadius) > 0:
        #dilateRadius = [ int(dilateMM/mask.GetSpacing()[0] + 0.5), int(dilateMM/mask.GetSpacing()[1] + 0.5), int(dilateMM/mask.GetSpacing()[2] + 0.5) ]
        dilateFilter = sitk.BinaryDilateImageFilter()

        dilateFilter.SetKernelRadius(dilateRadius)
        #dilateFilter.SetKernelType(sitk.sitkBall)
        dilateFilter.SetBackgroundValue(0)
        dilateFilter.SetForegroundValue(1)

        ignoreMask = sitk.GetImageFromArray(npIgnoreMask)
        ignoreMask.CopyInformation(mask)

        ignoreMask = dilateFilter.Execute(ignoreMask)
        npIgnoreMask = sitk.GetArrayFromImage(ignoreMask)
    """
    
    # NOTE: In reverse for numpy
    #windowSize = [ max(1, int(smallTumorMM/mask.GetSpacing()[2] + 0.5)), max(1, int(smallTumorMM/mask.GetSpacing()[1] + 0.5)), max(1, int(smallTumorMM/mask.GetSpacing()[0] + 0.5)) ]
    #windowSize = tuple(np.maximum(1.0, smallTumorMM/np.array(mask.GetSpacing()) + 0.5).astype(np.int32))
    #windowSize = tuple(np.maximum(1.0, smallTumorMM/np.array(mask.GetSpacing()) + 0.5).astype(int))
    windowSize = tuple((max(1, int(x)) for x in (smallTumorMM/np.array(mask.GetSpacing()) + 0.5)))

    probWindows = np.lib.stride_tricks.sliding_window_view(npProbMap, windowSize)
    ignoreWindows = np.lib.stride_tricks.sliding_window_view(npIgnoreMask, windowSize)

    probWindows = probWindows[::windowSize[0], ::windowSize[1], ::windowSize[2], ...]
    ignoreWindows = ignoreWindows[::windowSize[0], ::windowSize[1], ::windowSize[2], ...]

    newShape = [ probWindows.shape[0], probWindows.shape[1], probWindows.shape[2], -1 ]

    probWindows = probWindows.reshape(newShape)
    ignoreWindows = ignoreWindows.reshape(newShape)

    probWindows = np.percentile(probWindows, [perc*100.0], axis=3)[0,:,:,:]

    ignoreWindows = ignoreWindows.max(axis=3)
    fpScores = (probWindows[ignoreWindows == 0])

    allScores = np.concatenate((np.array(allScores), fpScores), axis=0)
    allLabels = np.concatenate((np.array(allLabels), np.zeros(fpScores.size)),axis=0)

    return allScores, allLabels

if __name__ == "__main__":
    #path="/data/AIR/RCC/Masks/1065/1065.nii.gz"
    #path="/data/AIR/RCC/Masks/1073/1073.nii.gz"

    #mask = LoadMask(path)

    #print(sitk.GetArrayViewFromImage(mask).max())

    #SaveImage(mask, "blah.nii.gz")

    ComputeLabelWeights("/data/AIR/RCC/NiftiNew", "/data/AIR/RCC/NiftiNew/trainList.txt")

