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

import SimpleITK as sitk
import numpy as np

def CalculateTumorDSC(npPrediction, npGroundTruth, tumorLabel=1, fullyConnected=True):
    if npPrediction.shape != npGroundTruth.shape:
        raise RuntimeError("Prediction and ground truth mask extents/channels do not match.")

    npMask = (npPrediction == tumorLabel).astype(np.int16)
    npGTMask = (npGroundTruth == tumorLabel).astype(np.int16)

    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(fullyConnected)

    ccGTMask = ccFilter.Execute(sitk.GetImageFromArray(npGTMask))
    gtObjectCount = ccFilter.GetObjectCount()

    ccMask = ccFilter.Execute(sitk.GetImageFromArray(npMask))
    #objectCount = ccFilter.GetObjectCount()

    npGTCCMask = sitk.GetArrayFromImage(ccGTMask) # XXX: This gets returned, make a copy!
    npCCMask = sitk.GetArrayViewFromImage(ccMask)

    tumorScores = []

    for gtCCLabel in range(1, gtObjectCount+1):
        npTmpGTMask = (npGTCCMask == gtCCLabel)

        ccLabels = np.unique(npCCMask[np.logical_and(npTmpGTMask, (npCCMask > 0))])

        npTmpMask = np.zeros(npCCMask.shape, dtype=npCCMask.dtype)
        for ccLabel in ccLabels:
            npTmpMask = np.logical_or(npTmpMask, (npCCMask == ccLabel))

        AcapB = np.logical_and(npTmpMask, npTmpGTMask).sum()
        A = npTmpGTMask.sum()
        B = npTmpMask.sum()

        dice = 2.0*AcapB / (A + B) if AcapB > 0.0 else 0.0

        tumorScores.append((gtCCLabel, dice))

    return np.array(tumorScores), npGTCCMask

if __name__ == "__main__":
    from rcc_common import LoadMask

    prediction = LoadMask("/data/AIR/RCC/NiftiNew3D/Masks/0040-Subject-00013/1210/mask_aligned.nii.gz")
    gtMask = LoadMask("/data/AIR/RCC/NiftiNew3D/Masks/0040-Subject-00013/1210/mask_aligned.nii.gz")

    npPrediction = sitk.GetArrayViewFromImage(prediction)
    npGTMask = sitk.GetArrayViewFromImage(gtMask)

    tumorDSC, _ = CalculateTumorDSC(npPrediction, npGTMask, tumorLabel=3)

    print(tumorDSC)
    print(tumorDSC.mean(axis=0)[1], tumorDSC.std(axis=0)[1])

