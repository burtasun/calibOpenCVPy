import cv2 as cv
class Pars:
    class Pattern:
        dims=[11,11]
        type = cv.CALIB_CB_SYMMETRIC_GRID
        distEdges = 3#distancia entre circulos
    pat=Pattern()
    minValidImgs = 3
    pathImgs = './KR6 vision/Calib_27_09_24/*.bmp'
    pathDirSave = './output/'
    class HandEye:
        pathJsonRobotPoses=r'.\handeye\1\handEyePts.dat'
        modeStationaryCam=True#false moving
        blobDetectPathPars='blobDetectorHe.dat'
        maxErrPosMM = 50
        maxErrRotDeg = 2
    handEye = HandEye()
    class Vis:
        previewImg = False
        previewBlobs = False
        previewPatterns = 0#0.25#Scale >0 / <=0 disable
        viewUndistort = False
    vis = Vis()
_pars = Pars()
_previewWin='PreviewWin'