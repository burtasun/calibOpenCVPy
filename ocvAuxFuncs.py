import numpy as np
import cv2 as cv
import math
from Pars import _pars

def createPtsPatternSymPattern():
    [dx,dy] = _pars.pat.dims
    objp = np.zeros((dx*dy,3), np.float32)
    objp[:,:2] = np.mgrid[0:dx,0:dy].T.reshape(-1,2)
    objp*=_pars.pat.distEdges
    return objp

def createPtsPattern():
    if _pars.pat.type == cv.CALIB_CB_SYMMETRIC_GRID:
        return createPtsPatternSymPattern()
    else:
        raise f'{_pars.pat.type} sin implementar!'



#previewImg
def previewImg(img:cv.Mat, msg = '', waitMs = 0):
    if msg!='':
        [textSize, baseLine] = cv.getTextSize(msg, 1, 1, 1)
        textOrigin = (img.shape[1] - 2 * textSize[0] - 10, img.shape[0] - 2 * baseLine - textSize[1])
        cv.putText(img, msg, textOrigin, 1, 1, (0, 255, 0))
    cv.imshow("previewWin", img)
    if (waitMs>=0):
        cv.waitKey(waitMs)
#previewImg


#obtener puntos imagen
def getPtsImg(
        # calibPars,
        img:cv.Mat, 
        blobDetector = cv.SimpleBlobDetector.create(),

) -> tuple[bool, cv.typing.MatLike]:
    #deteccion ptos
    if _pars.vis.previewBlobs:
        kp = blobDetector.detect(img)
        imgShow = cv.drawKeypoints(img, kp, None, [0,0,255])
        previewImg(imgShow, "previewBlobDetectorKPs, nKps " + str(len(kp)) + '\n')

    patDims = _pars.pat.dims
    #mas robusto frente a distorsion radial
    flags = _pars.pat.type | cv.CALIB_CB_CLUSTERING
    # circleGridFinderPars = cv.CirclesGridFinderParameters()
    # circleGridFinderPars.maxRectifiedDistance=2
    # circleGridFinderPars.squareSize=2.0
    [found,pts] = cv.findCirclesGrid(\
        img, patDims, flags, blobDetector=blobDetector)#, CirclesGridFinderParameters = circleGridFinderPars)
    if found:
        if _pars.vis.previewPatterns:
            imgShow = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            ptsMat = np.copy(pts)
            cv.drawChessboardCorners(imgShow, patDims, ptsMat, True)
            previewImg(imgShow, "patternImg", 500)
    return [found,pts]
#getPtsImg



#Metrica analoga a Matlab
#   suma cuadratica distancias respecto a final
def computeReprojectionErrors(
        objPts:list[np.array],
	    imgPts:list[np.array],
	    rvecs:list[np.array],
        tvecs:list[np.array],
	    cameraMatrix:np.array,
        distCoeffs:np.array
) -> tuple[float,list[float]] :
    perViewErrors = []
    totalErr=0.0
    totalPoints=0
    for i in range(len(objPts)):
        [imagePointsRepro,_] = cv.projectPoints(objPts[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        # print(f'{imgPts[i]} \t\t {imagePointsRepro}')
        errsPts = np.linalg.norm(imgPts[i]-imagePointsRepro,2,2)
        errImg = np.linalg.norm(errsPts,2,0)[0]
        n = objPts[i].shape[0]
        perViewErrors.append(math.sqrt(errImg*errImg / n))
        totalErr += errImg*errImg
        totalPoints += n
    meanErrs = math.sqrt(totalErr/totalPoints)
    return (meanErrs,perViewErrors)



# TODO integracion parcial, 
def runCalibration(
    objtPts:np.array,
    imgPts:np.array,
    imWidth,imHeight,
#    const cv::Size& patternSize, const double dimsSquarePattern,
#    cv::Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
#    vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
#    vector<float>& reprojErrs, double& totalAvgErr, vector<Point3f>& newObjPoints, bool guessUse)
) -> tuple[bool, np.array, np.array]: #valid / cameraMat(intrinsic) / distCoeffs
    #Parametros intrinsecos
    rms = 0.0

    iFixedPoint = -1; # patternSize.width - 1;

    flag = 0; #TODO integrar
    # if (guessUse):
    #     flag |= CALIB_USE_INTRINSIC_GUESS;//////////////
    # flag |= cv.CALIB_FIX_PRINCIPAL_POINT
    flag |= cv.CALIB_ZERO_TANGENT_DIST #deshabilitar calib dist tang
    # flag |= cv.CALIB_FIX_ASPECT_RATIO
    # flag |= cv.CALIB_FIX_K1
    # flag |= cv.CALIB_FIX_K2
    flag |= cv.CALIB_FIX_K3
    flag |= cv.CALIB_FIX_K4
    flag |= cv.CALIB_FIX_K5
    flag |= cv.CALIB_FIX_K6
    
    print("regresion parametros camara\n")
    
    termCriteria =  (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, np.finfo(float).eps)
    [totAverErrOpenCV, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints] = \
        cv.calibrateCameraRO(objtPts, imgPts, [imWidth,imHeight], iFixedPoint,
        None, None, None, None, None, flag,
        termCriteria)
    
    if totAverErrOpenCV < 0:
         print(f'No se ha encontrado una solucion!')
         return (False,cameraMatrix,distCoeffs)

    # consistencia numerica !nan o inf
    ok = cv.checkRange(cameraMatrix) and cv.checkRange(distCoeffs)
    if ok == False:
         print(f'Parametros intrinsecos no validos!')
         return (False,cameraMatrix,distCoeffs)
    
    [mearErr,perViewErrs]=computeReprojectionErrors(objtPts,imgPts,rvecs,tvecs,cameraMatrix,distCoeffs)

    print(f'totAverErrOpenCV: {totAverErrOpenCV}')
    print(f'AverSqErr: {mearErr}')
    print(f'cameraMatrix:\n {cameraMatrix}')
    print(f'distCoeffs:\n {distCoeffs}')

    return (True,cameraMatrix,distCoeffs)