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
        if (_pars.vis.previewPatterns>0):
            imgShow = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            ptsMat = np.copy(pts)
            cv.drawChessboardCorners(imgShow, patDims, ptsMat, True)
            s=_pars.vis.previewPatterns
            imgShowScaled = cv.resize(imgShow,None,None,s,s)
            previewImg(imgShowScaled, "patternImg", 500)
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

def saveIntrinsics(fnSave,cameraMatrix,distCoeffs):
    try:
        fs=cv.FileStorage(fnSave,cv.FileStorage_WRITE | cv.FILE_STORAGE_FORMAT_JSON)
        fs.write('cameraMatrix',cameraMatrix)
        fs.write('distCoeffs',distCoeffs)
        fs.release()
        return True
    except Exception:
        return False

def loadIntrinsics(fnLoad):
    try:
        fs=cv.FileStorage(fnLoad,cv.FileStorage_READ | cv.FILE_STORAGE_FORMAT_JSON)
        cameraMatrix = np.array(fs.getNode('cameraMatrix').mat())
        distCoeffs = np.array(fs.getNode('distCoeffs').mat())
        fs.release()
        return (cameraMatrix, distCoeffs)
    except Exception as e:
        print(f'Error: {e}')
        return None
def saveHandEye(fnSave,T_robot_cam:np.array,T_flange_pattern:np.array):
    try:
        fs=cv.FileStorage(fnSave,cv.FileStorage_WRITE | cv.FILE_STORAGE_FORMAT_JSON)
        fs.write('T_robot_cam',T_robot_cam)
        fs.write('T_flange_pattern',T_flange_pattern)
        fs.release()
        return True
    except Exception:
        return False

def loadHandEye(fnLoad):
    try:
        fs=cv.FileStorage(fnLoad,cv.FileStorage_READ | cv.FILE_STORAGE_FORMAT_JSON)
        T_robot_cam = np.array(fs.getNode('T_robot_cam').mat())
        T_flange_pattern = np.array(fs.getNode('T_flange_pattern').mat())
        fs.release()
        return (T_robot_cam,T_flange_pattern)
    except Exception as e:
        print(f'Error: {e}')
        return None
    
def img2uint8Percent(im:np.array,percentile100=100.0):
    mqMq=[(100-percentile100)/2,100-(100-percentile100)/2]
    mM=np.percentile(im,mqMq)
    return np.clip(255*(im-mM[0])/(mM[1]-mM[0]),0,255).astype(np.uint8)

def invPosRot(posRot):
    p,r=posRot
    #p_b = R_b_g * p_g  + t_b_g
    #p_g = R_b_g^T * p_b - R_b_g^T * t_b_g
    p = -r.transpose()@p
    r=r.transpose()
    return (p,r)
def posRot2posAndRot33(posRot, inv=False):
    pos=[]
    rot=[]
    for (p,r) in posRot:
        if not(r.shape[0]==3 and r.shape[1]==3):
            r,_=cv.Rodrigues(r)
        if inv:
            (p,r)=invPosRot((p,r))
        pos.append(np.array(p).reshape(3,1))
        rot.append(r)
    return pos, rot

def posRot2M44(posRot):
    p,r=posRot
    ret=np.eye(4)
    if not(r.shape[0]==3 and r.shape[1]==3):
        r,_=cv.Rodrigues(r)
    ret[0:3,0:3]=r
    ret[0:3,3]=np.array(p).ravel()
    return ret




#blobs rojo y verde esquinas
def segmentColoredBlobs(imBgr:np.ndarray,colors:np.ndarray, preview = False):
    imBgrUint = img2uint8Percent(imBgr,100)
    imHsv = cv.cvtColor(imBgrUint,cv.COLOR_BGR2HSV)
    h=imHsv[:,:,0]
    h=cv.medianBlur(h,3)
    s=imHsv[:,:,1]
    s=cv.medianBlur(s,3)
    regRed = 255*np.logical_and(h<64,s>150).astype(np.uint8)
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    regRed=cv.morphologyEx(regRed,cv.MORPH_CLOSE,se)
    regGreen = 255*np.logical_and(s>100,np.logical_and(h>80,h<150)).astype(np.uint8)
    regGreen=cv.morphologyEx(regGreen,cv.MORPH_CLOSE,se)
    def filterContours(contours):
        valids=[]
        centers=[]
        for cnt in contours:
            area=cv.contourArea(cnt)
            if not(area > 1000 and area < 6000): continue
            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            if not(circularity>0.2 and circularity <= 1000): continue
            rectRotated = cv.minAreaRect(cnt)
            sz = rectRotated[1]
            if not(sz[0]/sz[1] > 0.8 and sz[0]/sz[1] < 1.2): continue
            
            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centers.append((cx,cy))
            valids.append(cnt)
        return (centers,valids)

    cnts,im2 = cv.findContours(regRed, cv.RETR_LIST , cv.CHAIN_APPROX_SIMPLE)
    ctrRed,cntsRed=filterContours(cnts)
    cnts,im2 = cv.findContours(regGreen, cv.RETR_LIST , cv.CHAIN_APPROX_SIMPLE)
    ctrGreen,cntsGreen=filterContours(cnts)

    if len(ctrRed)!=1 and len(ctrGreen)!=1:
        return None
    if preview:
        # imBgrUint = cv.drawContours(imBgrUint, cntsRed, -1, (0,255,0), 3)
        cv.circle(imBgrUint, ctrRed[0],10,(0,0,255),-1)
        cv.circle(imBgrUint, ctrGreen[0],10,(0,255,0),-1)
        imBgrUint=cv.resize(imBgrUint,None,None,0.5,0.5)
        cv.imshow('previewWin',imBgrUint)
        cv.waitKey()
    return (ctrRed[0],ctrGreen[0])


#filtrado por varios atributos
def filterContours(contours, 
                   areaRange=[5000,30000], 
                   circularityRange=[0.5,1], 
                   rectWidthRange=None, rectAspectRatioRange=[0.7,1.4],
                   minRectFillRatio=None):
    valids=[]
    centers=[]
    for cnt in contours:
        area=cv.contourArea(cnt)
        if not(area > areaRange[0] and area < areaRange[1]): continue
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0: continue
        if circularityRange is not None:
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            if not(circularity>circularityRange[0] and circularity <= circularityRange[1]): continue

        rectRotated = cv.minAreaRect(cnt)
        center,sz,angle = rectRotated
        if rectAspectRatioRange is not None:
            if not(sz[0]/sz[1] > rectAspectRatioRange[0] and sz[0]/sz[1] < rectAspectRatioRange[1]): continue
        if rectWidthRange is not None:
            if not(sz[0]>rectWidthRange[0] and sz[0]<rectWidthRange[1]): continue
        if minRectFillRatio is not None:
            rectFillRatio = area/(sz[0]*sz[1])
            if rectFillRatio < minRectFillRatio: continue
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centers.append((cx,cy))
        valids.append(cnt)

    return (centers,valids)


def segmentGrayBlobs(gray:np.ndarray, preview = False):
    meanAdaptBlockSize = 50
    meanAdaptBlockSize = meanAdaptBlockSize+(meanAdaptBlockSize+1)%2#impar
    thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, meanAdaptBlockSize,2)
    thresh = cv.bitwise_not(thresh)
    # cv.imshow('win',thresh)
    # cv.waitKey()
    contours,im2 = cv.findContours(thresh, cv.RETR_LIST , cv.CHAIN_APPROX_SIMPLE)

    # imgShow=gray.copy()
    # imgShow=cv.drawContours(imgShow, contours, -1, 255, cv.FILLED)
    # imgShowResize=cv.resize(imgShow,None,None,0.25,0.25)
    # cv.imshow('win',imgShowResize)
    # cv.waitKey()

    areaRange=[5000,20000]
    circularityRange=None
    rectWidthRange=[80,120]
    rectAspectRatioRange=[0.8,1.2]
    minRectFillRatio = 0.8
    centers,validContours = filterContours(contours,
            areaRange, circularityRange, 
            rectWidthRange, rectAspectRatioRange, minRectFillRatio)

    # imgShow=gray.copy()
    # imgShow=cv.drawContours(imgShow, validContours, -1, 255, cv.FILLED)
    # imgShowResize=cv.resize(imgShow,None,None,0.25,0.25)
    # cv.imshow('win',imgShowResize)
    # cv.waitKey()

    #Filtrar por fondo circundante // pegado sobre fondo blanco
    #   extraccion media pixeles de resta de blob dilatado
    blobRefCenterCandidates=[]
    se=cv.getStructuringElement(cv.MORPH_ELLIPSE,(40,40))
    for j,(center,contour) in enumerate(zip(centers,validContours)):
        imgContours=np.zeros(gray.shape,np.uint8)
        cv.drawContours(imgContours,contour,-1,255,cv.FILLED)
        imgContourDilate=cv.dilate(imgContours,se)
        imgExtContours=imgContourDilate-imgContours
        meanCont = cv.mean(gray,mask=imgExtContours)[0]
        meanBlob = cv.mean(gray,mask=imgContours)[0]

        # imgShow=gray.copy()
        # cv.drawContours(imgShow,contour,-1,255,cv.FILLED)
        # print(j,meanBlob,meanCont)
        # imgShowResize = cv.resize(imgExtContours,None,None,0.25,0.25)
        # cv.imshow('win',imgShowResize)
        # cv.waitKey()
        
        if meanCont>80 and meanBlob < meanCont:
            blobRefCenterCandidates.append(center)

    if len(blobRefCenterCandidates)==0:
        print(f'No se detectaron blobs de referencia')
        return None
    if preview:
        imgShow = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        for j,(c,menVal) in enumerate(blobRefCenterCandidates):
            cv.putText(imgShow, f'{j}',(c[0]+50,c[1]+50), 1, 5, (0, 0, 255), 4)
            cv.circle(imgShow, c, 25, (0,0,255),cv.FILLED)
            imgShowResize = cv.resize(imgShow,None,None,0.25,0.25)
            cv.imshow('win',imgShowResize)
            cv.waitKey()
    return blobRefCenterCandidates

#Desambiguar origen patron con marcas anexas a origen (negro) y eje x (gris)
def disambiguatePatternPose(gray:np.array,rvec, tvec, cameraMatrix, distCoeffs):
    #esquinas patron
    pts=np.array([
        0,0,0,
        1,0,0,
        0,1,0,
        1,1,0
    ]).reshape(4,3).astype(float)
    pts*=_pars.pat.distEdges
    pts[:,0]*=_pars.pat.dims[0]-1
    pts[:,1]*=_pars.pat.dims[1]-1
    #Proyect pts img
    [imPts,_]=cv.projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs)
    imPts=imPts.squeeze()
    #asociar esquinas a blob cuadrado negro
    kps = segmentGrayBlobs(gray, False)
    if kps is None:
        print('no se encontro blob de referencia')
        return None
    #cuadrado de referencia cerca de origen, posteriormente de eje X
    #   si hay varios candidatos preserva el que tenga menor distancia respecto a cualquier esquina
    kps=np.array(kps)
    #kps vs esquinas
    deltas = imPts[:,np.newaxis,:]-kps[np.newaxis,:,:]
    dists = np.linalg.norm(deltas,2,axis=2)

    minsEachKP = np.min(dists,axis=0).squeeze()
    idKpClose = np.argmin(minsEachKP)
    sortIdx = np.argsort(dists[:,idKpClose],axis=0)
    if sortIdx[0]==0 and sortIdx[1]==1:
        return tvec, rvec
    
    o = pts[sortIdx[0],:]
    x = pts[sortIdx[1],:]
    v0x = (x-o)/np.linalg.norm(x-o,2)
    v0y = np.cross(np.array([0,0,1]),v0x)
    v0y = v0y/np.linalg.norm(v0y,2)
    R_orig_new = np.eye(3,3)
    R_orig_new[:,0]=v0x
    R_orig_new[:,1]=v0y
    R_cam_orig,_ = cv.Rodrigues(rvec)
    tvec = R_cam_orig@o.transpose()+tvec.squeeze()
    R_cam_new = R_cam_orig@R_orig_new
    rvec,_=cv.Rodrigues(R_cam_new)
    return tvec,rvec
# disambiguatePatternPose
