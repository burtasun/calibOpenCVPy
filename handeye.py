import os
import sys
import glob
import math
import numpy as np
import cv2 as cv
import tiffile

from Pars import _pars, Pars
from ocvAuxFuncs import *
from kukaDat2XYZQUAT import E6PosExtractor
    

def cleanupAndExitFail():
    cv.destroyAllWindows()
    exit(1)

def getRobotPoseAsPosRot(jsonPath):
    extractor = E6PosExtractor()
    f = open(jsonPath,'r')
    if f is None:
        None
    file_content = f.read()
    extractor.extract_positions(file_content)
    posMats = extractor.get_position_and_rotation_matrices()
    #output as ordered list
    ret=[]
    for [key,item] in posMats.items():
        ret.append(item)
    return ret

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

#Hand eye
#   estimando poses patron respecto a camara
#       se asume que se dispone de parametros calibracion camara
if __name__=='__main__':
    print('hand eye')
    _pars.pathImgs = './handeye/3/*.tiff'
    _pars.handEye.pathJsonRobotPoses='./handeye/3/handEyePts.dat'
    #Task: handEyeCalib
    #   Parse set pts robot
    #       list tuple pos & R33
    posRot_robot_flange = getRobotPoseAsPosRot(_pars.handEye.pathJsonRobotPoses)
    if (posRot_robot_flange is None) or (len(posRot_robot_flange)==0):
        print(f'no se pudo parsear {_pars.handEye.pathJsonRobotPoses}')

    #Get cam intrinsic and distCoeffs
    fnIntrinsics = f'{_pars.pathDirSave}intrinsics.json'
    [cameraMatrix,distCoeffs] = loadIntrinsics(fnIntrinsics)


    #   pts patron en frame local
    objp = createPtsPattern()

    #PosRot patron respecto a camara
    posRot_cam_pattern = []

    #blob detector opencv para extraccion puntos
    try:
        blobDetector = cv.SimpleBlobDetector.create()
        blobDetector.read(_pars.handEye.blobDetectPathPars)
    except:
        blobDetector.write(_pars.handEye.blobDetectPathPars)
        print(_pars.handEye.blobDetectPathPars)
    #Obtener lista de centroides / aristas set imagenes
    pathsImages = glob.glob(_pars.pathImgs)
    idValids=[]#para asociacion posterior
    # pathsImages=pathsImages[-2:]
    for i,fname in enumerate(pathsImages):
        im = tiffile.imread(fname)
        imBgr = cv.cvtColor(img2uint8Percent(im,100),cv.COLOR_RGB2BGR)#niapa
        gray = cv.cvtColor(imBgr,cv.COLOR_BGR2GRAY)
        if _pars.vis.previewImg:
             previewImg(imBgr, f'im {fname}')
        #extraer puntos ref img
        [found,pts]=getPtsImg(gray, blobDetector)
        if found==False:
            print(f'imagen {fname} sin patron detectado!')
            continue
        idValids.append(i)
        #extraer frame patron respecto a camara
        flags = cv.SOLVEPNP_SQPNP
        [retval, rvec, tvec]=cv.solvePnP(objp,pts,cameraMatrix,distCoeffs,flags=flags)
        if retval==False:
            print(f'{i} mal')
            continue
        idValids.append(i)
        posRot_cam_pattern.append((tvec,cv.Rodrigues(rvec)))
        print(f'{i+1}/{len(pathsImages)}')

        #Desambiguar origen patron
        def disambiguatePatternPose(imBgr:np.array,rvec, tvec, cameraMatrix, distCoeffs):
            #empleando circulo rojo (origen) y verde (eje X)
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
            #asociar esquinas a puntos de colores por distancia minima
            kps = segmentColoredBlobs(imBgr,np.array([0,0,255,0,255,0]).reshape((2,3)).astype(np.uint8),True)
            if kps is None:
                return None
            ids=[]
            kps=np.array(kps)
            for kp in kps:
                dists = np.linalg.norm(imPts-kp[np.newaxis,:],2,axis=1)
                argmin = np.argmin(dists)
                if dists[argmin]<200:
                    ids.append(int(argmin))
            if ids[0]==0 and ids[1]==1:
                return rvec, tvec
            o = pts[ids[0],:]
            x = pts[ids[1],:]
            v0x = (x-o)/np.linalg.norm(x-o,2)
            v0x = (x-o)/np.linalg.norm(x-o,2)
            v0y = np.cross(np.array([0,0,1]),v0x)
            v0y = v0y/np.linalg.norm(v0y,2)
            R_orig_new = np.eye(3,3)
            R_orig_new[:,0]=v0x
            R_orig_new[:,1]=v0y
            R_cam_orig,_ = cv.Rodrigues(rvec)
            tvec = R_cam_orig@o.transpose()+tvec.squeeze()
            R_cam_new = R_cam_orig@R_orig_new
            rvec,_ = cv.Rodrigues(R_cam_new)
            return rvec,tvec
        # disambiguatePatternPose

        rvec,tvec=disambiguatePatternPose(imBgr,rvec,tvec,cameraMatrix,distCoeffs)

        if _pars.vis.viewUndistort:
            imRect = cv.undistort(gray, cameraMatrix, distCoeffs)
            imRgb = cv.cvtColor(imRect,cv.COLOR_GRAY2RGB)
            pts=np.zeros([4,3],np.float32)
            pts[1:,:]=np.eye(3,3)*_pars.pat.distEdges
            [imPts,_]=cv.projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs)
            imPts=imPts.squeeze().astype(np.uint)
            for id in range(1,4):
                cv.line(imRgb,imPts[0,:],imPts[id,:],((id==3)*255,(id==2)*255,(id==1)*255),3)
            previewImg(imRgb)


    print(f'{len(idValids)} / {len(pathsImages)}!')
    # if len(imgpoints)<_pars.minValidImgs:
    #      print(f'el numero de imagenes con patron detectado, {len(imgpoints)}, es inferior a {_pars.minValidImgs}, SALIENDO')
    #      cleanupAndExitFail()
    # [imWidth,imHeight]=gray.shape

    # print('calibrando')
    # [okCalib,cameraMatrix,distCoeffs] = runCalibration(objpoints,imgpoints, imWidth, imHeight)

    # if okCalib==False:
    #     print("Calibracion fallida")
    #     exit(1)
    
    # fnSave = f'{_pars.pathDirSave}intrinsics.json'
    # try:
    #     fs=cv.FileStorage(fnSave,cv.FileStorage_WRITE | cv.FILE_STORAGE_FORMAT_JSON)
    #     fs.write('cameraMatrix',cameraMatrix)
    #     fs.write('distCoeffs',distCoeffs)
    #     fs.release()
    # except Exception:
    #     print('No se han guardado los parametros de calibracion')
    #     print(f'Calibracion guardada en {fnSave}')

    # if _pars.vis.viewUndistort:
    #     for idValid in idValids:
    #         img = cv.imread(images[idValid],cv.IMREAD_GRAYSCALE)
    #         imRect = cv.undistort(img, cameraMatrix, distCoeffs)
    #         imBoth = np.zeros((img.shape[0],img.shape[1]*2),dtype=np.uint8)
    #         imBoth[:,:img.shape[1]]=img
    #         imBoth[:,img.shape[1]:]=imRect
    #         imBothResize = cv.resize(imBoth,None,None,0.5,0.5)
    #         imBothResize=cv.cvtColor(imBothResize, cv.COLOR_GRAY2BGR)
    #         cv.putText(imBothResize,'ORIG',(int(imBothResize.shape[1]*0.25),20), 1, 1, (0, 255, 0))
    #         cv.putText(imBothResize,'RECT',(int(imBothResize.shape[1]*0.75),20), 1, 1, (0, 255, 0))
    #         previewImg(imBothResize, (f'{images[idValid]}'), 2000)

    # if okCalib==False:
    #      cleanupAndExitFail()
    
#main