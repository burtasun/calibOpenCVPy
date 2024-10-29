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
    #output as ordered list
    ret = extractor.get_position_and_rotation_matrices()
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
#Hand eye
#   estimando poses patron respecto a camara
#       se asume que se dispone de parametros calibracion camara
if __name__=='__main__':
    np.set_printoptions(precision=3)
    print('hand eye')
    _pars.pathImgs = './handeye/3/*.tiff'
    _pars.handEye.pathJsonRobotPoses='./handeye/3/handEyePts.dat'
    #Task: handEyeCalib
    #   Parse set pts robot
    #       list tuple pos & R33
    posRot_robot_flange_all = getRobotPoseAsPosRot(_pars.handEye.pathJsonRobotPoses)
    if (posRot_robot_flange_all is None) or (len(posRot_robot_flange_all)==0):
        print(f'no se pudo parsear {_pars.handEye.pathJsonRobotPoses}')

    #Get cam intrinsic and distCoeffs
    fnIntrinsics = f'{_pars.pathDirSave}intrinsics.json'
    [cameraMatrix,distCoeffs] = loadIntrinsics(fnIntrinsics)


    #   pts patron en frame local
    objp = createPtsPattern()

    #PosRot patron respecto a camara
    posRot_cam_pattern = []
    #Poses tcp respecto a robot
    posRot_robot_flange = []

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
        #extraer frame patron respecto a camara
        flags = cv.SOLVEPNP_SQPNP
        [retval, rvec, tvec]=cv.solvePnP(objp,pts,cameraMatrix,distCoeffs,flags=flags)
        if retval==False:
            print(f'{i} sin identificar pose patron respecto a camara')
            continue
        #Desambiguar origen patron con marcas de colores
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
            kps = segmentColoredBlobs(imBgr,np.array([0,0,255,0,255,0]).reshape((2,3)).astype(np.uint8),False)
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
                return tvec, rvec
            o = pts[ids[0],:]
            x = pts[ids[1],:]
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
        tvec,rvec=disambiguatePatternPose(imBgr,rvec,tvec,cameraMatrix,distCoeffs)
        
        #añadir poses robot & pose patron
        posRot_cam_pattern.append((tvec,rvec))
        posRot_robot_flange.append(posRot_robot_flange_all[i])
        idValids.append(i)
        print(f'{i+1}/{len(pathsImages)}')

        if _pars.vis.viewUndistort:
            imBgrRect=imBgr.copy()
            for i in range(3):
                imBgrRect[...,i]=cv.undistort(imBgr[...,i], cameraMatrix, distCoeffs)
            pts=np.zeros([4,3],np.float32)
            pts[1:,:]=np.eye(3,3)*_pars.pat.distEdges
            [imPts,_]=cv.projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs)
            imPts=imPts.squeeze().astype(np.uint)
            for id in range(1,4):
                cv.line(imBgrRect,imPts[0,:],imPts[id,:],((id==3)*255,(id==2)*255,(id==1)*255),3)
            previewImg(imBgrRect)
    #buclePoses

    print(f'{len(idValids)} / {len(pathsImages)}!')
    print('Hand eye calibration')

    t_robot_flange, r_robot_flange = posRot2posAndRot33(posRot_robot_flange,False)
    t_flange_robot, r_flange_robot = posRot2posAndRot33(posRot_robot_flange,True)
    t_cam_pattern, r_cam_pattern = posRot2posAndRot33(posRot_cam_pattern,False)
    t_pattern_cam, r_pattern_cam = posRot2posAndRot33(posRot_cam_pattern,True)

    #OpenCV denom: world = pattern // base: robot base //
    #Eye in hand
    #   => parametros por defecto
    #Pattern in hand / static eye
    #   => invertir camara y patron // 'world'(pattern) <=> 'cam'
    r_cam_robot = np.eye(3)
    t_cam_robot = np.zeros([3,1])
    r_pattern_flange = np.eye(3)
    t_pattern_flange = np.zeros([3,1])
    cv.calibrateRobotWorldHandEye(
        #inputs
        R_world2cam=r_pattern_cam,
        t_world2cam=t_pattern_cam,
        R_base2gripper=r_flange_robot,
        t_base2gripper=t_flange_robot,
        #outputs
        R_base2world=r_cam_robot,
        t_base2world=t_cam_robot,
        R_gripper2cam=r_pattern_flange,
        t_gripper2cam=t_pattern_flange,
        method=cv.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)

    T_robot_cam = posRot2M44(invPosRot((t_cam_robot,r_cam_robot)))
    print(f'T_robot_cam\n{T_robot_cam}')
    T_flange_pattern = posRot2M44(invPosRot((t_pattern_flange,r_pattern_flange)))
    print(f'T_flange_pattern\n{T_flange_pattern}')

    def computeMeanErrors():
        #Eye(4)~=T_cam_pattern_i * T_pattern_flange * T_flange_robot_i * T_robot_cam
        #   2 métricas
        #       media errores posición: err = (1/N)*Sum_i^N{||tx^2+ty^2+tz^2||^0.5}
        #       media errores rotación theta rodriguez: err = (1/N)*Sum_i^N{theta_i}
        T_pattern_flange = posRot2M44((t_pattern_flange,r_pattern_flange))
        T_robot_cam = posRot2M44(invPosRot((t_cam_robot,r_cam_robot)))
        
        N = len(posRot_cam_pattern)
        errsPos=np.zeros(N)#dists
        errsRot=np.zeros(N)#rot
        for i in range(N):
            errEye_i = \
                posRot2M44(posRot_cam_pattern[i]) @ \
                T_pattern_flange @ \
                posRot2M44(invPosRot((t_robot_flange[i], r_robot_flange[i]))) @ \
                T_robot_cam
            errsPos[i]=np.linalg.norm(errEye_i[0:3,3])
            #norm rodriguez (openCV) => theta
            errsRot[i]=np.linalg.norm(cv.Rodrigues(errEye_i[0:3,0:3])[0])
        meanErrs_posRot = np.array([np.mean(errsPos), np.mean(errsRot)])
        print(f'errsPos_MM:\n{errsPos}\n\n')
        print(f'errsRot_Deg:\n{errsRot*180/math.pi}\n\n')
        print(f'meanErrs\n\tPos_MM {meanErrs_posRot[0]}\n\tRot_Deg {meanErrs_posRot[1]*180/math.pi}\n\n')
        return meanErrs_posRot
    #computeMeanErrors
    meanErrs_posRot = computeMeanErrors()

    if not(meanErrs_posRot[0]<_pars.handEye.maxErrPosMM and meanErrs_posRot[1]<_pars.handEye.maxErrRotDeg*math.pi/180):
        print(f'Error de calibración excesivos:\n')
        print(f'\t\tcalib\tmax\n')
        print(f'\tpos\t{meanErrs_posRot[0]}\t{_pars.handEye.maxErrPosMM}')
        print(f'\trot\t{meanErrs_posRot[1]*180/math.pi}\t{_pars.handEye.maxErrRotDeg}')
        exit(1)
    fnSave = f'{_pars.pathDirSave}handEye.json'        
    if saveHandEye(fnSave,T_robot_cam,T_flange_pattern):
        print('Handeye params saved')
    else:
        print('Handeye params NOT saved')

#main