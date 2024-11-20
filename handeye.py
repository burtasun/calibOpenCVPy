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