import os
import sys
import glob
import math
import numpy as np
import cv2 as cv

from Pars import _pars
from ocvAuxFuncs import *

    

def cleanupAndExitFail():
    cv.destroyAllWindows()
    exit(1)


if __name__=='__main__':
    print('main')
    #Objetivos
    #   Generar calibración
    #   Rectificación
    #   Con factor de escala
    #       Combrobación entre centroides o esquinas
    #   Pasarlo a halcon
    #   Integrar en lazo de control

    objp = createPtsPattern()
    objpoints = [] # pts frame local patron 3D
    imgpoints = [] # pts imagen 2D px

    #blob detector opencv
    try:
        blobDetector = cv.SimpleBlobDetector.create()
        blobDetector.read('blobDetector.dat')
    except:
        blobDetector.write('blobDetector.dat')
        print('creando blobDetector.dat')
    #Obtener lista de centroides / aristas set imagenes
    images = glob.glob(_pars.pathImgs)
    idValids=[]#para previsualizacion posterior
    for i,fname in enumerate(images):
        # print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if _pars.vis.previewImg:
             previewImg(gray, f'im {fname}')
        #extraer puntos ref img
        [found,pts]=getPtsImg(gray, blobDetector)
        if found==False:
            print(f'imagen {fname} sin patron detectado!')
            continue
        idValids.append(i)
        #añadir puntos frame patron 3d
        objpoints.append(objp)
        #añadir puntos imagen
        imgpoints.append(pts)
        print(f'{i}/{len(images)}')
    if len(imgpoints)<_pars.minValidImgs:
         print(f'el numero de imagenes con patron detectado, {len(imgpoints)}, es inferior a {_pars.minValidImgs}, SALIENDO')
         cleanupAndExitFail()
    [imWidth,imHeight]=gray.shape

    print('calibrando')
    [okCalib,cameraMatrix,distCoeffs] = runCalibration(objpoints,imgpoints, imWidth, imHeight)

    if okCalib==False:
        print("Calibracion fallida")
        exit(1)
    
    fnSave = f'{_pars.pathDirSave}intrinsics.json'
    res = saveIntrinsics(fnSave,cameraMatrix,distCoeffs)
    if res==False:
        print('No se han guardado los parametros de calibracion')
    print(f'Calibracion guardada en {fnSave}')

    if _pars.vis.viewUndistort:
        for idValid in idValids:
            img = cv.imread(images[idValid],cv.IMREAD_GRAYSCALE)
            imRect = cv.undistort(img, cameraMatrix, distCoeffs)
            imBoth = np.zeros((img.shape[0],img.shape[1]*2),dtype=np.uint8)
            imBoth[:,:img.shape[1]]=img
            imBoth[:,img.shape[1]:]=imRect
            imBothResize = cv.resize(imBoth,None,None,0.5,0.5)
            imBothResize=cv.cvtColor(imBothResize, cv.COLOR_GRAY2BGR)
            cv.putText(imBothResize,'ORIG',(int(imBothResize.shape[1]*0.25),20), 1, 1, (0, 255, 0))
            cv.putText(imBothResize,'RECT',(int(imBothResize.shape[1]*0.75),20), 1, 1, (0, 255, 0))
            previewImg(imBothResize, (f'{images[idValid]}'), 2000)

    if okCalib==False:
         cleanupAndExitFail()
    
#main