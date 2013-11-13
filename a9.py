#npr.py
#import a2
import numpy as np
import random as rnd
import nprHelper as helper
import math

def brush(out, y, x, color, texture):
    '''out: the image to draw to. y,x: where to draw in out. color: the color of the stroke. texture: the texture of the stroke.'''
    coloredBox = np.zeros(texture.shape)
    coloredBox[:] = color
    textureH, textureW, textureC = texture.shape
    outH, outW, outC = out.shape
    halfH = int(textureH / 2)
    halfW = int(textureW / 2)
    if y - halfH >= 0 and y + halfH <= outH - 1 and x - halfW >= 0 and x + halfW <= outW - 1:
    	out[y - halfH : y - halfH + textureH, x - halfW : x - halfW + textureW] = texture * coloredBox + (1 - texture) * out[y - halfH : y - halfH + textureH, x - halfW : x - halfW + textureW]

def singleScalePaint(im, out, importance, texture, size=10, N=1000, noise=0.3):
    '''Paints with all brushed at the same scale using importance sampling.'''
    scaleFactor = min(float(size) / texture.shape[0], float(size) / texture.shape[1])
    scaledTexture = helper.scaleImage(texture, scaleFactor) if scaleFactor < 1 else texture.copy()
    if np.sum(importance) > 0:
	    for i in xrange(N * int(1.0 / np.average(importance))):
	    	y, x = (rnd.randrange(0, im.shape[0]), rnd.randrange(0, im.shape[1]))
	    	if rnd.random() <= importance[y, x, 0]:
		    	color = im[y, x] * (1 - (float(noise) / 2) + (noise * np.random.rand(3)))
		    	brush(out, y, x, color, scaledTexture)

def painterly(im, texture, N=10000, size=50, noise=0.3):
    '''First paints at a coarse scale using all 1's for importance sampling, then paints again at size/4 scale using the sharpness map for importance sampling.'''
    out = np.zeros(im.shape)
    singleScalePaint(im, out, np.ones(im.shape), texture, size, N, noise)
    singleScalePaint(im, out, helper.sharpnessMap(im), texture, size/4, N, noise)
    return out

def computeAngles(im):
    '''Return an image that holds the angle of the smallest eigenvector of the structure tensor at each pixel. If you have a 3 channel image as input, just set all three channels to be the same value theta.'''
    thetas = np.zeros(im.shape)
    tensors = helper.computeTensor(im)
    for y in xrange(im.shape[0]):
    	for x in xrange(im.shape[1]):
    		tensor = tensors[y, x]
    		eigvals, eigvecs = np.linalg.eigh(tensor)
    		vec = eigvecs[np.argmin(eigvals)]
    		thetas[y, x] = np.arctan2(vec[0], vec[1])
    return thetas

def singleScaleOrientedPaint(im, out, thetas, importance, texture, size, N, noise, nAngles=36):
    '''same as single scale paint but now the brush strokes will be oriented according to the angles in thetas.'''
    brushes = helper.rotateBrushes(texture, nAngles)
    scaleFactor = min(float(size) / texture.shape[0], float(size) / texture.shape[1])

    if scaleFactor < 1:
    	brushes = map(lambda brush: helper.scaleImage(brush, scaleFactor), brushes)

    if np.sum(importance) > 0:
	    for i in xrange(N * int(1.0 / np.average(importance))):
	    	y, x = (rnd.randrange(0, im.shape[0]), rnd.randrange(0, im.shape[1]))
	    	if rnd.random() <= importance[y, x, 0]:
	    		angleIndex = int(round(thetas[y, x, 0] * nAngles / (2 * math.pi)))
	    		if angleIndex >= nAngles:
	    			angleIndex = nAngles - 1
		    	color = im[y, x] * (1 - (float(noise) / 2) + (noise * np.random.rand(3)))
		    	brush(out, y, x, color, brushes[angleIndex])

def orientedPaint(im, texture, N=7000, size=50, noise=0.3):
    '''same as painterly but computes and uses the local orientation information to orient strokes.'''
    thetas = computeAngles(im)
    out = np.zeros(im.shape)
    singleScaleOrientedPaint(im, out, thetas, np.ones(im.shape), texture, size, N, noise)
    singleScaleOrientedPaint(im, out, thetas, helper.sharpnessMap(im), texture, size/4, N, noise)
    return out
