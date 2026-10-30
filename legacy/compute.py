import math
import sys
import random

import cv2
import os
import matplotlib
matplotlib.use('QtAgg')  # or 'Qt5Agg' if you have Qt installed

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageSequence
from matplotlib import animation
from matplotlib import font_manager
import pathlib

# sys.path.append('D:/Project/Ete/ReEte')
sys.path.append('/home/ukiriz/Ete')
import copy
from line_profiler import LineProfiler
from matplotlib.widgets import Button
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib.image import imread
import imageio
import scipy.ndimage

fontNames = ['Yu Gothic', 'Segoe UI Symbol', 'Times New Roman']
'''
path1 = '/home/ukiriz/.local/share/fonts/seguisym.ttf'
prop1 = font_manager.FontProperties(fname=path1)
path2 = '/home/ukiriz/.local/share/fonts/arial-unicode-ms.ttf'
prop2 = font_manager.FontProperties(fname=path2)
fontNames = ['Noto Sans CJK JP', prop1.get_name(), prop2.get_name()]
'''

profiler = LineProfiler()

depthMax = 2
hexMap = {'坤卦': (0, 0, 0), '艮卦2': (0, 0, 1), '艮卦': (0, 0, 1), '坎卦': (0, 1, 0), '巽卦': (0, 1, 1),
          '中卦': (0.5, 0.5, 0.5),
          '震卦2': (1, 0, 0), '震卦': (1, 0, 0), '離卦': (1, 0, 1), '兌卦': (1, 1, 0), '乾卦': (1, 1, 1)}
biMap = {'太阴': (0, 0), '少阴': (0, 1), '少阳': (1, 0), '太阳': (1, 1)}
monoMap = {'阴': (0,), '阳': (1,)}
reversedHexMap = {value: key for key, value in hexMap.items()}
reversedBiMap = {value: key for key, value in biMap.items()}
reversedMonoMap = {value: key for key, value in monoMap.items()}
reversedAllMap = {**reversedHexMap, **reversedBiMap, **reversedMonoMap}
reversedSymMap = {(0, 0, 0): '☷', (0, 0, 1): '☶', (0, 1, 0): '☵', (0, 1, 1): '☴', (1, 0, 0): '☳',
                  (1, 0, 1): '☲', (1, 1, 0): '☱', (1, 1, 1): '☰', (0.5, 0.5, 0.5): '𝌀', (0, 0): '⚏', (0, 1): '⚎',
                  (1, 0): '⚍', (1, 1): '⚌', (0,): '⚋', (1,): '⚊'}
vecMap = {'坤卦': [(0, 0), (0, 0)], '艮卦': [(-1, -1), (1, 1)], '坎卦': [(-1, 1), (1, 1)],
          '巽卦': [(-1, 0), (1, 0)],
          '中卦': [(0, 0), (0.5, 0.5)], '震卦': [(1, 1), (1, 1)], '離卦': [(1, -1), (1, 1)],
          '艮卦2': [(0, -1), (0, 1)], '震卦2': [(1, 0), (1, 0)],
          '兌卦': [(0, 1), (0, 1)], '乾卦': [(0, 0), (1, 1)]}
charMap = {'ġ': (0, 0, 0), 'g': (0, 0, 1), 'r': (0, 1, 0), 'v': (0, 1, 1), 'n': (0.5, 0.5, 0.5), 'd': (1, 0, 0),
           'b': (1, 0, 1), 's': (1, 1, 0), 'ṡ': (1, 1, 1), 'h': (0, 0), 'l̇': (0, 1), 'ṫ': (1, 0), 'x': (1, 1),
           'q': (0,), 'j': (1,)}
reversedCharMap = {value: key for key, value in charMap.items()}
charMapDyn = {'ċ': (0, 0, 0), 'c': (0, 0, 1), 'l': (0, 1, 0), 'f': (0, 1, 1), 'm': (0.5, 0.5, 0.5), 't': (1, 0, 0),
              'p': (1, 0, 1), 'z': (1, 1, 0), 'ż': (1, 1, 1)}
reversedCharMapDyn = {value: key for key, value in charMapDyn.items()}
charMapAll = {**charMap, **charMapDyn}
dimMap = {0: 'y', 1: 'x'}
dynMap = {0: '靜', 1: '動'}
leafSymbols = list(charMapAll.keys())
nodeSymbols = ['series', 'parallel', 'body', 'plan']
replaceDict = {'series': '串', 'parallel': '並', 'body': '身', 'plan': '陣'}

colorMode = False
colorA = (255, 0, 0)
colorB = (0, 200, 200)
unknownTree = ['parallel', ['series', ['ṡ', 0, 0]], ['series', ['m', 0, 0]]]
unknownTree = ['parallel', ['series', ['n', 0, 0]]]
tweening = 2
global axes
global fig
global ani
global axPrev
global axNext
global buttonPrevTree
global buttonNextTree
global buttonPrevSeq
global buttonNextSeq
global curGalleryInd
global curSeqInd
global numGallery
global bodyDict
global frameWidth
global frameHeight
global globalPixelMap
global globalCurLayer


def thresholdImage(image):
    thresh = 100
    fn = lambda x: 255 if x > thresh else 0
    image = image.convert("L").point(fn, mode='1')
    return image


def detectFlow(prevFrame, curFrame):
    print(prevFrame.shape, curFrame.shape)
    flow = cv2.calcOpticalFlowFarneback(prevFrame, curFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def visualizeFlow(image, flow, step=16, arrowScale=5, scaleFactor=1):
    image = image.convert('RGB')
    image = np.array(image)
    h, w = flow.shape[:2]
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T * arrowScale
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    for (x1, y1), (x2, y2) in lines:
        x1, y1, x2, y2 = x1 * scaleFactor, y1 * scaleFactor, x2 * scaleFactor, y2 * scaleFactor
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.circle(image, (x2, y2), 1, (0, 255, 0), -1)
    return Image.fromarray(image)


def detectEdge(image, useCanny=False):
    if useCanny:
        threshold = 50  # Adjust this threshold to control edge density
        # Apply edge detection based on pixel intensity differences
        edges = cv2.Canny(np.array(image), threshold, threshold * 3)
        # cv2.imwrite('./images/denseEdgesImage.png', edges)
        return Image.fromarray(edges)
    else:
        # Converting the image to grayscale, as edge detection
        # requires input image to be of mode = Grayscale (L)
        image = image.convert("L")
        # image.save(r"./images/grayscale.png")
        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        image = image.filter(ImageFilter.FIND_EDGES)

        # Saving the Image Under the name Edge_Sample.png
        # image.save(r"./images/edgeSample.png")
        return image


def detectContour(image):
    image = np.array(image)
    # Step 3: Preprocess the image (optional)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 4: Edge detection (optional)
    edges = cv2.Canny(blurred_image, 50, 150)  # Adjust the thresholds as needed

    # Step 5: Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy)
    # Analyze the hierarchy to find outer and inner contours
    outer_contours = []
    inner_contours = []

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            # Contour with no parent, it's an outer contour
            outer_contours.append(contours[i])
        else:
            # Contour with a parent, it's an inner contour
            inner_contours.append(contours[i])

    # Draw outer contours in green and inner contours in red
    contour_image = np.copy(image)
    cv2.drawContours(contour_image, outer_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_image, inner_contours, -1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Detected Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 6: Draw contours on the original image (optional)
    # contour_image = np.copy(image)
    # Create a blank canvas to draw the contours on
    canvas = np.zeros_like(image)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0),
                     2)  # Draw all contours in green with a thickness of 2 pixels

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Detected Contours', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extractChannels(image):
    rgbImage = image.convert('RGB')
    redImage = rgbImage.getchannel('R')
    greenImage = rgbImage.getchannel('G')
    blueImage = rgbImage.getchannel('B')
    print(redImage, greenImage, blueImage)
    return redImage, greenImage, blueImage


def findBestFit(pixelSums=None, pixelLens=None, flowSums=None, flowAbsSums=None, verbose=False,
                punishment=5, punishmentBi=1.5, punishmentDyn=2):
    # Find the trigram/bigram that a pixel region fits the best
    # pixelLens is required; either pixelSums or flowSUms & flowAbsSUms is required
    maxSums = [length * 255 for length in pixelLens]  # largest pixel sum possible
    if flowSums is None:  # static, using pixelSums
        sensations = [0] * len(pixelSums)
    else:  # dynamic, using flowSUms
        sensations = [[0, 0], [0, 0]]  # two regions, i.e. [A, B] where A: vector sum, B: magnitude sum

    if flowSums is None:
        for i in range(len(pixelSums)):
            if maxSums[i] == 0:
                ratio = 0
            else:
                ratio = pixelSums[i] / maxSums[i]
            # k = 0.3  # detected contours
            k = 1  # gray image
            sensations[i] = math.sqrt(1 - math.pow(ratio - k, 2) / math.pow(k, 2))
            if sensations[i] > 1:
                sensations[i] = 1  # sensation clipping
            if verbose:
                print("ratio", ratio)
    else:
        for i in range(2):  # bipartite regions
            for j, sums in enumerate([flowSums, flowAbsSums]):  # vector analysis + magnitude analysis
                ratio = sums[i] / pixelLens[i]
                if verbose:
                    hint = ' abs ' if j == 1 else ' '
                    print("flow" + hint + "ratio", ratio)
                k = 1
                if ratio < 0:
                    if ratio < -k:
                        sensations[i][j] = -1  # clipping
                    else:
                        sensations[i][j] = -math.sqrt(1 - math.pow(ratio + k, 2) / math.pow(k, 2))
                else:
                    if ratio > k:
                        sensations[i][j] = 1  # clipping
                    else:
                        sensations[i][j] = math.sqrt(1 - math.pow(ratio - k, 2) / math.pow(k, 2))
    scores = []
    if flowSums is None:
        referent = hexMap if len(pixelSums) == 3 else biMap  # either static trigram or static bigram
        for hexName in referent.keys():
            example = referent[hexName]
            absDif = []
            avgSensations = sum(sensations) / len(sensations)
            avgExample = sum(example) / len(example)
            relSensations = []
            relExample = []
            relDif = []
            for j in range(len(pixelSums)):
                absDif.append(abs(sensations[j] - example[j]))
                relSensations.append(sensations[j] - avgSensations)
                relExample.append(example[j] - avgExample)
                relDif.append(math.pow(abs(relSensations[j] - relExample[j]), 1))  # magnify similarity/difference
            absDif = sum(absDif)
            relDif = sum(relDif)
            scores.append(absDif / 2 + relDif / 2)  # absolute and relative differences
            origScore = scores[-1]
            punished = False
            if len(example) == 2 and example[0] == example[1] or \
                    len(example) == 3 and example[0] == example[1] and example[1] == example[2]:  # pure static yin/yang
                punished = True
                scores[-1] *= punishment
            if len(example) == 2:  # bigram
                punished = True
                scores[-1] *= punishmentBi
            if verbose:
                if punished:
                    print("Punished", origScore, scores[-1])
                print(hexName, "Absolute", absDif, "Relative", relDif, "Total", scores[-1])
    else:
        referent = hexMap
        for hexName in referent.keys():
            flowDif = []
            flowAbsDif = []
            for j in range(2):  # upper/lower region
                flowDif.append(abs(sensations[j][0] - vecMap[hexName][0][j]))
                flowAbsDif.append(abs(sensations[j][1] - vecMap[hexName][1][j]))
            flowDif = sum(flowDif)
            flowAbsDif = sum(flowAbsDif)
            scores.append(flowDif / 2 + flowAbsDif / 2)  # vector and magnitude differences
            example = referent[hexName]
            if example[0] == example[1] and example[1] == example[2]:  # pure dynamic yin/yang
                if verbose:
                    print("Punished", scores[-1], scores[-1] * punishmentDyn)
                scores[-1] *= punishmentDyn
            if verbose:
                print(hexName, "Vector", flowDif, "Absolute", flowAbsDif, "Total", scores[-1])
    score = np.min(scores)
    bestHex = list(referent.values())[np.argmin(scores)]

    return bestHex, score, sensations, referent, scores


def getVariance(lst):
    lstSum = sum([math.pow(val - 127.5, 2) for val in lst])
    return lstSum / len(lst)


def copy2d(arr):
    return [row[:] for row in arr]


def countPixels(pixels, orient, divNum=3, needHelp=False):
    # If without help, the pixels should be transposed for horizontal orientation
    height = len(pixels)
    width = len(pixels[0])
    if orient == 0:  # tripartite using horizontal lines
        measureA = height
    elif orient == 1:  # tripartite using vertical lines
        measureA = width
        if needHelp:
            pixels = np.array(pixels).T.tolist()
    division = measureA / divNum
    pixelSums = []
    pixelLens = []
    for j in range(divNum):
        pixelDiv = []
        for k in range(round(division * j), round(division * (j + 1))):
            pixelDiv = np.concatenate([pixelDiv, pixels[k]])
        pixelLens.append(len(pixelDiv))
        pixelSums.append(sum(pixelDiv))

    return pixelSums, pixelLens


# Finds the best way to split the image based on its pixels and optical flow
def splitImage(pixels, origin, depth=0, flow=None, bothOrient=False, flexible=True, buddhist=False):
    global depthMax
    width = len(pixels[0])
    height = len(pixels)
    print("Depth", depth)
    globalBestOrient = 0
    globalBestDynOrient = 0
    globalBestScore = 10
    globalBestDynScore = 10
    bestHex = [None, None]
    bestOffsets = [None, None]
    bestDynHex = [None, None]
    bestDynOffsets = [None, None]
    for orient in [0, 1]:
        offsets = [[0], [0, 0]]

        if flow is not None:
            dynOffset = 0
        bestScore = 10
        bestDynScore = 10
        print("Orientation:", orient, "Width:", width, "Height:", height)
        if orient == 0:  # tripartite using horizontal lines
            measureA = height
            measureB = width
        elif orient == 1:  # tripartite using vertical lines
            measureB = height
            measureA = width
            pixels = np.array(pixels).T.tolist()
            if flow is not None:
                flow = np.array(flow).transpose((1, 0, 2)).tolist()

        pixelSums = []
        pixelLens = []
        sensations = []
        scores = []
        hexagrams = []
        for divI, divNum in enumerate([2, 3]):  # bigram and trigram
            pixelSums, pixelLens = countPixels(pixels, orient, divNum)
            results = findBestFit(pixelSums=pixelSums[divI], pixelLens=pixelLens[divI], verbose=True)
            hexagrams.append(results[0])
            scores.append(results[1])
            sensations.append(results[2])
            for j in range(divNum):
                print("Sensation:", sensations[divI][j], "Pixel sum:", pixelSums[divI][j], "Max sum:",
                      pixelLens[divI][j] * 255,
                      "From", round(division * j), "to", round(division * (j + 1)), 'score', scores[-1])
        score = min(scores)
        hexagram = hexagrams[np.argmin(scores)]
        bestPixelSums = copy2d(pixelSums)
        bestPixelLens = copy2d(pixelLens)
        bestSensations = copy2d(sensations)

        if flow is not None:
            flowSums = [None, None]
            flowAbsSums = [None, None]
            division = measureA / 2
            for i in range(2):
                flowLst = []
                flowAbsLst = []
                for j in range(round(division * i), round(division * (i + 1))):
                    flowLst += [tup[1 - orient] for tup in flow[j]]
                    flowAbsLst += [abs(tup[1 - orient]) for tup in flow[j]]

                flowSums[i] = sum(flowLst)
                flowAbsSums[i] = sum(flowAbsLst)
            results = findBestFit(pixelLens=pixelLens[1], flowSums=flowSums, flowAbsSums=flowAbsSums, verbose=True)
            dynHexagram = results[0]
            dynScore = results[1]
            flowSensations = results[2]

            bestFlowSensations = flowSensations.copy()
            bestFlowSums = flowSums.copy()
            bestFlowAbsSums = flowAbsSums.copy()
            for i in range(2):
                print("Flow:", flowSensations[i])

        if round(division) > 0 and flexible:
            for i in [0, 1]:
                for j, moveDir in enumerate([1, -1]):
                    for divI, divNum in enumerate([2, 3]):
                        if divNum == 2 and i > 0:
                            continue
                        newPixelSums = copy2d(pixelSums)
                        newPixelLens = copy2d(pixelLens)
                        newOffsets = copy2d(offsets)
                        newSensations = copy2d(sensations)
                        division = measureA / divNum
                        firstRound = True
                        while True:
                            position = round(division * (i + 1)) + newOffsets[divI][i] + moveDir * int(moveDir == -1)
                            pixelNum = sum(pixels[position])
                            deltaPixelSums = pixelNum * moveDir
                            deltaPixelLens = measureB * moveDir
                            if newPixelSums[divI][i] + deltaPixelSums <= 0 or newPixelSums[divI][
                                i + 1] - deltaPixelSums <= 0 \
                                    or newPixelLens[divI][i] + deltaPixelLens <= 0 or newPixelLens[divI][
                                i + 1] - deltaPixelLens <= 0:
                                # avoids squeezing the middle space or creating empty space
                                print("avoids squeezing & empty", newPixelSums[divI][i], newPixelSums[divI][i + 1],
                                      deltaPixelSums)

                                break
                            newOffsets[divI][i] += moveDir
                            for k in [0, 1]:
                                addSign = 1 if k == 0 else -1
                                newPixelSums[divI][i + k] += deltaPixelSums * addSign
                                newPixelLens[divI][i + k] += deltaPixelLens * addSign
                            results = findBestFit(pixelSums=newPixelSums[divI], pixelLens=newPixelLens[divI])
                            hexagram = results[0]
                            newScore = results[1]
                            newSensations[divI] = results[2]

                            '''
                            print("Captured pixels:", pixelNum, "Sensations", newSensations, 'Variances', newVariances,
                                  "Pos:", position,
                                  "Line:", i, "Dir:", moveDir, "Offsets", newOffsets, "New score", newScore)
                            '''
                            # print("subtle dif", score - newScore)
                            if firstRound:
                                score = newScore
                                firstRound = False
                                continue
                            if newScore >= score - 0.001:  # error-tolerant rate
                                break
                            score = newScore
                        print("divNum", divNum, "moveDir", moveDir, "Current score:", score, "Best score", bestScore,
                              hexagram, newPixelSums[divI])
                        if score < bestScore:
                            print("Best score", score, "Hexagram", hexagram, "Sensations", newSensations, "Offsets",
                                  newOffsets, "Orient", orient)
                            bestScore = score
                            bestPixelSums = newPixelSums
                            bestPixelLens = newPixelLens
                            bestSensations = newSensations
                            bestOffsets[orient] = newOffsets
                            bestHex[orient] = hexagram
                        if score < globalBestScore:
                            globalBestScore = score  # globally best offsets consider every orientation
                            globalBestOrient = orient

                    if flow is not None and i == 0:
                        newFlowSums = flowSums.copy()
                        newFlowAbsSums = flowAbsSums.copy()
                        newFlowSensations = flowSensations.copy()
                        newDynOffset = dynOffset
                        division = measureA / 2
                        firstRound = True
                        while True:
                            position = round(division) + newDynOffset + moveDir * int(moveDir == -1)
                            if position >= len(flow):
                                break
                            print("position", position, "len", len(flow))
                            flowNum = sum([tup[1 - orient] for tup in flow[position]])
                            flowAbsNum = sum([abs(tup[1 - orient]) for tup in flow[position]])
                            deltaFlowSums = flowNum * moveDir
                            deltaFlowAbsSums = flowAbsNum * moveDir
                            if newFlowAbsSums[i] + deltaFlowAbsSums <= 0 \
                                    or newFlowAbsSums[i + 1] - deltaFlowAbsSums <= 0:
                                # avoids squeezing the middle space or creating empty space
                                print(newFlowAbsSums[i], newFlowAbsSums[i + 1], deltaFlowAbsSums)
                                break
                            newDynOffset += moveDir
                            for k in [0, 1]:
                                addSign = 1 if k == 0 else -1
                                newFlowSums[i + k] += deltaFlowSums * addSign
                                newFlowAbsSums[i + k] += deltaFlowAbsSums * addSign
                            results = findBestFit(pixelLens=newPixelLens[1], flowSums=newFlowSums,
                                                  flowAbsSums=newFlowAbsSums)
                            dynHexagram = results[0]
                            newDynScore = results[1]
                            newFlowSensations = results[2]
                            '''
                            print("Captured pixels:", pixelNum, "Sensations", newSensations, 'Variances', newVariances,
                                  "Pos:", position,
                                  "Line:", i, "Dir:", moveDir, "Offsets", newOffsets, "New score", newScore)
                            '''
                            # print("subtle dif", score - newScore)
                            if firstRound:  # it doesn't matter since we only have one type of dynamic score
                                dynScore = newDynScore
                                firstRound = False
                                continue
                            if newDynScore >= dynScore:  # error-tolerant rate
                                break
                            dynScore = newDynScore

                        if dynScore < bestDynScore:
                            print("Best score", dynScore, "Hexagram", dynHexagram, "Sensations", newFlowSensations,
                                  "Offsets",
                                  newDynOffset)
                            bestFlowSums = newFlowSums
                            bestFlowAbsSums = newFlowAbsSums
                            bestFlowSensations = newFlowSensations
                            bestDynOffsets[orient] = newDynOffset
                            bestDynHex[orient] = dynHexagram
                        if dynScore < globalBestDynScore:
                            globalBestDynScore = dynScore  # globally best offsets consider every orientation
                            globalBestDynOrient = orient
                pixelSums = bestPixelSums
                pixelLens = bestPixelLens
                sensations = bestSensations
                offsets = bestOffsets[orient]
                if flow is not None and i == 0:
                    flowSums = bestFlowSums
                    flowAbsSums = bestFlowAbsSums
                    flowSensations = bestFlowSensations
                    dynOffset = bestDynOffsets[orient]
        else:
            bestHex[orient] = hexagram
            bestOffsets[orient] = [[0], [0, 0]]
            if flow is not None:
                bestDynHex[orient] = dynHexagram
                bestDynOffsets[orient] = 0
            if score < globalBestScore:
                globalBestScore = score
                globalBestOrient = orient
                if flow is not None:
                    globalBestDynScore = dynScore
                    globalBestDynOrient = [0, 0]

    print("Best Hex", bestHex, "Best score", globalBestScore, "Best orientation", globalBestOrient,
          "Best Offsets",
          bestOffsets)

    offsetMap = [[0, 0], [1, 0], [0, 1]]
    offsetMapBi = [0, 1]
    iterable = [0, 1] if bothOrient else [globalBestOrient]
    tree = [[], []]
    rays = [[], []]
    tPixels = pixels
    pixels = np.array(tPixels).T.tolist()
    if flow is not None:
        tFlow = flow
        flow = np.array(flow).transpose((1, 0, 2)).tolist()

    for curOrient in iterable:
        divNum = len(bestHex[curOrient])
        divI = 0 if divNum == 2 else 1
        extOffsets = [0] + bestOffsets[curOrient][divI] + [0]
        if curOrient == 0:
            division = height / divNum
            rays[curOrient] += [[(origin[0], origin[1] + division * i + bestOffsets[curOrient][divI][i - 1]),
                                 (origin[0] + len(tPixels),
                                  origin[1] + division * i + bestOffsets[curOrient][divI][i - 1])]
                                for i in range(1, divNum)]
            if flow is not None:
                rays[curOrient] += [[(origin[0], origin[1] + height / 2 + bestDynOffsets[curOrient]),
                                     (origin[0] + len(tPixels), origin[1] + height / 2 + bestDynOffsets[curOrient])]]
            hexTup = [bestHex[curOrient], bestDynHex[curOrient]]
            if not buddhist:  # takes into account of the default evenness & stillness
                if hexTup[0] == (0.5, 0.5, 0.5):
                    hexTup[0] = []
                if hexTup[1] == (0, 0, 0):
                    hexTup[1] = []
            tree[curOrient] = [(origin[0] + width / 2, origin[1] + height / 2), tuple(hexTup)]
            if depth < depthMax:
                results = []
                for i in range(divNum):
                    if divNum == 3:
                        originY = origin[1] + height / divNum * i + offsetMap[i][0] * bestOffsets[curOrient][divI][0] + \
                                  offsetMap[i][1] * bestOffsets[curOrient][divI][1]
                    if divNum == 2:
                        originY = origin[1] + height / divNum * i + offsetMapBi[i] * bestOffsets[curOrient][divI][0]

                    print("Section", i, "Start", round(division * i) + extOffsets[i],
                          "End", round(division * (i + 1)) + extOffsets[i + 1],
                          "Origin", origin[0], originY)
                    if (round(division * (i + 1)) + extOffsets[i + 1]) - (round(division * i) + extOffsets[i]) < 3:
                        print("too narrow")
                        results.append([])
                        continue
                    if flow is not None:
                        subFlow = flow[round(division * i) + extOffsets[i]:
                                       round(division * (i + 1)) + extOffsets[i + 1]]
                    else:
                        subFlow = None
                    results.append(splitImage(pixels[round(division * i) + extOffsets[i]:
                                                     round(division * (i + 1)) + extOffsets[i + 1]],
                                              (origin[0], originY),
                                              depth=depth + 1,
                                              flow=subFlow,
                                              bothOrient=bothOrient,
                                              flexible=flexible, buddhist=buddhist))
                print('results', results, bestHex)
                if (results[0] == [] or results[0][1][0] == [] or not bestHex[0] == results[0][1][0][1]) or \
                        (results[1] == [] or results[1][1][0] == [] or not bestHex[0] == results[1][1][0][1]) or \
                        (results[2] == [] or results[2][1][0] == [] or not bestHex[0] == results[2][1][0][1]):
                    for result in results:
                        if result:
                            rays[curOrient] += result[0][0] + result[0][1]
                            tree[curOrient].append(result[1])
                        else:
                            tree[curOrient].append([])
                else:
                    if not bestHex[0] == results[0][1][1][1] \
                            or not bestHex[0] == results[1][1][1][1] \
                            or not bestHex[0] == results[2][1][1][1] \
                            or bestHex[0] not in [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)]:
                        for result in results:
                            if result:
                                rays += result[0][1]  # divide the other dimension
                                tree[curOrient].append([[], result[1][1]])
                            else:
                                tree[curOrient].append([])
                        print("only add orient", orient, tree)
                    else:
                        print("skip monotonous", bestHex[0])

        else:
            division = width / divNum

            rays[curOrient] += [[(origin[0] + division * i + bestOffsets[curOrient][divI][i - 1], origin[1]),
                                 (origin[0] + division * i + bestOffsets[curOrient][divI][i - 1],
                                  origin[1] + len(tPixels[0]))] for i
                                in range(1, divNum)]
            if flow is not None:
                rays[curOrient] += [[(origin[0] + width / 2 + bestDynOffsets[curOrient], origin[1]),
                                     (origin[0] + width / 2 + bestDynOffsets[curOrient], origin[1] + len(tPixels[0]))]]
            hexTup = [bestHex[curOrient], bestDynHex[curOrient]]
            if not buddhist:
                if hexTup[0] == (0.5, 0.5, 0.5):
                    hexTup[0] = []
                if hexTup[1] == (0, 0, 0):
                    hexTup[1] = []
            tree[curOrient] = [(origin[0] + width / 2, origin[1] + height / 2), tuple(hexTup)]
            if depth < depthMax:
                results = []
                for i in range(divNum):
                    if divNum == 3:
                        originX = origin[0] + width / divNum * i + offsetMap[i][0] * bestOffsets[curOrient][divI][0] + \
                                  offsetMap[i][1] * bestOffsets[curOrient][divI][1]
                    if divNum == 2:
                        originX = origin[0] + width / divNum * i + offsetMapBi[i] * bestOffsets[curOrient][divI][0]

                    print("Section", i, "Start", round(division * i) + extOffsets[i],
                          "End", round(division * (i + 1)) + extOffsets[i + 1],
                          "Ext offsets", extOffsets, "Division", division,
                          "Origin", originX, origin[1])
                    if (round(division * (i + 1)) + extOffsets[i + 1]) - (round(division * i) + extOffsets[i]) < 3:
                        print("too narrow")
                        results.append([])
                        continue
                    if flow is not None:
                        subFlow = np.array(tFlow[round(division * i) + extOffsets[i]:
                                                 round(division * (i + 1)) + extOffsets[i + 1]]).transpose(
                            (1, 0, 2)).tolist()
                    else:
                        subFlow = None
                    results.append(splitImage(np.array(tPixels[round(division * i) + extOffsets[i]:
                                                               round(division * (i + 1)) + extOffsets[
                                                                   i + 1]]).T.tolist(),
                                              (originX, origin[1]),
                                              depth=depth + 1,
                                              flow=subFlow,
                                              bothOrient=bothOrient,
                                              flexible=flexible, buddhist=buddhist))
                print(results, bestHex)
                if (results[0] == [] or results[0][1][1] == [] or not bestHex[1] == results[0][1][1][1]) or \
                        (results[1] == [] or results[1][1][1] == [] or not bestHex[1] == results[1][1][1][1]) or \
                        (results[2] == [] or results[2][1][1] == [] or not bestHex[1] == results[2][1][1][1]):
                    for result in results:
                        if result:
                            rays[curOrient] += result[0][0] + result[0][1]
                            tree[curOrient].append(result[1])
                        else:
                            tree[curOrient].append([])
                else:
                    if not bestHex[1] == results[0][1][0][1] \
                            or not bestHex[1] == results[1][1][0][1] \
                            or not bestHex[1] == results[2][1][0][1] \
                            or bestHex[1] not in [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)]:
                        for result in results:
                            if result:
                                rays[curOrient] += result[0][0]  # divide the other dimension
                                tree[curOrient].append([result[1][0], []])
                            else:
                                tree[curOrient].append([])
                        print("only add orient", orient, tree)
                    else:
                        print("skip monotonous", bestHex[1])
    return rays, tree


def produceDense():
    # Define image properties
    image_size = (1000, 1000)

    # Generate random noise image
    # random_image = np.random.randint(0, 256, image_size, dtype=np.uint8)
    random_image = np.random.choice([0, 255], image_size)
    # Save the random image
    cv2.imwrite('./images/randomImage.png', random_image)
    # detectEdge(random_image, useCanny=True)


def drawRays(image, rays, scaleFactor=1):
    global colorA
    global colorB
    rgbImage = image.convert('RGB')
    rayImage = ImageDraw.Draw(rgbImage)
    for orient in [0, 1]:
        branch = rays[orient]
        for ray in branch:
            newRay = [(ray[0][0] * scaleFactor, ray[0][1] * scaleFactor),
                      (ray[1][0] * scaleFactor, ray[1][1] * scaleFactor)]
            if ray[0][0] == ray[1][0]:
                color = colorB
            if ray[0][1] == ray[1][1]:
                color = colorA
            rayImage.line(newRay, fill=color, width=0)
    return rgbImage


def drawLabels(image, tree, allTrees=False, scaleFactor=1):
    rgbImage = image.convert('RGB')
    labelImage = ImageDraw.Draw(rgbImage)
    labelHelper(labelImage, tree, allTrees, scaleFactor=scaleFactor)
    return rgbImage


def labelHelper(image, tree, allTrees, scaleFactor=1):
    global colorA
    global colorB
    size = 30
    if not tree:
        return None
    unicodeFont = ImageFont.truetype("uming.ttc", size)
    for orient in [0, 1]:
        branch = tree[orient]
        if not branch:
            continue
        dynamicExists = branch[1][1] is not None

        for i, isDynamic in enumerate([False, True]):
            if not branch[1][i]:
                continue
            if dynamicExists:
                shiftX = -1 if isDynamic is False else 1
                wordLen = 2.5 if len(branch[1][i]) == 3 else 3.5
            else:
                if isDynamic:
                    break
                shiftX = -1
                wordLen = 1.5 if len(branch[1][i]) == 3 else 2.5
            shiftY = -0.5 if orient == 0 else 0.5
            coord = ((branch[0][0] + size / scaleFactor * wordLen / 2 * shiftX) * scaleFactor,
                     (branch[0][1] + size / scaleFactor * shiftY) * scaleFactor)
            if len(branch) == 2 or allTrees:
                if not orient:
                    color = colorA
                else:
                    color = colorB
                hexName = reversedHexMap[tuple(branch[1][i])][0] if len(branch[1][i]) == 3 else reversedBiMap[
                    tuple(branch[1][i])]

                # print('thar barr', branch, isDynamic, hexName + dynMap[i] * dynamicExists + dimMap[orient], branch[0], size / scaleFactor * wordLen / 2 * shiftX)
                # print(hexName)
                image.text(coord, hexName + dynMap[i] * dynamicExists + dimMap[orient], font=unicodeFont, fill=color)
            if len(branch) > 2:
                for subtree in branch[2:]:
                    labelHelper(image, subtree, allTrees, scaleFactor=scaleFactor)


def drawTree(tree):
    global depthMax
    width = height = int(math.pow(3, depthMax + 1))
    pixels = np.zeros((height, width, 2), dtype=int)
    pixels = treeHelper(tree, (0, 0), width, height, pixels)
    mask = pixels[:, :, 1] == 0
    pixels[mask, 1] = 1
    normPixels = pixels[:, :, 0] / pixels[:, :, 1] * 255
    array = np.array(normPixels, dtype=np.uint8)
    image = Image.fromarray(array, 'L')
    return image


def treeHelper(tree, origin, width, height, pixels):
    # print(tree, origin, width, height, pixels)
    for orient in [0, 1]:
        branch = tree[orient]
        if not branch or not branch[1][0]:
            continue
        divNum = len(branch[1][0])
        if orient == 0:
            division = height / divNum
            for i in range(divNum):
                for k in range(round(division * i), round(division * (i + 1))):
                    for j in range(width):
                        pixels[origin[1] + k, origin[0] + j, 0] += branch[1][0][i]
                        pixels[origin[1] + k, origin[0] + j, 1] += 1
                if len(branch) > 2 and branch[2 + i] != []:
                    subDiv = height / divNum
                    pixels = treeHelper(branch[2 + i], (origin[0], origin[1] + round(division * i)), width,
                                        round(subDiv * (i + 1)) - round(subDiv * i),
                                        pixels)
        elif orient == 1:
            division = width / divNum
            for i in range(divNum):
                for k in range(round(division * i), round(division * (i + 1))):
                    for j in range(height):
                        pixels[origin[1] + j, origin[0] + k, 0] += branch[1][0][i]
                        pixels[origin[1] + j, origin[0] + k, 1] += 1
                if len(branch) > 2 and branch[2 + i] != []:
                    subDiv = width / divNum
                    pixels = treeHelper(branch[2 + i], (origin[0] + round(division * i), origin[1]),
                                        round(subDiv * (i + 1)) - round(subDiv * i), height,
                                        pixels)

    return pixels


def convert2GuideTree(tree):
    guideTree = []
    for orient in [0, 1]:
        branch = tree[orient]
        if not branch:
            continue
        guideBranch = []
        for i, isDynamic in enumerate([False, True]):
            if not branch[1][i] or branch[1][i] is None:
                continue
            referent = reversedCharMapDyn if isDynamic else reversedCharMap
            char = referent[branch[1][i]]
            guideBranch.append(['series', [char, 1 - orient, 0]])
        compLst = []
        if len(branch) > 2:
            for comp in branch[2:]:
                if not comp:
                    continue
                subTree = convert2GuideTree(comp)
                if subTree:
                    if subTree[0] == 'series':
                        subTree = subTree[1:]
                        compLst += subTree
                    else:
                        compLst.append(subTree)
            if compLst:
                guideBranch.append(['series'] + compLst)
        if not guideBranch:
            continue
        elif len(guideBranch) > 1:
            guideTree.append(['series', ['parallel'] + guideBranch])
        else:
            guideTree.append(guideBranch[0])
    if not guideTree:
        return []
    elif len(guideTree) > 1:
        guideTree = ['parallel'] + guideTree
    else:
        guideTree = guideTree[0]
    return guideTree


def reverseGuideTree(guideTree):
    comps = guideTree[1:]
    if guideTree[0] == 'series':
        newGuideTree = ['series'] + comps[::-1]
        for i in range(len(comps)):
            if newGuideTree[1 + i] == 'filler':
                continue
            newGuideTree[1 + i] = reverseGuideTree(newGuideTree[1 + i])
    elif guideTree[0] in ['parallel', 'body', 'plan']:
        newGuideTree = [guideTree[0]]
        for i in range(len(comps)):
            if guideTree[1 + i] == 'filler' or isinstance(guideTree[1 + i], int):
                continue
            newGuideTree.append(reverseGuideTree(guideTree[1 + i]))
    else:
        newGuideTree = guideTree.copy()
        if guideTree[0] in charMap:
            newGuideTree[0] = reversedCharMap[charMap[guideTree[0]][::-1]]
        elif guideTree[0] in charMapDyn:
            newGuideTree[0] = reversedCharMapDyn[charMapDyn[guideTree[0]][::-1]]

    return newGuideTree


def isPair(lst):
    return isinstance(lst, list) and len(lst) == 2 and isinstance(lst[0], int) and isinstance(lst[1], int)


def filterLayer(guideTree, layer, codeMode=True, code=None, mirrorMode=True, verbose=False):
    # Single out elements that belong to the specified layer
    # Code mode: pure branches (later explained) that belong to contiguous layers are assigned code pairs
    # Mirror mode: the specified layer's symmetrical layer counterpart is considered simultaneously
    if code is None:
        code = 0
    newGuideTree = [guideTree[0]]
    addCode = codeMode and guideTree[0] == 'parallel' and len(guideTree) > 2 and guideTree[-1] != 'filler'
    if addCode:  # if need to add code pairs
        impureBranches = []  # branches that involve multiple layers
        pureBranches = {}  # a dictionary that matches branches that only involve one layer with that layer's index
        totalLayerLst = []
        bodyCode = -1
        for branch in guideTree[1:]:
            if isinstance(branch, int):
                bodyCode = branch
                continue
            layerLst = getAllLayers(branch)
            layerSet = set(layerLst)
            if len(layerSet) > 1:  # a branch that involves more than one layer is considered "impure"
                impureBranches.append(branch)
            else:  # a branch that involves only one layer is considered "pure"
                curLayer = list(layerSet)[0]
                pureBranches.setdefault(curLayer, []).append(branch)
                totalLayerLst += layerLst

        totalLayerLst = sorted(set(totalLayerLst))
        for i in range(len(totalLayerLst)):  # only care about layers that contain the pure branches
            if totalLayerLst[i] == layer or mirrorMode and totalLayerLst[i] == -layer:
                codePair = [code, code + 1]
                if i == 0 or totalLayerLst[i] - 1 != totalLayerLst[i - 1]:
                    codePair[0] = 0  # no input code
                if i == len(totalLayerLst) - 1 or totalLayerLst[i] + 1 != totalLayerLst[i + 1]:
                    codePair[1] = 0  # no output code
                for pureBranch in pureBranches[totalLayerLst[i]]:
                    soakCode(pureBranch, codePair, totalLayerLst[i])
                    newGuideTree.append(pureBranch)
                    if verbose:
                        print('soaked', pureBranch, newGuideTree, codePair, totalLayerLst, i)
            if i < len(totalLayerLst) - 1:
                code += 1  # replace with another code
        for branch in impureBranches:  # further filter those impure branches with the new code
            branch, code = filterLayer(branch, layer, codeMode=codeMode, code=code, mirrorMode=mirrorMode,
                                       verbose=verbose)
            newGuideTree.append(branch)
        if bodyCode != -1:  # add body code to the end
            newGuideTree.append(bodyCode)
    elif guideTree[0] in ['series', 'parallel', 'body', 'plan']:
        # no code pairs; just single out elements that belong to the specified layer
        for comp in guideTree[1:]:
            if comp == 'filler' or isinstance(comp, int):
                newGuideTree.append(comp)
                continue
            if comp[0] in ['series', 'parallel', 'body', 'plan']:
                comp, code = filterLayer(comp, layer, codeMode=codeMode, code=code, mirrorMode=mirrorMode,
                                         verbose=verbose)
                if verbose:
                    print("comp", comp, "code", code)
                newGuideTree.append(comp)
            elif comp[2] == layer or mirrorMode and comp[2] == -layer:  # its symmetrical counterpart also considered
                newGuideTree.append(comp)
    return newGuideTree, code


def soakCode(tree, codePair, layer):
    if tree[0] in ['series', 'parallel', 'body', 'plan']:
        for comp in tree[1:]:
            if comp == 'filler' or isinstance(comp, int):
                continue
            soakCode(comp, codePair, layer)
    else:
        if isPair(tree[-1]) and tree[-2] == layer:
            print("yello", tree, codePair)
            tree[-1] = codePair  # update the current code pair if soaked repetitively
        elif tree[-1] == layer:
            tree.append(codePair)  # attach a new code pair


def fillDynamic(guideTree):
    if guideTree[0] in ['series', 'parallel', 'body', 'plan']:
        newGuideTree = [guideTree[0]]
        for i, comp in enumerate(guideTree):
            if i == 0 or isinstance(comp, int):
                continue
            newGuideTree.append(fillDynamic(comp))
    elif guideTree[0] in charMapDyn:
        newGuideTree = ['parallel', ['series', guideTree],
                        ['series', ['j', guideTree[1], guideTree[2]]], 'filler']
    else:
        newGuideTree = guideTree
    return newGuideTree


def getAllLayers(guideTree):
    layerLst = []
    if guideTree[0] in ['series', 'parallel', 'body', 'plan']:
        for comp in guideTree[1:]:
            if comp == 'filler' or isinstance(comp, int):
                continue
            layerLst += getAllLayers(comp)
    else:
        layerLst = [guideTree[2]]
    return layerLst


def countDepth(tree, maxDepth=0):
    if tree[0] == 'series':
        maxDepth += 1
    if tree[0] == 'parallel' or tree[0] == 'series':
        pastMaxDepth = maxDepth
        for branch in tree[1:]:
            depth = countDepth(branch, maxDepth=pastMaxDepth)
            if depth > maxDepth:
                maxDepth = depth
    return maxDepth


def countBranches(tree):
    # Count the minimum requirement of space division for clearly depicting the tree
    branchNum = 0
    if tree[0] in ['series', 'parallel']:
        for branch in tree[1:]:
            if branch == 'filler' or isinstance(branch, int):
                continue
            newBranchNum = countBranches(branch)
            if newBranchNum > branchNum:
                branchNum = newBranchNum
    elif tree[0] in ['body', 'plan']:
        branchNums = []
        for branch in tree[1:]:
            if branch == 'filler' or isinstance(branch, int):
                continue
            branchNums.append(countBranches(branch))
        branchNum = np.prod(branchNums)
    else:
        return len(charMapAll[tree[0]])

    if tree[0] == 'series':
        branchNum *= len(tree[1:])
    return branchNum


def drawGuideTree(tree, usefulProbs={}, layerMode=False, verbose=False, pixelMap=None, bodyMotionPayload=None, isInit=False):
    global depthMax
    global unknownTree
    global bodyDict
    global frameWidth
    global frameHeight
    global globalPixelMap
    global tweening

    print('saizu', frameWidth, frameHeight)
    print('useful probs', usefulProbs)

    indexMap = np.arange(frameHeight * frameWidth).reshape(frameHeight, frameWidth)

    if isInit:  # Create a blank canvas for the first frame
        pixelPayload = np.full((frameHeight * frameWidth, 2), [0.0, 0])
    else:
        pixelPayload = pixelMap.reshape(frameHeight, frameWidth, 2)

    # Condition maps record conditions to produce / enable certain elements on the next layer
    # Pixels, bodies, and plans are produced, whereas motions are enabled
    # [] is made of elements like (0.0, 0)
    # first digit = expected pixel value, second digit = output code
    condPixelPayload = [[] for _ in range(frameWidth * frameHeight)]  # static conditions

    # [] is made of elements like ((0.0, 0.0), 0) or (0.0, 0)
    # (float, int) only applies to ż, m, ċ, where the first float is the motion vector magnitude
    # for (float tuple, int), tuple = expected motion vector, int = output code
    condMotionPayload = [[] for _ in range(frameWidth * frameHeight)]  # dynamic conditions

    if usefulProbs != {} and layerMode:  # select the elements driven by the inputs from the layer below
        choiceDict = {}
        for code in usefulProbs:
            choiceDict[code] = np.random.rand() < usefulProbs[code]  # roll the dice
        if verbose:
            print("Choices made", choiceDict)
        tree = testElements(tree, choiceDict)
        if verbose:
            print("Survived tree", tree)

        if len(tree) <= 1:
            tree = unknownTree

        tree = cleanGuideTree(tree, keepCode=True)
        if len(tree) <= 1:
            tree = unknownTree

        if verbose:
            print('Cleaned Tree', tree)
    print('init', isInit)

    # Generate new material at every frame for those input-driven elements
    drawFrame(tree, indexMap, pixelPayload, condPixelPayload, [0, 0], frameWidth, frameHeight, isInit=isInit)


    if layerMode:
        # Mark the regions whose patterns have to be scanned
        codeMap = {}
    else:
        codeMap = None

    # a new motion map is created in every key frame
    motionPayload = np.zeros((frameHeight * frameWidth, 2), dtype=float)

    # global movements in the frame
    planMovements(tree, indexMap, pixelPayload, motionPayload, condMotionPayload, (0, 0), frameWidth, frameHeight)

    if isinstance(bodyMotionPayload, np.ndarray):
        motionPayload += bodyMotionPayload  # bodies move the non-body pixels in turn

    pixelMaps = runMovements(indexMap, pixelPayload, motionPayload)  # include inbetweening frames

    background = np.copy(pixelMaps[-1])  # without bodies
    macroMotionMaps = [np.copy(motionPayload) for _ in range(tweening)]


    if not isInit:
        updateBodyTrees(tree, isInit)
        for tup in bodyDict.values():
            tup[1] = False  # restore the "isUpdated" booleans
    rotateJoints(tree)

    # bodies' appearance and motion are computed
    bodyPixelMaps, bodyMotionMap = moveBodies(tree, macroMotionMaps)


    if bodyPixelMaps:
        pixelMaps = [pixelMaps[i] + bodyPixelMaps[i] for i in range(tweening)]

    for i, frame in enumerate(pixelMaps):
        mask = frame[:, :, 1] == 0
        frame[:, :, 0][mask] = 127.5
        frame[:, :, 1][mask] = 1

    # Refers to the whole frame; convenient for the application of shedding elements (z and f) in bodies
    globalPixelMap[globalCurLayer] = pixelMaps[-1]


    for i, frame in enumerate(pixelMaps):
        quotient = frame[:, :, 0] / frame[:, :, 1]
        pixelMaps[i] = np.clip(quotient, 0, 255)


    if layerMode:
        # Scan the new frame and produce the outputs for the next layer
        print('weowo', tree, codeMap)
        codeMap = scanRegions(tree, pixelMaps[-1], codeMap, (0, 0), frameWidth, frameHeight)
        codeMap = {output: sum(probs) / len(probs) for output, probs in codeMap.items()}
        # codeMap = {output: 0.5 for output, probs in codeMap.items()}
        if verbose:
            print('New Code Map', codeMap, tree)

    return pixelMaps, codeMap, tree, background, bodyMotionMap


def lst2tup(lst):
    if isinstance(lst, list):
        return tuple(lst2tup(item) for item in lst)
    else:
        return lst


def tup2lst(tup):
    if isinstance(tup, tuple):
        return [tup2lst(item) for item in tup]
    else:
        return tup


def testElements(guideTree, choiceDict):
    # Only keep elements permitted by the dice results
    if guideTree[0] in ['series', 'parallel', 'body', 'plan']:
        newGuideTree = [guideTree[0]]
        for comp in guideTree[1:]:
            if comp == 'filler' or isinstance(comp, int):
                newGuideTree.append(comp)
                continue
            newGuideTree.append(testElements(comp, choiceDict))
        return newGuideTree
    else:
        if isPair(guideTree[-1]) and guideTree[-1][0] != 0:  # this element is subject to the selection procedure
            if not choiceDict[guideTree[-1][0]]:
                return unknownTree
        return guideTree.copy()  # the element survives


def cleanGuideTree(guideTree, keepCode=True):
    if not isinstance(guideTree, list):  # strings and integers do not need to be processed
        if guideTree == 'filler' or isinstance(guideTree, int):
            return guideTree
    else:
        if not guideTree:
            return []
        if isPair(guideTree[-1]) and not keepCode:
            guideTree = guideTree[:-1]
        if guideTree[0] in ['series', 'parallel', 'body', 'plan']:
            if len(guideTree) == 1:
                return []
            newGuideTree = [guideTree[0]]
            completeExists = False
            for comp in guideTree[1:]:
                result = cleanGuideTree(comp, keepCode=keepCode)
                if result == 'filler' or isinstance(result, int):
                    newGuideTree.append(result)
                    # If the fuseau only has a "filler" string, then there is not a single complete branch
                    continue
                if len(result) > 1:
                    completeExists = True
                if result:
                    if result[0] == 'series' and guideTree[0] == 'series':
                        newGuideTree += result[1:]  # if the series node has a series child, one "series" is enough
                    else:
                        newGuideTree.append(result)
            if completeExists:
                if newGuideTree[0] == 'parallel' and len(newGuideTree) == 2:
                    if newGuideTree[1] == 'filler' or isinstance(newGuideTree[1], int):
                        return []
                    else:
                        return newGuideTree[1]  # if the parallel node has a single child, "parallel" is unnecessary
                else:
                    return newGuideTree
            else:
                return []
        else:
            return guideTree.copy()


def checkDeath(body):  # As long as there is one pixel or motion in the body, including its children, it is alive
    if not np.all(body.pixelMap == 0) or not np.all(body.moveMap == 0):
        return False
    for childBody in body.jointLst + body.bodyLst:
        if not checkDeath(childBody):
            return False
    return True

def drawFrame(tree, indexMap, pixelPayload, condPixelPayload, origin, width, height, isInit):
    global bodyDict
    # Only draw the non-body parts and reap the bodies (body node and coded elements)
    if tree[0] == 'parallel':
        if tree[-1] == 'filler' or isinstance(tree[-1], int):
            tree = tree[:-1]  # ignore "filler" string
        for branch in tree[1:]:
            drawFrame(branch, indexMap, pixelPayload, condPixelPayload, origin, width, height, isInit)
    elif tree[0] == 'series':
        division = width / len(tree[1:])  # horizontal = body forward
        for i, comp in enumerate(tree[1:]):
            drawFrame(comp, indexMap, pixelPayload, condPixelPayload, [origin[0] + round(division * i), origin[1]],
                                 round(division * (i + 1)) - round(division * i), height, isInit)
    elif tree[0] in ['body', 'plan']: # and tree[-1] not in bodyDict:
        print("let's create a new body", tree)
        bodyDict.setdefault(tree[-1], [[], False])[0].append(
            Body(tree, origin, width, height, tree[0] == 'plan', isInit))  # record a new body

        if checkDeath(bodyDict[tree[-1]][0][-1]):  # if the new body has nothing inside, it is destroyed immediately
            print('destroyed!!')
            bodyDict[tree[-1]][0] = bodyDict[tree[-1]][0][:-1]
    elif tree[0] in charMap:
        print('treeeee', tree)

        # Update the condition pixel map
        if isPair(tree[-1]) and tree[-1][1] != 0:
            addCondHex(tree, indexMap, condPixelPayload, origin, width, height, tree[-1][1])

        # Only input-driven or initial elements can be infinitely produced
        if isPair(tree[-1]) and tree[-1][0] != 0 or isInit:
            addHex(tree, indexMap, pixelPayload, origin, width, height)


def updateBodyTrees(guideTree, isInit):
    global bodyDict
    if guideTree[0] in ['series', 'parallel']:
        for comp in guideTree[1:]:
            if comp == 'filler':
                continue
            updateBodyTrees(comp, isInit)
    elif guideTree[0] in ['body', 'plan'] and isinstance(guideTree[-1], int):
        for body in bodyDict[guideTree[-1]][0]:
            body.update(guideTree, isInit)  # stop at the surface and let the bodies do the rest


def rotateJoints(guideTree):
    global bodyDict
    if guideTree[0] in ['series', 'parallel']:
        for comp in guideTree[1:]:
            if comp == 'filler':
                continue
            rotateJoints(comp)
    elif guideTree[0] in ['body', 'plan'] and isinstance(guideTree[-1], int):
        for body in bodyDict[guideTree[-1]][0]:
            body.rotate()  # stop at the surface and let the bodies do the rest


def moveBodies(guideTree, macroMotionMaps):
    global bodyDict
    global tweening

    framePixelMapSeqs = []
    frameMotionMaps = []
    if guideTree[0] in ['series', 'parallel']:
        for comp in guideTree[1:]:
            if comp == 'filler':
                continue
            result = moveBodies(comp, macroMotionMaps)
            if result != ([], []):
                framePixelMapSeqs.append(result[0])
                frameMotionMaps.append(result[1])
    elif guideTree[0] in ['body', 'plan'] and isinstance(guideTree[-1], int):
        for body in bodyDict[guideTree[-1]][0]:
            body.moveBody(macroMotionMaps, isGlobal=True)  # stop at the surface and let the bodies do the rest
            framePixelMaps = []
            frameMotionMaps = []
            # Take the original frame maps out of the accommodating maps
            for i in range(tweening):
                framePixelMaps.append(body.accomPixelMaps[i][body.accomOrigin[1]:body.accomOrigin[1] + frameHeight,
                                                             body.accomOrigin[0]:body.accomOrigin[0] + frameWidth])

            framePixelMapSeqs.append(framePixelMaps)
            frameMotionMaps.append(body.prevMotionMaps[0][body.accomOrigin[1]:body.accomOrigin[1] + frameHeight,
                                                      body.accomOrigin[0]:body.accomOrigin[0] + frameWidth])

    if not framePixelMapSeqs and not frameMotionMaps:
        return [], []
    if len(framePixelMapSeqs) > 1:
        framePixelMaps = [sum(pair) for pair in zip(*framePixelMapSeqs)]
    elif len(framePixelMapSeqs) == 1:
        framePixelMaps = framePixelMapSeqs[0]

    if len(frameMotionMaps) > 1:
        frameMotionMap = sum(frameMotionMaps)
    elif len(frameMotionMaps) == 1:
        frameMotionMap = frameMotionMaps[0]
    return framePixelMaps, frameMotionMap


def overlayMap(bigMap, smallMap, originX, originY):
    # Determine the size of the big and small matrices
    bigH, bigW = bigMap.shape[:2]
    smallH, smallW = smallMap.shape[:2]

    # Calculate the start and end indices in both matrices
    startX = max(originX, 0)
    startY = max(originY, 0)
    endX = min(originX + smallW, bigW)
    endY = min(originY + smallH, bigH)

    if startX >= endX or startY >= endY:
        return  # the small matrix falls outside the big matrix

    smallStartX = max(-originX, 0)
    smallStartY = max(-originY, 0)
    smallEndX = smallStartX + (endX - startX)
    smallEndY = smallStartY + (endY - startY)

    # Add the overlapping region
    bigMap[startY:endY, startX:endX] += smallMap[smallStartY:smallEndY, smallStartX:smallEndX]


def coord2Angle(x, y):
    # Calculates the angle (in radians) from the positive x-axis to the point (x, y).
    return math.atan2(-y, x)


def angle2Coord(angle):
    # Finds the point whose line connected to the origin forms a certain angle with the x-axis and has a length of 1
    return math.cos(angle), -math.sin(angle)


def tree2Angles(guideTree, add=0):
    if guideTree[0] in ['series', 'parallel', 'body', 'plan']:
        newGuideTree = [guideTree[0]]
        for comp in guideTree[1:]:
            if comp == 'filler' or isinstance(comp, int):
                newGuideTree.append(comp)
                continue
            newGuideTree.append(tree2Angles(comp))
        return newGuideTree
    else:
        newGuideTree = guideTree.copy()

        if guideTree[1] == 0:
            newGuideTree[1] = 0.0
        elif guideTree[1] == 1:
            newGuideTree[1] = 270.0
        if add != 0:
            newGuideTree[1] += add
            newGuideTree[1] %= 360
        return newGuideTree


def orient2Angle(orient):
    return 270.0 if orient == 1 else 0.0


def vec2Angle(vec, orient):
    needFlip = 1 if vec == -1 else 0
    return (orient * math.pi * 3 / 2 + math.pi * needFlip) % (math.pi * 2)


def cleanTiny(map, threshold=1e-15):
    # Set extremely small values in a matrix to 0
    map[np.abs(map) < threshold] = 0


def rotateMap(oldIndexMap, angle, isPixel=True):
    # Rotate the temporary maps and cover them on the original maps
    if angle == 0.:
        return oldIndexMap

    angleDeg = math.degrees(angle)
    if isPixel:
        # Rotate the map according to the specified vector direction
        # Fill the background with nan, which will be deleted if it becomes unnecessary borders
        rotatedMap = scipy.ndimage.rotate(oldIndexMap, angleDeg, reshape=True,
                                          order=1, mode='constant')
    else:
        rotatedMap = scipy.ndimage.rotate(oldIndexMap, angleDeg, reshape=True,
                                          order=1, mode='constant')

        def addAngle(coord):
            oldAngle = coord2Angle(coord[1], coord[0])
            newX, newY = angle2Coord(oldAngle + angle)
            return newY, newX

        maskZeros = (rotatedMap == [0.0, 0.0]).all(axis=2)
        validMask = ~maskZeros
        if np.any(validMask):
            rotatedMap[validMask] = np.apply_along_axis(addAngle, 1, rotatedMap[validMask])
    # Put back onto the original map
    return rotatedMap


def locateBoundRect(matrix, mode=0):
    # mode: 0 = pixel, 1 = movements, 2 = boolean
    if mode == 0:
        nonZeroIndices = np.argwhere(matrix[:, :, 1] != 0)
    elif mode == 1:
        nonZeroIndices = np.argwhere(~(matrix == 0).all(axis=2))
    elif mode == 2:
        nonZeroIndices = np.argwhere(matrix[:, :] == True)

    if nonZeroIndices.size == 0:
        return np.iinfo(np.int64).max, np.iinfo(np.int64).max, -1, -1  # No non-zero elements found

    # Find the bounding rectangle for all non-zero elements
    minRow, minCol = nonZeroIndices.min(axis=0)
    maxRow, maxCol = nonZeroIndices.max(axis=0)

    return minRow, minCol, maxRow + 1, maxCol + 1


def labelBodies(tree, bodyCode=0, inBody=False, treeRest=[]):
    if tree[0] in ['series', 'parallel', 'body', 'plan']:
        labeledTree = [tree[0]]
        childInBody = inBody
        if tree[0] in ['body', 'plan']:  # you are entering body
            inBody = True
            childInBody = True
        elif tree[0] == 'parallel' and inBody:  # sub-joint parallel nodes cannot be joints unless through bodies
            inBody = True
            childInBody = False

        for i, branch in enumerate(tree[1:]):
            if tree[0] in ['body', 'plan']:
                treeRest = tree[i + 1:] + treeRest
                if not treeRest:  # skip base-level elements, unless there are bodies
                    childInBody = False

            newBranch, bodyCode = labelBodies(branch, bodyCode, childInBody, treeRest)
            labeledTree.append(newBranch)
    else:
        return tree, bodyCode

    # In-body parallel nodes are joints (except for sub-joint parallel nodes)
    # We don't record body branches in the body dict
    if tree[0] in ['body', 'plan'] or tree[0] == 'parallel' and tree[-1] != 'filler' and inBody:
        labeledTree.append(bodyCode)
        bodyCode += 1
    return labeledTree, bodyCode


def fillGray(pixelMap):
    mask = pixelMap[:, :, 1] == 0
    pixelMap[mask, 0] += 127.5
    pixelMap[mask, 1] += 1


class Body:
    global bodyDict
    global frameWidth
    global frameHeight
    global tweening

    def __init__(self, tree, origin, width, height, isPlan, isInit):
        self.tree, self.origin = tree, origin
        self.isPlan = isPlan  # plan means a static organization of movements
        # self.tree = tree2Angles(self.tree)  # Turn the accents into angles for each element in the tree
        self.width, self.height = width, height
        self.bodyLst, self.jointLst = [], []
        self.nextBranch = []
        # Pixel map itself is not rotated, but it will appear rotated after applying the angle for display
        self.pixelMap = np.full((height, width, 2), [0.0, 0])
        # self.pixelMap = np.full((height, width, 2), [127.5, 1])
        # Move map is computed after pixel map is computed
        self.motionMap = np.zeros((height, width, 2), dtype=float)
        # Maps and parameters that take account of the children joints and bodies
        self.pixelMaps, self.motionMaps, self.accomEmptyMaps, self.origins, self.widths, self.heights = None, None, None, None, None, None
        # Accom size based on frame size
        self.accomSize, self.accomOrigin = computeAccom(frameWidth, frameHeight)
        # Pixel maps and move maps placed within a larger frame
        self.accomPixelMaps = [np.full(self.accomSize, [0.0, 0]) for _ in range(tweening)]
        self.accomMotionMaps = [np.zeros(self.accomSize, dtype=float) for _ in range(tweening)]
        # For preventing the body's motion - applied to outside - from coming back to its macro-movements
        # Its size is equal to frame size
        self.prevMotionMaps = None
        # The sum of the child bodies' previous movement maps
        self.childMotionMap = None
        # Current angle and current angle in the global frame
        self.angle, self.globalAngle = 0, 0
        # Current angles in hold that belong to a joint
        self.angleLst = []
        # Center of the bounding box in the beginning (useful for rotation)
        self.centerInit = (self.origin[0] + self.width // 2, self.origin[1] + self.height // 2)
        # Origin only changeable through macro-movements
        self.originMacro = [0, 0]
        # How many frames have the body been present
        self.age = 0

        if self.tree[0] in ['body', 'plan']:  # ['body', body branch 1, body branch 2, ...]
            self.updateHelper(self.tree[1], self.pixelMap, self.motionMap, [0, 0], self.width, self.height,
                              self.tree[2:-1],  # cut the body code
                              False, isInit)
        elif self.tree[0][0] == 'parallel':  # [['parallel', ...], body branch 2, body branch 3, ...]
            for branch in self.tree[0][1:]:
                if branch == 'filler' or isinstance(branch, int):  # cut the joint code
                    continue
                self.updateHelper(branch, self.pixelMap, self.motionMap, [0, 0], self.width, self.height,
                                  self.tree[1:], False, isInit)

        emptyMask = self.pixelMap[:, :, 1] == 0
        self.emptyMap = np.full((height, width, 2), [0.0, 0])
        self.emptyMap[:, :, 1][emptyMask] = 1

    def update(self, tree, isInit):
        self.tree = tree
        if self.tree[0] in ['body', 'plan']:  # ['body', body branch 1, body branch 2, ...]
            self.updateHelper(self.tree[1], self.pixelMap, self.motionMap, [0, 0], self.width, self.height,
                              self.tree[2:-1],  # cut the body code
                              True, isInit)
        elif self.tree[0][0] == 'parallel':  # [['parallel', ...], body branch 2, body branch 3, ...]
            # Decompose the parallel node because updateHelper() confuses it with another joint
            for branch in self.tree[0][1:]:
                if branch == 'filler' or isinstance(branch, int):  # cut the joint code
                    continue
                self.updateHelper(branch, self.pixelMap, self.motionMap, [0, 0], self.width, self.height,
                                  self.tree[1:], True, isInit)

        # Air that surrounds the bodies
        emptyMask = self.pixelMap[:, :, 1] == 0
        self.emptyMap = np.full((self.height, self.width, 2), [0.0, 0])
        self.emptyMap[:, :, 1][emptyMask] = 1

    def rotate(self, parentAngle=0):
        if self.tree[0][0] == 'parallel':
            # Upper-level dynamic elements rotate; Base-level dynamic elements move
            if self.angleLst:
                print('\tpast angle', self.angle / math.pi * 180, 'angle list', [ang / math.pi * 180 for ang in self.angleLst], 'tree', self.tree)
                self.angle += sum(self.angleLst) / len(self.angleLst)
                self.angle %= 2 * math.pi  # prevent from exceeding 360 degrees
                print('\tnow angle', self.angle / math.pi * 180)
                self.angleLst = []  # reset angle list immediately after using it
                self.globalAngle = parentAngle + self.angle
        for body in self.jointLst + self.bodyLst:
            print('\t\tenter body', body.tree, body.jointLst, body.bodyLst)
            body.rotate(self.angle)

    def propagateRecall(self):
        # Sum up all children's movement maps
        self.childMotionMap = np.zeros(self.accomSize, dtype=float)
        for body in self.bodyLst + self.jointLst:
            if body.prevMotionMaps is None:
                continue
            self.childMotionMap += body.prevMotionMaps[-1]


    def moveInternal(self, macroMotionMaps):
        # Compute the movements of the maps of the self that are yet to be rotated
        self.accomPixelMaps = [np.full(self.accomSize, [0.0, 0]) for _ in range(tweening)]
        self.accomMotionMaps = [np.zeros(self.accomSize, dtype=float) for _ in range(tweening)]


        cleanTiny(self.pixelMap)
        cleanTiny(self.motionMap)

        cleanTiny(self.pixelMap)
        cleanTiny(self.motionMap)

        print('intern', self.width, self.height, self.origin, self.accomOrigin, self.pixelMap.shape, self.accomPixelMaps[0].shape, self.tree, len(self.jointLst))

        overlayMap(self.accomPixelMaps[0], self.pixelMap, self.accomOrigin[0] + self.origin[0],
                   self.accomOrigin[1] + self.origin[1])
        overlayMap(self.accomMotionMaps[0], self.motionMap, self.accomOrigin[0] + self.origin[0],
                   self.accomOrigin[1] + self.origin[1])
        '''
        self.accomPixelMaps[0][self.accomOrigin[1] + self.origin[1]:self.accomOrigin[1] + self.origin[1] + self.height,
                               self.accomOrigin[0] + self.origin[0]:self.accomOrigin[0] + self.origin[0] + self.width] += self.pixelMap
        self.accomMotionMaps[0][self.accomOrigin[1] + self.origin[1]:self.accomOrigin[1] + self.origin[1] + self.height,
                               self.accomOrigin[0] + self.origin[0]:self.accomOrigin[0] + self.origin[0] + self.width] += self.motionMap
        '''

        accomEmptyMap = np.zeros(self.accomSize, dtype=float)
        overlayMap(accomEmptyMap, self.emptyMap, self.accomOrigin[0] + self.origin[0], self.accomOrigin[1] + self.origin[1])

        # Movements of local pixels
        self.accomPixelMaps, self.accomMotionMaps, childMotionMaps, self.accomEmptyMaps = runMovements(self.accomPixelMaps[0], self.accomMotionMaps[0], self.childMotionMap, accomEmptyMap, isBody=True, isPlan=self.isPlan)
        # Recalibrate origin and size after local movements
        self.recalibrate(updatingOrigin=True)


        childMacroMotionMaps = []

        if self.bodyLst and any([not b.isPlan for b in self.bodyLst]):
            for i in range(tweening):
                childMacroMotionMaps.append(np.zeros((frameHeight, frameWidth, 2), dtype=float))
                if self.isPlan:
                    overlayMap(childMacroMotionMaps[-1], macroMotionMaps[-1], 0, 0)
                else:
                    overlayMap(childMacroMotionMaps[-1], self.accomMotionMaps[i], -self.accomOrigin[0],
                                                                              -self.accomOrigin[1])
                    if childMotionMaps:
                        overlayMap(childMacroMotionMaps[-1], childMotionMaps[i], -self.accomOrigin[0],
                                                                            -self.accomOrigin[1])

        # Movements of child bodies and rotation
        for body in self.bodyLst:
            if body.isPlan:
                body.moveBody([])
            else:
                body.moveBody(childMacroMotionMaps)
            for i in range(tweening):
                self.accomPixelMaps[i] += body.accomPixelMaps[i]
                self.accomMotionMaps[i] += body.accomMotionMaps[i]


        # Internal movements of child joints and rotation
        for joint in self.jointLst:
            joint.moveInternal(macroMotionMaps)
            joint.propagateOverlay(self.accomPixelMaps, self.accomMotionMaps, self.accomOrigin)
            print('ende')

        if (self.bodyLst + self.jointLst) or True:
            # The movements of children bodies are applied back to the parent body
            self.propagateRecall()

            # Recalibrate origin and size after movements of child bodies
            self.recalibrate(updatingOrigin=False)

        # Becomes older after each internal update
        self.age += 1


    def moveBody(self, macroMotionMaps, isGlobal=False):
        # Self movements and rotation + joint movements and rotation
        self.moveInternal(macroMotionMaps)

        for i in range(tweening):
            self.accomPixelMaps[i][:, :] = [0.0, 0]
            self.accomMotionMaps[i][:, :] = [0, 0]

        self.propagateOverlay(self.accomPixelMaps, self.accomMotionMaps, self.accomOrigin)

        if not self.isPlan:
            # Movements of the entire body
            newMacroMotionMaps = []
            for i, m in enumerate(macroMotionMaps):
                newMacroMotionMaps.append(np.zeros(self.accomSize, dtype=float))
                if self.prevMotionMaps is None:
                    overlayMap(newMacroMotionMaps[-1], m, *self.accomOrigin)
                else:  # counteract the part of macro-movements that come from the body's own movements
                    if isGlobal or True:
                        overlayMap(newMacroMotionMaps[-1], m - self.prevMotionMaps[i][self.accomOrigin[1]:self.accomOrigin[1] + frameHeight,
                                             self.accomOrigin[0]:self.accomOrigin[0] + frameWidth], *self.accomOrigin)
                    else:
                        overlayMap(newMacroMotionMaps[-1], m -
                                   self.accomMotionMaps[i][self.accomOrigin[1]:self.accomOrigin[1] + frameHeight,
                                             self.accomOrigin[0]:self.accomOrigin[0] + frameWidth], *self.accomOrigin)

            print('eecc', np.sum(newMacroMotionMaps[-1][:,:,0]), self.tree)
            self.accomPixelMaps, self.accomMotionMaps = self.runMacroMovements(newMacroMotionMaps, self.accomSize)

        # Body saves its previous movement map
        self.prevMotionMaps = self.accomMotionMaps

    def recalibrate(self, updatingOrigin=True):
        # Recalibrate the body's origin and size by cropping the blank margins of its body maps
        # These sequences are "impure" - they contain pixels of the child bodies
        self.pixelMaps, self.motionMaps, self.origins, self.widths, self.heights = [], [], [], [], []

        for curAccomPixelMap, curAccomMotionMap, curAccomEmptyMap, in zip(self.accomPixelMaps, self.accomMotionMaps, self.accomEmptyMaps):
            staticRect = locateBoundRect(curAccomPixelMap, mode=0)
            dynamicRect = locateBoundRect(curAccomMotionMap, mode=1)
            emptyRect = locateBoundRect(curAccomEmptyMap, mode=0)
            if staticRect != (np.iinfo(np.int64).max, np.iinfo(np.int64).max, -1, -1) or \
                    dynamicRect != (np.iinfo(np.int64).max, np.iinfo(np.int64).max, -1, -1) or \
                    emptyRect != (np.iinfo(np.int64).max, np.iinfo(np.int64).max, -1, -1):
                tups = list(zip(staticRect, dynamicRect, emptyRect))
                print('tups', tups)

                # Parameters obtained from cropping the body maps
                cutOriginY, cutOriginX, cutEndY, cutEndX = min(tups[0]), min(tups[1]), max(tups[2]), max(tups[3])

                # o' = o + Δo = o + [c - (a + o)] = c - a
                self.origins.append([cutOriginX - self.accomOrigin[0], cutOriginY - self.accomOrigin[1]])
                print('cut', cutOriginX, cutOriginY, cutEndX, cutEndY, curAccomPixelMap.shape, curAccomMotionMap.shape, self.origins[-1], self.tree)


                # Cut out the body maps
                self.pixelMaps.append(np.copy(curAccomPixelMap[cutOriginY:cutEndY, cutOriginX:cutEndX]))


                self.motionMaps.append(np.copy(curAccomMotionMap[cutOriginY:cutEndY, cutOriginX:cutEndX]))
                print('new shapes', self.pixelMaps[-1].shape, self.motionMaps[-1].shape, updatingOrigin)
                self.widths.append(cutEndX - cutOriginX)
                self.heights.append(cutEndY - cutOriginY)

            else:
                # Append empty maps if not succeeded
                self.pixelMaps.append(np.full((self.height, self.width, 2), [0.0, 0]))
                self.motionMaps.append(np.zeros((self.height, self.width, 2), dtype=float))
                self.origins.append(self.origin)
                self.widths.append(self.width)
                self.heights.append(self.height)
        # self.pixelMap, self.motionMap = self.pixelMaps[-1], self.motionMaps[-1]
        if updatingOrigin:
            self.origin, self.width, self.height = self.origins[-1], self.widths[-1], self.heights[-1]
            self.pixelMap = self.pixelMaps[-1]
            self.motionMap = self.motionMaps[-1]


    def propagateOverlay(self, accomPixelMaps, accomMotionMaps, accomOrigin):
        # Overlay the rotated pixel maps and move maps of every body and joint

        if self.tree[0][0] == 'parallel':  # Joint saves its previous movement map
            self.prevMotionMaps = [np.zeros(accomMotionMaps[0].shape, dtype=float) for _ in range(tweening)]

        for i in range(tweening):
            rotatedMap = rotateMap(self.pixelMaps[i], self.angle)

            # The vector from the initial bounding box center (assuming without macro-movements) to the current center
            a = self.origins[i][0] + self.widths[i] // 2 - self.originMacro[0] - self.centerInit[0]
            b = -(self.origins[i][1] + self.heights[i] // 2 - self.originMacro[1] - self.centerInit[1])
            deltaX = a * math.cos(self.angle) - b * math.sin(self.angle)
            deltaY = -(a * math.sin(self.angle) + b * math.cos(self.angle))
            # Align the rotated bounding box center to initial center
            newOrigin = (self.centerInit[0] - rotatedMap.shape[1] // 2 + round(deltaX) + accomOrigin[0] + self.originMacro[0],
                         self.centerInit[1] - rotatedMap.shape[0] // 2 + round(deltaY) + accomOrigin[1] + self.originMacro[1])

            print('report', self.tree)
            print('accomOrigin', accomOrigin, 'newOrigin', newOrigin, 'origins[i]', self.origins[i], 'originMacro', self.originMacro, 'origin', self.origin)
            print('a', a, 'b', b, 'deltaX', deltaX, 'deltaY', deltaY, 'angle', self.angle / math.pi * 180, 'centerInit', self.centerInit, 'widths', self.widths[i], 'heights', self.heights[i], 'rotated shape', rotatedMap.shape)
            overlayMap(accomPixelMaps[i], rotatedMap, *newOrigin)

            rotatedMap = rotateMap(self.motionMaps[i], self.angle, isPixel=False)
            overlayMap(accomMotionMaps[i], rotatedMap, *newOrigin)

            if self.tree[0][0] == 'parallel':  # Joint saves its previous movement map
                overlayMap(self.prevMotionMaps[i], rotatedMap, newOrigin[0], newOrigin[1])




    def runMacroMovements(self, macroMotionMaps, accomSize):
        allEmpty = True
        for curMacroMotionMap in macroMotionMaps:
            if not np.all(curMacroMotionMap == 0):
                allEmpty = False
        if allEmpty:
            return self.accomPixelMaps, self.accomMotionMaps
        height, width = macroMotionMaps[-1].shape[:2]
        curX, curY = self.origin
        tentX, tentY = curX, curY

        newPixelMaps, newMotionMaps = [], []
        initPixelMap = self.accomPixelMaps[0]
        initMacroMotionMap = macroMotionMaps[-1]
        dirY, dirX = avgMovements(initPixelMap, initMacroMotionMap)
        initSignY, initSignX = getSign(dirY), getSign(dirX)
        print('init sign', initSignX, initSignY)

        isInBounds = True
        isInit = True
        existsInit = True
        while existsInit and isInBounds:
            if isInit:
                isInit = False
            else:
                curY = tentY
                curX = tentX
                newPixelMap = np.full(self.accomPixelMaps[0].shape, [0.0, 0])
                overlayMap(newPixelMap, initPixelMap, round(curX - self.origin[0]), round(curY - self.origin[1]))
                existsInit = existsMovement(newPixelMap, initMacroMotionMap, initSignX, initSignY)

            if dirY == 0 and dirX == 0:
                break
            tentY = curY + dirY  # tentative coordinates
            tentX = curX + dirX


            isInBounds = -accomSize[1] / 4 <= tentY < height + accomSize[1] / 4 \
                      and -accomSize[0] / 4 <= tentX < width + accomSize[1] / 4  # inside the frame boundary (or not?)


        for frameInd in range(tweening):
            midY = round((curY - self.origin[1]) * (frameInd + 1) / tweening)
            midX = round((curX - self.origin[0]) * (frameInd + 1) / tweening)
            newPixelMap = np.full(self.accomPixelMaps[0].shape, [0.0, 0])
            overlayMap(newPixelMap, self.accomPixelMaps[frameInd], midX, midY)
            newPixelMaps.append(newPixelMap)
            newMotionMap = np.zeros(self.accomMotionMaps[0].shape, dtype=float)
            overlayMap(newMotionMap, self.accomMotionMaps[frameInd], midX, midY)
            newMotionMaps.append(newMotionMap)


        self.propagateMacro(round(curX - self.origin[0]), round(curY - self.origin[1]))

        self.origin = [round(curX), round(curY)]  # Recalibrate origin after macro-movements

        return newPixelMaps, newMotionMaps

    def propagateMacro(self, deltaX, deltaY):
        for body in self.jointLst + self.bodyLst:
            # Joints and bodies move with their parent body
            body.originMacro[0] += deltaX
            body.originMacro[1] += deltaY
            body.origin[0] += deltaX
            body.origin[1] += deltaY
            print('new macro', body.origin, body.tree)
            body.propagateMacro(deltaX, deltaY)


    def updateHelper(self, tree, pixelMap, motionMap, origin, width, height, treeRest, isUpdating, isInit):
        # isUpdating: replacing trees with the new version, isInit: drawing the first frame
        global bodyDict
        global globalPixelMap
        global globalCurLayer

        # Find all bodies and joints that link to itself
        if tree[0] == 'series':
            division = width / len(tree[1:])  # horizontal = body forward
            for i, comp in enumerate(tree[1:]):
                self.updateHelper(comp, pixelMap, motionMap, [origin[0] + round(division * i), origin[1]],
                                  round(division * (i + 1)) - round(division * i), height, treeRest, isUpdating, isInit)
        elif tree[0] == 'parallel':
            if len(treeRest) == 0 or not isinstance(tree[-1], int):
                # Base-level or sub-joint parallel nodes are not joints
                for branch in tree[1:]:
                    if branch == 'filler' or isinstance(branch, int):
                        continue
                    self.updateHelper(branch, pixelMap, motionMap, origin, width, height, treeRest, isUpdating, isInit)
            elif not isUpdating:  # initializing joints
                # Add a child joint that still preserves the deeper embedding levels
                childJoint = Body([tree] + treeRest, [self.origin[0] + origin[0], self.origin[1] + origin[1]],
                                  width, height, False, isInit)
                self.jointLst.append(childJoint)  # second boolean: on the base level
                bodyDict.setdefault(tree[-1], [[], False])[0].append(childJoint)
            elif not bodyDict[tree[-1]][1]:  # second boolean: already updated the whole body "species"
                # Update every joint of the same species that belongs to a body
                for body in bodyDict[tree[-1]][0]:
                    body.update([tree] + treeRest, isInit)
                bodyDict[tree[-1]][1] = True
        elif tree[0] in ['body', 'plan']:
            if not isUpdating:
                # Add a child body that still preserves the deeper embedding levels
                childBody = Body(tree[:-1] + treeRest + [tree[-1]], [self.origin[0] + origin[0], self.origin[1] + origin[1]],
                                 width, height, tree[0] == 'plan', isInit)
                self.bodyLst.append(childBody)
                bodyDict.setdefault(tree[-1], [[], False])[0].append(childBody)
            else:
                if not bodyDict[tree[-1]][1]:  # second boolean: already updated the whole body "species"
                    for body in bodyDict[tree[-1]][0]:
                        body.update(tree[:-1] + treeRest + [tree[-1]], isInit)
                    bodyDict[tree[-1]][1] = True
        else:
            # Source = initially every element, then conditioned input-driven elements
            isSource = isPair(tree[-1]) and tree[-1][0] != 0 or isInit

            # Condition = elements with output code that conditionalizes elements on the next layer
            isCondition = isPair(tree[-1]) and tree[-1][1] != 0

            # Only input-driven elements are productive. They act like water sources.
            # Elements without an input are neutral. They are normally passable.
            # On the other hand, elements with a non-driving input acts as obstacles to the water flow.
            # When the flow ends at base-level elements, it produces observable effect.
            if tree[0] in charMap:
                if len(treeRest) > 0:  # More embedding levels deeper down to go!
                    if tree[0] == 's':
                        if tree[1] == 0:
                            self.updateHelper(treeRest[0], pixelMap, motionMap,
                                              [origin[0], origin[1]],
                                              round(width / 3 * 2), height, treeRest[1:],
                                              isUpdating, isInit)
                        elif tree[1] == 1:
                            self.updateHelper(treeRest[0], pixelMap, motionMap,
                                              [origin[0], origin[1]],
                                              width, round(height / 3 * 2), treeRest[1:],
                                              isUpdating, isInit)
                        return
                    elif tree[0] == 'v':  # two thirds is a whole
                        if tree[1] == 0:
                            self.updateHelper(treeRest[0], pixelMap, motionMap,
                                              [origin[0] + round(width / 3), origin[1]],
                                              round(width / 3 * 2), height, treeRest[1:],
                                              isUpdating, isInit)
                        elif tree[1] == 1:
                            self.updateHelper(treeRest[0], pixelMap, motionMap,
                                              [origin[0], origin[1] + round(height / 3)],
                                              width, round(height / 3 * 2), treeRest[1:],
                                              isUpdating, isInit)
                        return
                    else:
                        n_gram = len(charMap[tree[0]])  # how many divisions (bigram or trigram)
                    if tree[1] == 0:
                        division = width / n_gram
                        for i in range(n_gram):
                            if charMap[tree[0]][i] == 1:
                                self.updateHelper(treeRest[0], pixelMap, motionMap,
                                                  [origin[0] + round(division * i), origin[1]],
                                                  round(division * (i + 1)) - round(division * i), height, treeRest[1:],
                                                  isUpdating, isInit)
                    elif tree[1] == 1:
                        division = height / n_gram
                        for i in range(n_gram):
                            if charMap[tree[0]][i] == 1:
                                self.updateHelper(treeRest[0], pixelMap, motionMap,
                                                  [origin[0], origin[1] + round(division * i)],
                                                  width, round(division * (i + 1)) - round(division * i), treeRest[1:],
                                                  isUpdating, isInit)
                else:  # we reached the base level, so let's fill it with pixels
                    if isCondition:
                        addCondHex(tree, self.indexMap, self.condPixelPayload, origin, width, height, tree[-1][1])
                    if isSource:
                        addHex(tree, self.indexMap, self.pixelPayload, origin, width, height)

            elif tree[0] in charMapDyn and isSource or tree[0] in ['z', 'f'] and self.age == 1:
                # Since both base-level and upper elements have observable effects, we need to check if they are sources
                if len(treeRest) > 0:
                    orientSum = [0, 0]
                    # Vector sums of motion -> overall orientation
                    if tree[1] == 0:  # horizontal
                        orientSum[0] = sum(vecMap[reversedHexMap[charMapDyn[tree[0]]]][0])
                    elif tree[1] == 1:  # vertical
                        orientSum[1] = sum(vecMap[reversedHexMap[charMapDyn[tree[0]]]][0])
                    self.angleLst.append(coord2Angle(*orientSum))
                else:  # let's fill it with movement vectors
                    if tree[0] in ['z', 'f']:
                        if not isInit:
                            mask = np.full((height, width), True)
                            rotatedMask = rotateMap(mask, self.globalAngle)
                            rotatedMask = np.repeat(rotatedMask[:, :, np.newaxis], 2, axis=2)

                            print('originee', origin, height, width, rotatedMask.shape, tree)
                            newOrigin = (round(self.origin[0] - (rotatedMask.shape[0] - width) / 2),
                                        round(self.origin[1] - (rotatedMask.shape[1] - height) / 2))
                            # The view of global map allowed for shedding
                            viewPixelMap = np.full(rotatedMask.shape, (0.0, 0))
                            overlayMap(viewPixelMap, globalPixelMap[globalCurLayer], *newOrigin)
                            viewPixelMap = np.where(rotatedMask, viewPixelMap, [0, 0])
                            shedMap = np.zeros(viewPixelMap.shape, dtype=float)
                            planMovements(tree, viewPixelMap, shedMap, origin, viewPixelMap.shape[1], viewPixelMap.shape[0], self.globalAngle)
                            rotatedShedMap = rotateMap(shedMap, -self.globalAngle, isPixel=False)
                            rotatedMask = rotatedMask[:, :, 0]
                            rotatedMask = rotateMap(rotatedMask, -self.globalAngle)
                            cutOriginY, cutOriginX, cutEndY, cutEndX = locateBoundRect(rotatedMask, mode=2)
                            overlayMap(self.motionMap, rotatedShedMap[cutOriginY:cutEndY, cutOriginX:cutEndX], *origin)
                    else:
                        planMovements(tree, self.pixelMap, self.motionMap, origin, width, height)


def applyHex(tree, indexMap, origin, width, height, updateFn):
    orient = tree[1]

    if tree[0] == 'n':  # n acts as transparent element
        return

    values = charMap[tree[0]]
    n_gram = len(values)

    if orient == 1:  # vertical
        division = height / n_gram
        for i, val in enumerate(values):
            region = indexMap[
                origin[1] + round(division * i): origin[1] + round(division * (i + 1)),
                origin[0]: origin[0] + width
            ]
            for idx in region.flat:
                updateFn(idx, val)
    else:  # horizontal
        division = width / n_gram
        for i, val in enumerate(values):
            region = indexMap[
                origin[1]: origin[1] + height,
                origin[0] + round(division * i): origin[0] + round(division * (i + 1))
            ]
            for idx in region.flat:
                updateFn(idx, val)

def addHex(tree, indexMap, pixelPayload, origin, width, height):
    def update(idx, val):
        pixelPayload[idx, 0] += 255 * val
        pixelPayload[idx, 1] += 1

    applyHex(tree, indexMap, origin, width, height, update)

def addCondHex(tree, indexMap, condPixelPayload, origin, width, height, outputCode):
    def update(idx, val):
        condPixelPayload[idx].append((255 * val, outputCode))

    applyHex(tree, indexMap, origin, width, height, update)


def computeAccom(width, height):
    accomSize = width + height
    accomSize += 1 if accomSize % 2 == 1 else 0

    # The smallest size that accommodates all types of rotations and movements at the boundary
    accomSize = [accomSize, accomSize, 2]

    # The origin of the frame map to be placed on the accommodating map
    accomOrigin = [height // 2, width // 2]

    if height * 3 > accomSize[0]:
        accomSize[0] = height * 3
        accomOrigin[1] = height
    if width * 3 > accomSize[1]:
        accomSize[1] = width * 3
        accomOrigin[0] = width

    return accomSize, accomOrigin

def getStages(angle):
    angle = int(angle)
    stages = [1, 2, 3, 4, 1, 2, 3][angle // 90:angle // 90 + 2]
    if angle % 90 == 0:
        stages = [stages[0]]
    return stages

def applyOblique(tree, indexMap, pixelPayload, motionPayload, origin, width, height, updateFn, requiredLayers=2):
    # Detect the first line of pixels cut from a given angle
    # requiredLayers means the number of layers that need to be shed in one round

    angleDeg = orient2Angle(tree[1])
    rad = math.radians(angleDeg)
    stepX, stepY = angle2Coord(rad - math.pi / 2)
    moveX, moveY = angle2Coord(rad) if tree[0] == 'z' else angle2Coord(math.pi + rad)
    stages = getStages(angleDeg)

    if tree[0] != 'z':
        stages.reverse()

    print('params', angleDeg, stepX, stepY, moveX, moveY, stages, tree, 'index map:', indexMap.shape, 'pixel payload:', pixelPayload.shape, 'motion payload:', motionPayload.shape, width, height, origin)
    passedLayers = 0
    for curStage in stages:
        if curStage == 1:
            curRange = range(width - 1, - 1, -1)
            fixedY, fixedX = 0, None
        elif curStage == 2:
            curRange = range(0, height)
            fixedY, fixedX = None, 0
        elif curStage == 3:
            curRange = range(0, width)
            fixedY, fixedX = height - 1, None
        elif curStage == 4:
            curRange = range(height - 1, - 1, -1)
            fixedY, fixedX = None, width - 1

        if tree[0] != 'z':  # f uses the opposite order
            curRange = list(curRange)
            curRange.reverse()

        for var in curRange:
            curX = fixedX if fixedX is not None else var
            curY = fixedY if fixedY is not None else var
            found = False

            while - 0.5 <= curX < indexMap.shape[1] - 0.5 and - 0.5 <= curY < indexMap.shape[0] - 0.5:
                newCoords = (round(curY), round(curX))
                idx = indexMap[newCoords]
                print(curY, curX, height, width, newCoords, indexMap.shape)
                if pixelPayload[idx][1] > 0:
                    found = True
                updateFn(motionPayload, idx, moveX, moveY, tree[-1][1])

                curX += stepX
                curY += stepY

            if found:
                passedLayers += 1
            if passedLayers >= requiredLayers:
                break
        if passedLayers >= requiredLayers:
            break

def scanOblique(tree, indexMap, pixelPayload, motionPayload, origin, width, height):
    def update(motionPayload, idx, moveX, moveY, _):
        motionPayload[idx][0] += moveY
        motionPayload[idx][1] += moveX

    applyOblique(tree, indexMap, pixelPayload, motionPayload, origin, width, height, update)

def scanCondOblique(tree, indexMap, pixelPayload, condMotionPayload, origin, width, height):
    def update(condMotionPayload, idx, moveX, moveY, outputCode):
        condMotionPayload[idx].append(((moveY, moveX), outputCode))

    applyOblique(tree, indexMap, pixelPayload, condMotionPayload, origin, width, height, update)

def planMovements(tree, indexMap, pixelPayload, motionPayload, condMotionPayload, origin, width, height):
    # Compute the movement vectors based on the tree
    if tree[0] == 'parallel':
        if tree[-1] == 'filler' or isinstance(tree[-1], int):
            tree = tree[:-1]  # ignore "filler" string
        for branch in tree[1:]:
            planMovements(branch, indexMap, pixelPayload, motionPayload, condMotionPayload, origin, width, height)
    elif tree[0] == 'series':
        division = width / len(tree[1:])  # horizontal = body forward
        for i, comp in enumerate(tree[1:]):
            planMovements(comp, indexMap, pixelPayload, motionPayload, condMotionPayload, (origin[0] + round(division * i), origin[1]),
                          round(division * (i + 1)) - round(division * i), height)
    elif tree[0] in ['ż', 'm', 'ċ']:
        # Random movements (Brownian motion)
        intensity = charMapDyn[tree[0]][0]

        # Update the condition motion map if it is a condition
        if isPair(tree[-1]) and tree[-1][1] != 0:
            region = indexMap[origin[1] : origin[1] + height][origin[0] : origin[0] + width]
            for idx in region.flat:
                condMotionPayload[idx].append((intensity, tree[-1][1]))

        # Unconditioned elements and input-driven conditioned elements are enabled
        # Condition elements are disabled if they are not input-driven
        if not (isPair(tree[-1]) and tree[-1][0] == 0 and tree[-1][1] != 0):
            region = indexMap[origin[1] : origin[1] + height][origin[0] : origin[0] + width]
            for idx in region.flat:
                if np.random.rand() < intensity:
                    motionPayload[idx] = [random.choice([-intensity, intensity]), random.choice([-intensity, intensity])]

        '''
        intensity = charMapDyn[tree[0]][0]

        randomDirY, randomDirX = random.choice([-intensity, 0, intensity]), random.choice([-intensity, 0, intensity])
        print('randdir', randomDirY, randomDirX)
        for i in range(height):
            for j in range(width):
                motionMap[origin[1] + i][origin[0] + j] = [randomDirY, randomDirX]

        '''
    elif tree[0] in ['z', 'f']:
        # Update the condition motion map if it is conditional
        if isPair(tree[-1]) and tree[-1][1] != 0:
            scanCondOblique(tree, indexMap, pixelPayload, condMotionPayload, origin, width, height)

        # Unconditioned elements and input-driven conditioned elements are enabled
        # Condition elements are disabled if they are not input-driven
        if isPair(tree[-1]) and tree[-1][0] == 0 and tree[-1][1] != 0:
            return

        # Only "corrode" the top layer
        scanOblique(tree, indexMap, pixelPayload, motionPayload, origin, width, height)
    elif tree[0] in charMapDyn:
        # Lay out the two movement regions according to the dynamic element
        orient = orient2Angle(tree[1])  # trigram's orientation
        if orient == 270.0:  # vertical
            division = height / 2
        elif orient == 0.0:  # horizontal
            division = width / 2

        move = vecMap[reversedHexMap[charMapDyn[tree[0]]]][0]  # [0] = vectors, not magnitudes

        if isPair(tree[-1]) and tree[-1][1] != 0:
            outputCode = tree[-1][1]
            if orient == 270.0:
                for i in range(2):
                    region = indexMap[origin[1] + round(division * i): origin[1] + round(division * (i + 1)), origin[0]: origin[0] + width]
                    for idx in region.flat:
                        condMotionPayload[idx].append(((move[i], None), outputCode))
            else:
                for i in range(2):
                    region = indexMap[origin[1]: origin[1] + height, origin[0] + round(division * i): origin[0] + round(division * (i + 1))]
                    for idx in region.flat:
                        condMotionPayload[idx].append(((None, move[i]), outputCode))

        # Unconditioned elements and input-driven conditioned elements are enabled
        # Condition elements are disabled if they are not input-driven
        if isPair(tree[-1]) and tree[-1][0] == 0 and tree[-1][1] != 0:
            return

        if orient == 270.0:
            for i in range(2):
                region = indexMap[origin[1] + round(division * i): origin[1] + round(division * (i + 1)), origin[0]: origin[0] + width]
                for idx in region.flat:
                    motionPayload[idx][0] += move[i]
        else:
            for i in range(2):
                region = indexMap[origin[1]: origin[1] + height, origin[0] + round(division * i): origin[0] + round(division * (i + 1))]
                for idx in region.flat:
                    motionPayload[idx][1] += move[i]


def getSign(value):
    if value == 0:
        return 0
    elif value > 0:
        return 1
    else:
        return -1


def runMovements(indexMap, pixelPayload, motionPayload, childMotionPayload=None, emptyPayload=None, isBody=False, isPlan=False):
    global tweening
    newPixelPayloads = []
    newMotionPayloads = []
    newChildMotionPayloads = []
    newEmptyPayloads = []
    height, width = indexMap.shape
    for _ in range(tweening):
        newPixelPayloads.append(np.copy(pixelPayload))
        if isBody:
            newMotionPayloads.append(np.copy(motionPayload))
            newEmptyPayloads.append(np.copy(emptyPayload))
            if childMotionPayload is not None:
                newChildMotionPayloads.append(np.copy(childMotionPayload))

    if (not isBody and np.all(motionPayload == 0) or
            isBody and
            (childMotionPayload is None or np.all(motionPayload == 0) and np.all(childMotionPayload == 0))):
        # Leave early if there is no motion to compute
        return (newPixelPayloads, newMotionPayloads, newChildMotionPayloads, newEmptyPayloads) if isBody else newPixelPayloads

    # Precompute index → position mapping
    ys, xs = np.indices(indexMap.shape)
    idx_to_y = np.empty(pixelPayload.shape[0], dtype=int)
    idx_to_x = np.empty(pixelPayload.shape[0], dtype=int)
    idx_to_y[indexMap] = ys
    idx_to_x[indexMap] = xs

    for idx in range(indexMap.shape[0]):
        if not isBody and pixelPayload[idx][1] == 0:  # don't compute empty pixel positions
            continue

        curY, curX = idx_to_y[idx], idx_to_x[idx]
        initY, initX = curY, curX
        tentY, tentX = None, None
        dirY, dirX = motionPayload[idx][0], motionPayload[idx][1]
        if dirY == 0 and dirX == 0:
            continue
        initSignY, initSignX = getSign(dirY), getSign(dirX)
        isInBounds = True
        isInit = True

        # go forever until intercardinal direction changes or stops
        while getSign(dirY) == initSignY and getSign(dirX) == initSignX and isInBounds:
            if isInit:
                isInit = False
            else:
                curY = tentY
                curX = tentX
                dirY, dirX = motionPayload[idx]
                if isBody:
                    dirY += childMotionPayload[idx][0]
                    dirX += childMotionPayload[idx][1]

            if dirY == 0 and dirX == 0:  # pixel stops moving
                break

            tentY = curY + dirY  # tentative coordinates
            tentX = curX + dirX

            isInBounds = -0.5 <= tentY < height and -0.5 <= tentX < width  # inside the frame boundary

        if curY == initY and curX == initX:  # remained in the same place after all
            continue

        # Generate tween animation
        for frameInd in range(tweening):
            # Transfer the value at the original position to the new position
            newPixelPayloads[frameInd][idx] -= pixelPayload[idx]
            if isBody:
                if not isPlan:  # movement runs on itself like an airflow
                    newMotionPayloads[frameInd][idx] -= motionPayload[idx]
                    newChildMotionPayloads[frameInd][idx] -= childMotionPayload[idx]
                newEmptyPayloads[frameInd][idx] -= emptyPayload[idx]
            if isInBounds:  # if crosses the frame boundary, pixel disappears from the screen
                midY = round(initY + (curY - initY) * (frameInd + 1) / tweening)
                midX = round(initX + (curX - initX) * (frameInd + 1) / tweening)
                midIdx = indexMap[midY][midX]
                newPixelPayloads[frameInd][midIdx] += pixelPayload[idx]
                if isBody:
                    if not isPlan:
                        newMotionPayloads[frameInd][midIdx] += motionPayload[idx]
                        newChildMotionPayloads[frameInd][midY][midX] += childMotionPayload[idx]
                    newEmptyPayloads[frameInd][midIdx] += emptyPayload[idx]

    return (newPixelPayloads, newMotionPayloads, newChildMotionPayloads, newEmptyPayloads) if isBody else newPixelPayloads


def avgMovements(pixelMap, motionMap):  # Sum up all the forces being applied to the pixels
    sumY, sumX, numY, numX = 0, 0, 0, 0
    for i in range(len(pixelMap)):
        for j in range(len(pixelMap[0])):
            if pixelMap[i, j, 1] > 0 and (motionMap[i, j, 0] != 0 or motionMap[i, j, 1] != 0):  # accumulate the movement if it is not a static pixel
                sumY += motionMap[i, j][0]
                sumX += motionMap[i, j][1]
                numY += 1
                numX += 1
    numY = 1 if numY == 0 else numY
    numX = 1 if numX == 0 else numX
    return sumY / numY, sumX / numX

def existsMovement(pixelMap, motionMap, initSignX, initSignY):
    exists = False
    for i in range(len(pixelMap)):
        for j in range(len(pixelMap[0])):
            if pixelMap[i, j, 1] > 0 and (motionMap[i, j, 0] != 0 or motionMap[i, j, 1] != 0):  # accumulate the movement if it is not a static pixel
                if getSign(motionMap[i, j, 0]) == initSignY and getSign(motionMap[i, j, 1]) == initSignX:
                    exists = True
    return exists

def scanRegions(tree, pixelMap, codeMap, origin, width, height):
    # Find the regions that need to be scanned and compute their output
    if tree[0] == 'parallel':
        if tree[-1] == 'filler' or isinstance(tree[-1], int):
            tree = tree[:-1]  # ignore "filler" string
        for branch in tree[1:]:
            codeMap = scanRegions(branch, pixelMap, codeMap, origin, width, height)
    elif tree[0] == 'series':
        division = width / len(tree[1:])  # horizontal = body forward
        for i, comp in enumerate(tree[1:]):
            codeMap = scanRegions(comp, pixelMap, codeMap, (origin[0] + round(division * i), origin[1]),
                                  round(division * (i + 1)) - round(division * i), height)
    else:
        # If it is an element and has an output code
        if isPair(tree[-1]) and tree[-1][1] != 0:
            mapLegend = tree[-1][1]
            tree = tree[:-1]
        else:
            return codeMap
        divNum = len(charMapAll[tree[0]])
        pixelSums, pixelLens = countPixels(pixelMap, tree[1], divNum, needHelp=True)
        results = findBestFit(pixelSums=pixelSums, pixelLens=pixelLens, punishment=1, punishmentBi=1, punishmentDyn=1)
        referent = results[3]
        scores = results[4]
        maxScore = max(scores)  # worst case
        similarity = (maxScore - scores[list(referent.values()).index(charMap[tree[0]])]) / maxScore
        print('ssscore', tree[0], divNum, len(pixelSums), results, maxScore, similarity)
        codeMap.setdefault(mapLegend, []).append(similarity)
    return codeMap


def readVideo(name, jump=1, addition=''):
    video = cv2.VideoCapture('./images/videos/' + addition + '/' * bool(addition) + name + '.mp4')
    success, image = video.read()
    count = 0
    folder = './images/frames/' + addition + ' ' + name + '/'
    path = pathlib.Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    while success:
        cv2.imwrite('./images/frames/' + addition + ' ' + name + '/' * bool(addition) + 'frame %d.png' % count,
                    image)  # save frame as JPEG file
        for i in range(jump):
            success, image = video.read()
        print('Read a new frame: ', success)
        count += jump
    print("Finished reading the video.", count, "frames in total.")


def readGif(name, jump=1):
    gif = imageio.get_reader('./images/videos/' + name + '.gif')
    for count in range(0, len(gif), jump):
        imageio.imsave("./images/frames/frame %d.png" % count, gif.get_data(count))
    print("Finished reading the gif.", len(gif), "frames in total.")


def initAxes(showMode=[True, False, True], numAxes=1):  # image, zero layer tree, complete tree
    global axes
    global fig
    fig = plt.figure(figsize=(9, 5))
    axes = []
    numCol = int(showMode[0]) + int(showMode[1]) * 2 + int(showMode[2])
    sumMode = sum([int(mode) for mode in showMode])
    widthRatios = [1 / sumMode / 3] * int(showMode[0]) + [0.5 / sumMode / 2] * 2 * int(showMode[1]) + [
        1 / sumMode] * int(
        showMode[2])
    realNumAxes = numAxes + 1 if showMode[1] else numAxes  # add the buttons below
    heightRatios = [0.9 / (realNumAxes - 1)] * (realNumAxes - 1) + [0.1] if showMode[1] else [1 / numAxes] * numAxes
    gs = fig.add_gridspec(realNumAxes, numCol, height_ratios=heightRatios, width_ratios=widthRatios)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
    curCol = 0
    if showMode[0]:
        for i in range(numAxes):
            textImgPair = gs[i, curCol].subgridspec(2, 1, height_ratios=[0.2, 0.8])
            axes.append(fig.add_subplot(textImgPair[0]))  # title
            axes.append(fig.add_subplot(textImgPair[1]))  # image
        curCol += 1
    if showMode[1]:
        axes.append(fig.add_subplot(gs[:-1, curCol:curCol + 2]))
        axes.append(fig.add_subplot(gs[-1, curCol], ))  # button 1
        axes.append(fig.add_subplot(gs[-1, curCol + 1]))  # button 2
        curCol += 2
    if showMode[2]:
        axes.append(fig.add_subplot(gs[:, curCol]))
    return fig


def initSourceAxes(showMode):
    global axes
    global fig
    fig = plt.figure()
    axes = []
    print("source show mode", showMode)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 0.1])
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
    upper = gs[0, 0].subgridspec(1, 2, width_ratios=[0.3, 0.7]) if showMode[0] and showMode[2] else None
    if showMode[0]:
        if upper is None:
            upperLeft = gs[0, 0].subgridspec(3, 1, height_ratios=[0.1, 0.8, 0.1])
        else:
            upperLeft = upper[0].subgridspec(3, 1, height_ratios=[0.1, 0.8, 0.1])
        axes.append(fig.add_subplot(upperLeft[0]))
        axes.append(fig.add_subplot(upperLeft[1]))
    if showMode[2]:
        if upper is None:
            upperRight = gs[0, 0].subgridspec(2, 1, height_ratios=[0.9, 0.1])
        else:
            upperRight = upper[1].subgridspec(2, 1, height_ratios=[0.9, 0.1])
        axes.append(fig.add_subplot(upperRight[0]))
        upperRightButtons = upperRight[1].subgridspec(1, 2, width_ratios=[0.5, 0.5])
        axes.append(fig.add_subplot(upperRightButtons[0]))
        axes.append(fig.add_subplot(upperRightButtons[1]))

    lower = gs[1, 0].subgridspec(1, 3, width_ratios=[0.3, 0.4, 0.3])
    axes.append(fig.add_subplot(lower[0]))
    axes.append(fig.add_subplot(lower[1]))
    axes.append(fig.add_subplot(lower[2]))

    return fig


def updateFrame(frame, frames, numAxes):
    # Function to update the animation at each frame
    axes[numAxes + 1].clear()  # Clear previous frame's content
    axes[numAxes + 1].imshow(frames[frame], cmap='gray', vmin=0, vmax=255)  # Display new frame's content
    axes[numAxes + 1].axis('on')
    axes[numAxes + 1].tick_params(left=False, right=False, labelleft=False, labelbottom=False)
    return [axes[numAxes], axes[numAxes + 1]]


def playFrames(frames, numAxes, maxLayer):
    global fig
    axes[numAxes].axis("off")
    axes[numAxes].text(0.5, 0.5, "Simulated Layer " + str(int(maxLayer - numAxes / 2)), ha='center', va='center',
                       fontsize=10, family='sans-serif')
    ani = animation.FuncAnimation(fig, updateFrame, frames=len(frames), fargs=(frames, numAxes),
                                  init_func=None,
                                  blit=True, interval=20)

    return ani


def optimizeLayerRange(layerLst):
    # Optimize the layer range by omitting layers that are detached from the zero layer group
    layerSetDif = set(range(min(layerLst), max(layerLst) + 1)).difference(set(layerLst))
    if layerSetDif == set():
        minLayer = min(layerLst)
        maxLayer = max(layerLst) + 1
    else:
        negLayers = {val for val in layerSetDif if val < 0}
        posLayers = {val for val in layerSetDif if val > 0}
        minLayer = max(negLayers) + 1 if negLayers else min(layerLst)
        maxLayer = min(posLayers) if posLayers else max(layerLst) + 1
        if maxLayer == 0:
            maxLayer = 1
    return list(range(minLayer, maxLayer)), minLayer, maxLayer


def compLayers(guideTree, layerMode=True, verbose=False,
               fillDyn=True, mirrorMode=True, keyLen=20, fWidth=33, fHeight=33):
    # Simulate the dynamics specified by the guide trees
    global unknownTree
    global bodyDict
    global tweening
    global globalCurLayer
    global globalPixelMap
    global frameWidth
    global frameHeight

    if guideTree[0] not in ['series', 'parallel', 'body', 'plan']:
        guideTree = ['series', guideTree]
    guideTree = labelBodies(guideTree)[0]
    print('labelled', guideTree)
    layerLst, minLayer, maxLayer = optimizeLayerRange(getAllLayers(guideTree))

    # Filter out the trees that belong to each layer
    layerTrees = {}
    for layer in range(minLayer, maxLayer):
        if -layer not in layerTrees.keys():
            # Check if the symmetrical counterpart tree exists. They share the same "layer tree" space
            # Thus assuming mirror mode is on
            if fillDyn:
                # Complement the dynamic elements with static elements
                filledTree = fillDynamic(guideTree)
                if verbose:
                    print('Filled Tree', filledTree)
            else:
                filledTree = copy.deepcopy(guideTree)
            layerTrees[layer] = \
                filterLayer(filledTree, layer, codeMode=True, mirrorMode=mirrorMode, verbose=verbose)[0]

            cleanedTree = cleanGuideTree(layerTrees[layer])
            # If the layer tree is empty, it is replaced with the unknown tree
            layerTrees[layer] = unknownTree if cleanedTree is None else cleanedTree

            if verbose:
                print('Layer', layer, 'Tree', layerTrees[layer])
        else:
            layerTrees[layer] = layerTrees[-layer]  # e.g. layer -2 and layer 2 are put together

    # Timetable specifies probabilities that serve as inputs across every layer at every frame
    timetable = {}
    videos = [[] for _ in range(maxLayer - minLayer)]
    zeroLayerTrees = []
    layerPixelMaps = [None for _ in range(maxLayer - minLayer)]
    layerMotionMaps = [None for _ in range(maxLayer - minLayer)]
    bodyDicts = [{} for _ in range(maxLayer - minLayer)]
    globalPixelMap = {}
    frameWidth, frameHeight = fWidth, fHeight

    for curFrameInd in range(keyLen):  # traverse through the layers at every frame
        usefulProbs = {}
        print('current frame index', curFrameInd)
        for i, layer in enumerate(range(minLayer, maxLayer)):
            globalCurLayer = layer
            if layer - 1 in timetable:
                # Probabilities useful for subsequent computations
                # Provided by the layer below current layer and the layer below the mirrored layer
                usefulProbs |= timetable[layer - 1]
            if -layer - 1 in timetable:
                usefulProbs |= timetable[-layer - 1]
            bodyDict = bodyDicts[i]
            newVideo, newProbs, guideTree, layerPixelMaps[i], layerMotionMap = drawGuideTree(layerTrees[layer],
                                                                                  usefulProbs=usefulProbs,
                                                                                  layerMode=layerMode,
                                                                                  verbose=verbose,
                                                                                  pixelMap=layerPixelMaps[i],
                                                                                  isInit=curFrameInd == 0)
            if layerMotionMap is not None:
                layerMotionMaps[i] = layerMotionMap
            bodyDicts[i] = bodyDict
            if layer == 0:
                if guideTree[0] == 'series' and len(guideTree) == 2 and guideTree[1][0] == 'parallel':
                    zeroLayerTrees.append(guideTree[1])  # Simplify the tree and add it to the 0th-layer tree list
                else:
                    zeroLayerTrees.append(guideTree)

            # Add new frames to the video sequence for this layer
            # The addition's length is controlled by the tweening parameter
            videos[i] += newVideo
            timetable[layer] = newProbs
    if not zeroLayerTrees:
        zeroLayerTrees = [unknownTree] * keyLen
    return videos, layerLst, zeroLayerTrees


def initButtons(zeroGraphs, startAx, replaceText=True, videos=None, useSymbols=False, isSource=False):
    global axPrev
    global axNext
    global buttonPrevTree
    global buttonNextTree
    global buttonPrevSeq
    global buttonNextSeq
    global curGalleryInd
    global curSeqInd
    global numGallery
    global axes
    curGalleryInd = 0
    curSeqInd = 0
    if zeroGraphs is not None:
        if isinstance(zeroGraphs[0], list):
            numSeq = len(zeroGraphs)
            numGallery = len(zeroGraphs[0])
        else:
            numGallery = len(zeroGraphs)
    else:
        numSeq = len(videos)
    arrowPrevImg = imread('./images/arrowPrev.png')  # Replace with your arrow image file
    arrowNextImg = imread('./images/arrowNext.png')

    if not isSource or zeroGraphs is not None:
        buttonPrevTree = Button(axes[startAx + 1], '', image=arrowPrevImg)
        buttonNextTree = Button(axes[startAx + 2], '', image=arrowNextImg)
        buttonPrevTree.on_clicked(lambda event: updateGallery((curGalleryInd - 1) % numGallery, zeroGraphs, startAx,
                                                              replaceText=replaceText, useSymbols=useSymbols))
        buttonNextTree.on_clicked(lambda event: updateGallery((curGalleryInd + 1) % numGallery, zeroGraphs, startAx,
                                                              replaceText=replaceText, useSymbols=useSymbols))
    if isSource:
        buttonPrevSeq = Button(axes[startAx + 3], '', image=arrowPrevImg)
        axes[startAx + 4].axis("off")
        axes[startAx + 4].text(0.5, 0.5, 'Sequence ' + str(curSeqInd + 1), ha='center', va='center', fontsize=12)
        buttonNextSeq = Button(axes[startAx + 5], '', image=arrowNextImg)

        buttonPrevSeq.on_clicked(lambda event: updateGallery((curSeqInd - 1) % numSeq, zeroGraphs, startAx,
                                                             replaceText=replaceText, videos=videos,
                                                             useSymbols=useSymbols, changeSeq=True))
        buttonNextSeq.on_clicked(lambda event: updateGallery((curSeqInd + 1) % numSeq, zeroGraphs, startAx,
                                                             replaceText=replaceText, videos=videos,
                                                             useSymbols=useSymbols, changeSeq=True))


def updateGallery(index, zeroGraphs, startAx, replaceText=True, videos=None, useSymbols=False, changeSeq=False):
    global axes
    global curGalleryInd
    global curSeqInd
    global ani
    global textProps
    if changeSeq:
        curSeqInd = index
        axes[startAx + 4].cla()
        axes[startAx + 4].axis("off")
        axes[startAx + 4].text(0.5, 0.5, 'Sequence ' + str(curSeqInd + 1), ha='center', va='center', fontsize=12)
    else:
        curGalleryInd = index
    if zeroGraphs is not None:
        axes[startAx].cla()
        if isinstance(zeroGraphs[0], list):
            drawNetwork(zeroGraphs[curSeqInd][curGalleryInd], startAx, replaceText, useSymbols)
            axes[startAx].set_title('Zeroth Layer ' + str(curGalleryInd + 1) + '/' + str(len(zeroGraphs[0])))
        else:
            drawNetwork(zeroGraphs[curGalleryInd], startAx, replaceText, useSymbols)
            axes[startAx].set_title('Zeroth Layer ' + str(curGalleryInd + 1) + '/' + str(len(zeroGraphs)))
    if videos is not None:
        axes[0].clear()
        axes[0].cla()  # Clear the current axes
        ani[0] = None
        ani[0] = playFrames(videos[curSeqInd], 0, 0)
    plt.ion()


def drawNetwork(graph, startAx, replaceText, useSymbols):
    global axes
    labels, colorMap, pos, fontFamily = decorateNetwork(graph, replaceText=replaceText, useSymbols=useSymbols)

    # nx.draw_networkx_nodes(graph, pos, ax=axes[startAx], node_size=1000, node_color='white', edgecolors=colorMap)
    nx.draw_networkx_edges(graph, pos, ax=axes[startAx])
    for tup, labelData in labels.items():
        isStatic = tup[1][0] in charMap if isinstance(tup[1], tuple) else False
        unique = tup[0]

        if replaceText:  # circular nodes, which include trigram symbol nodes and character nodes
            fontColor = 'white' if isStatic else 'black'
        else:  # plain text nodes
            fontColor = 'black'

        if isinstance(labelData, dict):  # hexagram nodes
            label = labelData['label']
            rotation = labelData['rotation']
            ha = labelData['ha']
            va = labelData['va']
            axes[startAx].text(pos[unique][0], pos[unique][1], label, rotation=rotation, ha=ha, va=va, size=12,
                               color=fontColor, weight='bold', family=fontFamily[unique])
        else:  # text nodes
            label = labelData
            nx.draw_networkx_labels(graph, pos, ax=axes[startAx], labels={unique: label}, font_size=12,
                                    font_color=fontColor, font_weight='bold', font_family=fontFamily[unique])
        if isStatic:
            nx.draw_networkx_nodes([unique], pos, ax=axes[startAx], node_size=1000, node_color=colorMap[unique],
                                   edgecolors='white')
        else:
            nx.draw_networkx_nodes([unique], pos, ax=axes[startAx], node_size=1000, node_color='white',
                                   edgecolors=colorMap[unique])


def decorateNetwork(graph, replaceText=True, useSymbols=False):
    global fontNames
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")  # Using Spectral Layout
    colorMap = {}
    fontFamily = {}
    labels = {}
    label2Color = {0: 'mediumseagreen', 1: 'coral', 2: 'crimson', -1: 'blueviolet', -2: 'darkslategrey'}
    for tup in graph.nodes(data=True):
        unique = tup[0]  # unique node name
        if isinstance(tup[1]['label'], list):
            tup = (unique, lst2tup(tup[1]['label']))
        else:
            tup = (unique, tup[1]['label'])
        if replaceText:
            layer = None
            if isinstance(tup[1], tuple):
                layer = tup[1][2]
            if layer is not None:
                colorMap[unique] = mcolors.CSS4_COLORS[label2Color[layer]]
            else:
                colorMap[unique] = mcolors.CSS4_COLORS['dimgray']

            if not isinstance(tup[1], list) and tup[1] in replaceDict:
                if useSymbols:
                    fontFamily[unique] = fontNames[1]
                    if tup[1] == nodeSymbols[0]:
                        labels[tup] = '—'
                    elif tup[1] == nodeSymbols[1]:
                        labels[tup] = '//'
                    elif tup[1] == nodeSymbols[2]:
                        labels[tup] = '*'
                    elif tup[1] == nodeSymbols[3]:
                        labels[tup] = '#'
                else:
                    fontFamily[unique] = fontNames[0]
                    labels[tup] = replaceDict[tup[1]]
            elif tup[1][0] in leafSymbols:
                if useSymbols:
                    labels[tup] = {
                        'label': reversedSymMap[charMapAll[tup[1][0]]][
                            0],
                        'rotation': 90 * (tup[1][1] - 1), 'ha': 'center', 'va': 'center'}
                    '''
                    labels[tup] += (tup[1][0] in charMap) * "/" \
                                      + (tup[1][0] in charMapDyn) * "\\"
                    labels[tup] += dimMap[1 - tup[1][1]]
                    '''
                    fontFamily[unique] = fontNames[1]
                else:
                    labels[tup] = \
                        reversedAllMap[charMapAll[tup[1][0]]]
                    if len(charMapAll[tup[1][0]]) == 3:
                        labels[tup] = labels[tup][0]
                    '''
                    labels[tup] += (tup[1][0] in charMap) * '靜' \
                                      + (tup[1][0] in charMapDyn) * '動'
                    '''
                    # labels[tup] += dimMap[int(0 <= tup[1][1] < 45 or 135 <= tup[1][1] < 225 or 315 <= tup[1][1] < 360)]
                    labels[tup] += dimMap[1 - tup[1][1]]
                    fontFamily[unique] = fontNames[0]
        else:
            colorMap[unique] = 'white'
            fontFamily[unique] = fontNames[2]
            labels[tup] = tup[1]
    return labels, colorMap, pos, fontFamily


def playLayers(videos, layerLst=[0], showMode=[True, False, True], isSource=False, mirrorMode=False, extraDir='',
               extraName='', noWindow=False):
    global axes
    global ani
    global fig
    layerLst = sorted(set(layerLst))
    if mirrorMode:
        layerLst = [val for val in layerLst if val <= 0] if any([val for val in layerLst if val <= 0]) else layerLst
    if not noWindow:
        if isSource:
            fig = initSourceAxes(showMode=showMode)
        else:
            fig = initAxes(showMode=showMode, numAxes=max(layerLst) + 1 - min(layerLst))

    if showMode[0]:
        layerNum = max(layerLst) + 1 - min(layerLst)
        if not noWindow:
            ani = []
            for i in range(layerNum):
                ani.append(None)
        for i in range(layerNum):
            # Create an empty image plot for the first animation
            # im1 = gridAx.imshow(videos[layer][0], animated=True)
            # print(mirrorMode, layerNum, layerLst, layerNum - 1 - i, ani, len(videos))

            # Convert pixel maps into frames
            videos[i] = [frame.convert('RGB') for frame in videos[i]]
            for j, frame in enumerate(videos[i]):
                frame.save('./images/results/simulations/layer ' + str(layerLst[i]) + ' - ' + str(j) + '.png')
            gifPath = './images/results/simulations/' + extraDir + '/layer ' + str(layerLst[i]) + extraName + '.gif'
            os.makedirs(os.path.dirname(gifPath), exist_ok=True)
            if os.path.exists(gifPath):
                os.remove(gifPath)
            videos[i][0].save(gifPath, save_all=True, append_images=videos[i][1:], loop=0, duration=150)
            if not noWindow:
                ani[layerNum - 1 - i] = playFrames(videos[i], (layerNum - 1 - i) * 2, max(layerLst))


def flattenLists(nestedLst):
    flattened = []

    for item in nestedLst:
        if isinstance(item, list):
            flattened.extend(flattenLists(item))
        else:
            flattened.append(item)

    return flattened


def createGifGrid(gif_2d_list, output_path='animated_output.gif'):
    # Load all frames from each GIF and find the GIF with the maximum number of frames
    gif_frames = []
    max_frames = 0
    for row in gif_2d_list:
        row_frames = []
        for gif_path in row:
            frames = [frame.copy() for frame in ImageSequence.Iterator(Image.open(gif_path))]
            max_frames = max(max_frames, len(frames))
            row_frames.append(frames)
        gif_frames.append(row_frames)

    rows, cols = len(gif_frames), len(gif_frames[0])

    # Calculate total size of the grid
    total_width = cols * gif_frames[0][0][0].width
    total_height = rows * gif_frames[0][0][0].height

    # Create each frame of the output GIF
    output_frames = []
    for frame_index in range(max_frames):
        grid_frame = Image.new('RGBA', (total_width, total_height))
        y_offset = 0
        for row in gif_frames:
            x_offset = 0
            for gif in row:
                # Loop back to the start if the current index exceeds the number of frames in the GIF
                gif_frame = gif[frame_index % len(gif)]
                grid_frame.paste(gif_frame, (x_offset, y_offset))
                x_offset += gif_frame.width
            y_offset += gif_frame.height
        output_frames.append(grid_frame)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the animated GIF
    output_frames[0].save(output_path, save_all=True, append_images=output_frames[1:], loop=0, duration=100)


def analyzeImages(filename, start, end, frameJump=1, staticMode=True, dynamicMode=False, colorMode=False,
                  scaleFactor=1):
    # Analyze all images starting with filename from start to end after being scaled
    lastPixels = [None, None, None] if colorMode else [None]
    imageSeq = []
    treeImageSeq = []
    resultGuideTreeCollection = []
    for frameOrd, frameNum in enumerate(range(start, end, frameJump)):
        image = Image.open(r"./images/frames/" + filename + ' ' + str(frameNum) + ".png")
        # image = Image.open(r"./images/frames/layer 0 - " + str(frameNum) + ".png")
        print(image.mode)
        # detectContour(image)
        # sourceImages = [detectEdge(image, useCanny=True)]
        # sourceImages = [Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY))]
        # sourceImages = extractChannels(image) if colorMode else [
        #    Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY))]
        # sourceImages = [image]
        sourceImages = [image.convert('L')]
        resultImages = []
        resultTreeImages = []
        resultVideos = []
        resultGuideTrees = []
        for i, image in enumerate(sourceImages):
            if len(sourceImages) == 3:
                image.save('./images/results/' + ['red', 'green', 'blue'][i] + 'Image.png')
            # image.rotate(-90)
            width, height = image.size
            if frameOrd == 0:
                initWidth, initHeight = width, height
            elif width != initWidth or height != initHeight:
                image = image.resize((initWidth, initHeight), resample=Image.BOX)
                print("resized", image.size)
            pixels = list(image.getdata())
            pixels = [pixels[i * initWidth:(i + 1) * initWidth] for i in range(initHeight)]
            if staticMode:
                rays, tree = splitImage(pixels, (0, 0), bothOrient=True, flexible=False, buddhist=False)
                print('rays', rays)
                print('tree', tree)
                treeImage = drawTree(tree)
                treeImage.save('./images/results/tree' + str(frameNum) + '-' + str(i) + '.png')
                resultTreeImages.append(treeImage)
                image = image.resize((round(image.size[0] * scaleFactor), round(image.size[1] * scaleFactor)),
                                     resample=Image.BOX)
                image = drawRays(image, rays, scaleFactor=scaleFactor)
                image = drawLabels(image, tree, allTrees=True, scaleFactor=scaleFactor)
                image.save(r'./images/results/result' + str(frameNum) + '-' + str(i) + '.png')
                resultImages.append(image)
                guideTree = convert2GuideTree(tree)
                if guideTree[0] == 'series':
                    guideTree = ['parallel', guideTree]
                resultGuideTrees.append(guideTree)
            if dynamicMode:
                if lastPixels[i] is not None:
                    flow = detectFlow(np.array(lastPixels[i]), np.array(pixels))
                    rays, tree = splitImage(pixels, (0, 0), flow=flow, bothOrient=True, flexible=True, buddhist=True)
                    print('rays', rays)
                    print('tree', tree)
                    if not staticMode:
                        treeImage = drawTree(tree)
                        treeImage.save('./images/results/treeDyn' + str(frameNum) + '-' + str(i) + '.png')
                        resultTreeImages.append(treeImage)
                    image = image.resize((image.size[0] * scaleFactor, image.size[1] * scaleFactor), resample=Image.BOX)
                    image = visualizeFlow(image, flow, scaleFactor=scaleFactor)
                    image = drawRays(image, rays, scaleFactor=scaleFactor)
                    image = drawLabels(image, tree, allTrees=True, scaleFactor=scaleFactor)
                    image.save(r'./images/results/resultDyn' + str(frameNum) + '-' + str(i) + '.png')
                    resultImages.append(image)
                    guideTree = convert2GuideTree(tree)
                    if guideTree[0] == 'series':
                        guideTree = ['parallel', guideTree]
                    resultGuideTrees.append(guideTree)
                    video, probDicts, guideTreeLst, moveHistDict = drawGuideTree(guideTree, frameDiv=20)
                    video[0].save(r'./images/results/treeDynGuide' + str(frameNum) + '-' + str(i) + '.png')
                    resultVideos.append(video)
                lastPixels[i] = pixels

        if len(resultTreeImages) == 3:
            mergedImage = Image.merge("RGB", (resultTreeImages[0], resultTreeImages[1], resultTreeImages[2]))
            mergedImage.save('./images/results/mergedTree' + str(frameNum) + '.png')
            treeImageSeq.append(mergedImage)
        elif len(resultTreeImages) == 1:
            treeImageSeq.append(resultTreeImages[0])
            imageSeq.append(resultImages[0])
        if len(resultVideos) == 3:
            mergedVideo = []
            for i in range(len(resultVideos[0])):
                mergedVideo.append(Image.merge("RGB", (resultVideos[0][i], resultVideos[1][i], resultVideos[2][i])))
            mergedVideo[0].save('./images/results/mergedVideo' + str(frameNum) + '.gif'
                                , save_all=True, append_images=mergedVideo[1:], loop=0, duration=100)
        print('Result guide trees', resultGuideTrees)
        resultGuideTreeCollection.append(resultGuideTrees)
    print('Result guide tree collection', resultGuideTreeCollection)  # attention: dynamic lacks the first guide tree
    treeImageSeq[0].save('./images/results/treeDynSeq.gif'
                         , save_all=True, append_images=treeImageSeq[1:], loop=0, duration=250)
    imageSeq[0].save('./images/results/resultSeq.gif'
                     , save_all=True, append_images=imageSeq[1:], loop=0, duration=250)
    return resultGuideTreeCollection


def main():
    global axes
    global ani
    import bayes_pixels as bayes
    global tweening
    # sourceTree = ['series', ['h', 0, 0], ['body', ['parallel', ['series', ['t', 0, 0]], ['series', ['t', 0, 0]], ['body', ['parallel', ['series', ['h', 0, -1]], ['series', ['j', 0, 0]], ['series', ['t', 1, 0]], ['series', ['t', 0, 0]]], ['series', ['r', 1, 0]]]]], ['h', 0, 0]]

    sourceTree = ['series', ['body', ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]], ['body',
                                                                            ['parallel',
                                                                             ['series', ['j', 0, 0]],
                                                                             ['parallel', ['h', 0, -1],
                                                                             ['series', ['t', 0, 0]],
                                                                             ['series', ['t', 1, 0]]]],
                                                                            ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 0, 0]]
                                                                             ]]]], ['n', 0, 0]]

    # shang
    sourceTree = ['body', ['parallel', ['c', 1, 0], ['j', 0, 0]], ['series',
                  ['body', ['parallel', ['t', 0, 0], ['j', 0, 0], ['body', ['parallel', ['ṫ', 1, 0], ['t', 1, 0]],
                                        ['parallel', ['j', 0, 0], ['body', ['parallel', ['ṫ', 1, 0], ['t', 1, 0]]]],
                                        ['parallel', ['j', 0, 0], ['body', ['parallel', ['ṫ', 1, 0], ['t', 1, 0]]]]]],
                   ['r', 0, 0], ['r', 0, 0]], ['n', 0, 0]]]

    # fractal tree
    sourceTree = ['body', ['parallel', ['series', ['c', 1, 0]], ['series', ['j', 0, 0]]],
                  ['series', ['body', ['r', 1, 0], ['r', 1, 0], ['r', 1, 0]],
                  ['body', ['parallel', ['series', ['t', 1, 0]], ['series', ['j', 0, 0]]],
                                ['series',
                                 ['body',
                                 ['parallel', ['series', ['c', 1, 0]], ['series', ['t', 0, 0]], ['series', ['s', 1, 0]]],
                                 ['parallel', ['j', 0, 0], ['body',
                                                            ['parallel', ['ṫ', 1, 0], ['t', 1, 0]],
                                                                    ['parallel', ['j', 0, 0],
                                                           ['body', ['parallel', ['l̇', 1, 0], ['t', 1, 0]]]],
                                                                    ['parallel', ['j', 0, 0],
                                                           ['body', ['parallel', ['l̇', 1, 0], ['t', 1, 0]]]]]]
                                 ],
                                 ['h', 0, 0],
                                 ['body',
                                 ['parallel', ['series', ['c', 1, 0]], ['series', ['c', 0, 0]], ['series', ['v', 1, 0]]],
                                  ['parallel', ['j', 0, 0], ['body',
                                                             ['parallel', ['ṫ', 1, 0], ['t', 1, 0]],
                                                                     ['parallel', ['j', 0, 0],
                                                            ['body', ['parallel', ['l̇', 1, 0], ['c', 1, 0]]]],
                                                                     ['parallel', ['j', 0, 0],
                                                              ['body', ['parallel', ['l̇', 1, 0], ['c', 1, 0]]]]
                                                             ]]
                                  ],
                                 ],

                   ['r', 0, 0]], ['n', 0, 0]]]

    # rotating double bars
    sourceTree = ['body', ['b', 0, 0], ['parallel', ['parallel', ['h', 0, -1], ['series', ['t', 1, 0], ['t', 0, 0]]], ['j', 0, 0]], ['r', 0, 0]]


    # 熒
    sourceTree = ['body', ['parallel', ['body', ['parallel', ['t', 0, 0], ['c', 1, 0], ['j', 0, 0]]],
                                       ['body', ['parallel', ['t', 0, 0], ['t', 1, 0], ['j', 0, 0]]]],
                    ['parallel', ['r', 0, 0], ['body', ['parallel', ['c', 1, 0], ['l̇', 0, 0]],
                                               ['parallel', ['r', 1, 0], ['body', ['l̇', 0, 0], ['r', 1, 0], ['parallel', ['parallel', ['l', 1, 0], ['h', 0, -1]], ['b', 1, 0]], ['r', 1, 0]]]]],
                          ['r', 0, 0], ['r', 0, 0], ['j', 0, 0]]

    # push
    sourceTree = ['series', ['body', ['parallel', ['t', 0, 0], ['j', 0, 0]]], ['n', 0, 0], ['j', 0, 0], ['n', 0, 0]]

    # revolving car
    sourceTree = ['body', ['r', 0, 0], ['r', 1, 0], ['r', 0, 0], ['r', 1, 0], ['series', ['body', ['parallel', ['body', ['series', ['h', 0, 0]]], ['series', ['t', 0, 0]], ['body',
                                                                            ['parallel',
                                                                             ['series', ['j', 0, 0]],
                                                                             ['parallel', ['h', 0, -1],
                                                                             ['series', ['t', 0, 0]],['series', ['t', 0, 0]],
                                                                             ['series', ['t', 1, 0]]]],
                                                                            ['parallel', ['body', ['series', ['r', 1, 0]]], ['series', ['t', 0, 0]]
                                                                             ]]]], ['n', 0, 0]]]

    # 蠕动
    sourceTree = ['series', ['n', 0, 0], ['body', ['parallel', ['p', 0, 0], ['t', 0, 0], ['j', 0, 0]]], ['n', 0, 0]]

    sourceTree = ['series', ['n', 0, 0], ['body', ['parallel', ['t', 1, 0], ['j', 0, 0]], ['series', ['body', ['parallel', ['b', 1, 0], ['t', 1, 0]]], ['n', 0, 0]]], ['n', 0, 0]]
    sourceTree = ['parallel', ['body', ['parallel', ['t', 1, 0], ['j', 0, 0]], ['series', ['n', 0, 0], ['body', ['parallel', ['b', 0, 0], ['t', 0, 0], ['z', 0, 0], ['f', 0, 0]]], ['n', 0, 0]]], ['body', ['r', 0, 0], ['r', 0, 0]]]




    # 来回
    sourceTree = ['series', ['parallel', ['body', ['j', 0, 0]], ['t', 0, 0]], ['c', 0, 0]]
    sourceTree = ['body', ['r', 0, 0], ['r', 1, 0], ['series', ['body', ['parallel', ['parallel', ['body', ['h', 0, 0]], ['t', 0, 0]], ['body', ['parallel', ['parallel', ['h', 0, -1], ['t', 1, 0], ['t', 0, 0]], ['j', 0, 0]], ['parallel', ['series', ['body', ['r', 1, 0]]], ['series', ['j', 0, 0]]]]]], ['n', 0, 0]]]


    # chaos
    sourceTree = ['parallel', ['n', 0, 0], ['body', ['d', 0, 0], ['r', 1, 0], ['j', 0, 0]], ['body', ['parallel', ['b', 0, 0], ['ż', 0, 0]]]]


    sourceTree = ['body', ['r', 0, 0], ['body', ['parallel', ['t', 0, 0], ['t', 1, 0], ['j', 0, 0]], ['parallel', ['f', 0, 0], ['b', 0, 0]]]]


    sourceTree = ['series', ['body', ['r', 0, 0], ['r', 1, 0], ['parallel', ['h', 0, -1], ['parallel', ['body', ['j', 0, 0]], ['t', 0, 0]]]], ['n', 0, 0]]

    sourceTree = ['series', ['parallel', ['h', 0, -1], ['body', ['parallel', ['body', ['j', 0, 0]], ['t', 0, 0]]]], ['n', 0, 0], ['n', 0, 0], ['n', 0, 0]]


    # two phases
    sourceTree = ['parallel', ['plan', ['ṫ', 0, 0], ['parallel', ['body', ['j', 0, 0]], ['t', 0, 0]]],
                  ['plan', ['l̇', 1, 0], ['c', 0, 0]]]

    # four phases
    sourceTree = ['parallel', ['plan', ['ṫ', 1, 0], ['series', ['parallel', ['body', ['j', 0, 0]], ['t', 0, 0]], ['parallel', ['body', ['j', 0, 0]], ['t', 1, 0]]]],
                                ['plan', ['l̇', 1, 0], ['series', ['c', 1, 0], ['c', 0, 0]]]]

    # oblique cut
    sourceTree = ['body', ['r', 0, 0], ['r', 1, 0], ['parallel', ['b', 0, 0], ['body', ['parallel', ['t', 0, 0], ['c', 1, 0], ['j', 0, 0]], ['parallel', ['f', 0, 0]]]]]
    # wandering eye
    sourceTree = ['body',  ['parallel', ['parallel', ['h', 0, -1], ['ż', 0, 0]], ['body',['r', 0, 0], ['r', 1, 0],['r', 0, 0], ['parallel', ['r', 1, 0], ['parallel', ['h', 0, -1], ['t', 1, 0]]], ['b', 0, 0]]]]

    # sourceTree = ['series', ['parallel', ['parallel', ['h', 0, -1], ['body', ['parallel', ['j', 0, -1], ['d', 0, 0]]]], ['t', 0, 0]], ['t', 1, 0]]  # --> d should only be generated once

    # sourceTree = ['body', ['parallel', ['b', 1, 0], ['parallel', ['h', 0, -1], ['ż', 1, 0]]]]

    # sourceTree = ['body', ['parallel', ['t', 0, 0], ['body', ['d', 0, 0], ['j', 0, 0]]]]
    #sourceTree = ['series', ['body', ['series', ['parallel', ['t', 0, 0], ['t', 0, 0], ['t', 0, 0], ['t', 0, 0], ['t', 1, 0], ['j', 0, 0]],
    #                                  ['parallel', ['t', 0, 0], ['j', 0, 0]]], ['parallel', ['b', 0, 0], ['t', 1, 0]]]]
    #sourceTree = ['parallel', ['d', 0, 0], ['body', ['parallel', ['series', ['t', 1, 0]], ['series', ['t', 0, 0]], ['series', ['j', 0, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['body', ['g', 0, 0], ['r', 1, 0]]]]]]
    # sourceTree = ['body', ['parallel', ['series', ['c', 1, 0]], ['series', ['j', 0, 0]]], ['series', ['h', 0, 0], ['body', ['ṫ', 1, 0], ['parallel', ['series', ['h', 0, -1]], ['series', ['c', 1, 0]], ['series', ['t', 0, 0]], ['series', ['j', 0, 0]]],  ['parallel', ['series', ['r', 0, 0]]]]]]
    #sourceTree = ['body', ['parallel', ['series', ['c', 1, 0]], ['series', ['j', 0, 0]]],
    #              ['series', ['d', 0, 0], ['g', 0, 0]]]
    # sourceTree = ['parallel', ['series', ['b', 0, 0]], ['series', ['d', 1, 0]]]

    # scanning tree
    sourceTree = ['parallel', ['l̇', 0, 0], ['series', ['parallel', ['s', 0, -1], ['series', ['parallel', ['l̇', 0, 0]]],
                                           ['series', ['parallel', ['ċ', 0, 0]], ['parallel', ['q', 1, -1]],
                                            ['parallel', ['n', 0, -1]]]], ['parallel', ['g', 1, -1]],
                                ['parallel', ['p', 0, -1]]], ['series', ['parallel', ['l̇', 0, -1],
                                                                         ['series', ['parallel', ['ġ', 0, -1]],
                                                                          ['parallel', ['r', 1, 0]],
                                                                          ['parallel', ['d', 1, 0]]],
                                                                         ['series', ['parallel', ['z', 0, -1]],
                                                                          ['parallel', ['t', 1, -1]]],
                                                                         ['series', ['parallel', ['v', 0, 0]]]],
                                                              ['parallel', ['j', 1, -1]], ['parallel', ['ġ', 1, -1]]]]
    sourceTree = ['series', ['parallel', ['series', ['ġ', 0, -1]], ['series', ['n', 0, -1]], ['series', ['parallel', ['series', ['t', 0, 0]], ['series', ['parallel', ['series', ['ṡ', 0, -1]], ['series', ['b', 0, 0]]]]]]], ['n', 0, 0]]
    sourceTree = ['series', ['parallel', ['body', ['b', 0, 0]], ['t', 0, 0]], ['n', 0, 0]]
    sourceTree = ['series', ['parallel', ['parallel', ['b', 0, 0], ['body', ['ṡ', 0, 1]]], ['t', 0, -1]], ['parallel', ['ṡ', 0, -1], ['d', 0, 0]]]
    sourceTree = ['series', ['parallel', ['b', 0, 0], ['d', 0, -1], ['g', 0, -1]]]
    print('source tree', sourceTree)
    videos, layerLst, zeroLayerTrees = compLayers(sourceTree, keyLen=16, verbose=True, fillDyn=False, mirrorMode=True, fWidth=33, fHeight=33)
    print("videos", videos)
    print("layer lst", layerLst)
    print("zero", zeroLayerTrees)
    graph = nx.DiGraph()
    labelCounter = {}
    print('source', sourceTree)
    nodeTree = bayes.convert2NodeTree(sourceTree, labelCounter=labelCounter, graph=graph)
    zeroGraphs = []
    for i in range(len(zeroLayerTrees)):
        zeroGraphs.append(nx.DiGraph())
    testNodeCol = [bayes.convert2NodeTree(testTree, graph=zeroGraphs[i]) for i, testTree in enumerate(zeroLayerTrees)]
    bayes.showPlot(graph=graph, videos=videos, layerLst=layerLst, zeroGraphs=zeroGraphs, replaceText=True,
                   mirrorMode=True, useSymbols=False)
    # plt.show()

    # readGif('rotating', jump=1)
    # for i in range(1, 6):
    #    readVideo(str(i), jump=10, addition='drenching')
    res = analyzeImages('line', 0, 21, scaleFactor=5, staticMode=False, dynamicMode=True)
    print('res', res)


if __name__ == "__main__":
    main()