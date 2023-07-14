import argparse
from concurrent.futures import wait
import cv2 as cv
import numpy as np
from itertools import product
from math import sqrt, cos, pi
from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt
from dahuffman import HuffmanCodec
import math
import datetime


def extractYUV(file_name, height, width):
    """
    Extracts the Y, U, and V components of the frames in the given video file.
    :param file_name: filepath of video file to extract frames from.
    :param height: height of video.
    :param width: width of video.
    :param start_frame: first frame to be extracted.
    :param end_frame: final frame to be extracted.
    :
    """

    fp = open(file_name, 'rb')
    fp.seek(0, 2)  # Seek to end of file
    fp_end = fp.tell()  # Find the file size

    frame_size = height * width * 3 // 2  # Size of a frame in bytes
    num_frame = fp_end // frame_size  # Number of frames in the video
    print("This yuv file has {} frame imgs!".format(num_frame))
    fp.seek(0, 0)  # Seek to the start of the first frame

    YUV = []
    for i in range(num_frame):
        yuv = np.zeros(shape=frame_size, dtype='uint8', order='C')
        for j in range(frame_size):
            yuv[j] = ord(fp.read(1))  # Read one byte from the file

        img = yuv.reshape((height * 3 // 2, width)).astype('uint8')  # Reshape the array    
        
        # YUV420
        y = np.zeros((height, width), dtype='uint8', order='C')
        u = np.zeros((height // 2) * (width // 2), dtype='uint8', order='C')
        v = np.zeros((height // 2) * (width // 2), dtype='uint8', order='C')
        
        # assignment
        y = img[:height, :width]
        u = img[height : height * 5 // 4, :width]
        v = img[height * 5 // 4 : height * 3 // 2, :width]

        # reshape
        u = u.reshape((height // 2, width // 2)).astype('uint8')
        v = v.reshape((height // 2, width // 2)).astype('uint8')

        # save
        YUV.append({'y': y, 'u': u, 'v': v})


    fp.close()
    print("job done!")
    return YUV, num_frame

def YUV2RGB(y, u, v, height, width):
    '''
    Converts YUV to RGB.
    :param y: Y component.
    :param u: U component.
    :param v: V component.
    :param height: height of image.
    :param width: width of image.
    :return: RGB components.
    '''
    yuv = np.zeros((height * 3 // 2, width), dtype='uint8', order='C')
    y = y.reshape((height, width))
    u = u.reshape((-1, width))
    v = v.reshape((-1, width))
    yuv[:height, :width] = y
    yuv[height : height*5//4, :width] = u
    yuv[height*5//4 : height*3//2, :width] = v
    rgb = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_I420)  
    return rgb

def quantize(mat, width, height, isInv=False, isLum=True):
    '''
    Performs quantization or its inverse operation on an image matrix.
    :param mat: DCT coefficient matrix or quantized image matrix.
    :param width: width of matrix.
    :param height: height of matrix.
    :param isInv: flag indicating whether inverse quantization is to be performed.
    :param isLum: flag indicating which image quantization matrix should be used (luminance for Y component, chrominance for Cb/Cr components.).
    :return: image matrix that has undergone quantization or its inverse.
    '''
    quantized = np.zeros((height, width))
    scale = 31

    DC_step_size = 8
    AC_step_size = 2 * scale
    # Perform quantization or its inverse depending on isInv flag.
    if isInv:
        quantized = (mat * AC_step_size).astype(np.int32)
        quantized[0:width:8, 0:height:8] = (mat[0:width:8, 0:height:8] * DC_step_size).astype(np.int32)
    else:
        quantized = (mat / AC_step_size).astype(np.int32)
        quantized[0:width:8, 0:height:8] = (mat[0:width:8, 0:height:8] / DC_step_size).astype(np.int32)
    return quantized

def extractCoefficients(mat, width, height):
    '''
    Extracts the DC and AC coefficients of the quantized 8x8 block within a frame and places it in a single row of a
    coefficient matrix according to zigzag pattern.
    :param mat: input image matrix.
    :param width: width of image.
    :param height: height of image.
    :return: coefficent matrix with 64 DC and AC coefficents for column values, for each pixel of the 8x8 block.
    '''
    numRows = (height // 8) * (width // 8)  # No. of rows in coefficient matrix is number of 8x8 blocks in the image.
    coeffMat = np.zeros((numRows, 64))
    matIdx = np.array([0,  1,  5,  6, 14, 15, 27, 28,
                    2,  4,  7, 13, 16, 26, 29, 42,
                    3,  8, 12, 17, 25, 30, 41, 43,
                    9, 11, 18, 24, 31, 40, 44, 53,
                    10, 19, 23, 32, 39, 45, 52, 54,
                    20, 22, 33, 38, 46, 51, 55, 60,
                    21, 34, 37, 47, 50, 56, 59, 61,
                    35, 36, 48, 49, 57, 58, 62, 63])
    for N, M in product(range(0, height, 8), range(0, width, 8)):
        if N >= height // 8*8 or M >= width // 8*8:
            break
        num = N // 8 * width // 8 + M // 8

        coeffMat[num][matIdx] = mat[N:N+8, M:M+8].reshape(-1)

    return coeffMat

def IextractCoefficients(coeffMat,width,height):
    """
    :Reconstruct block
    :param width: width of frame.
    :param height: height of frame.
    """
    blockMat = np.zeros((height, width))
    matIdx = np.array([0,  1,  5,  6, 14, 15, 27, 28,
                    2,  4,  7, 13, 16, 26, 29, 42,
                    3,  8, 12, 17, 25, 30, 41, 43,
                    9, 11, 18, 24, 31, 40, 44, 53,
                    10, 19, 23, 32, 39, 45, 52, 54,
                    20, 22, 33, 38, 46, 51, 55, 60,
                    21, 34, 37, 47, 50, 56, 59, 61,
                    35, 36, 48, 49, 57, 58, 62, 63])
    for N, M in product(range(0, height, 8), range(0, width, 8)):
        if N >= height // 8*8 or M >= width // 8*8:
            break
        num = N // 8 * width // 8 + M // 8
        blockMat[N:N+8, M:M+8] = coeffMat[num][matIdx].reshape((8,8))

    return blockMat

def motionEstimation(y_curr, y_ref, u_ref, v_ref, width, height):
    '''
    Computes motion estimation for an image based on its reference frame.
    :param y_curr: Y component of current frame; motion estimation is soley done on Y component.
    :param y_ref: Y component of reference frame.
    :param u_ref: u component of reference frame.
    :param v_ref: v component of reference frame.
    :param width: width of frame.
    :param height: height of frame.
    :return: YUV components of predicted frame, coordinate matrix for quiver plot, and motion vector matrices.
    '''
    # Init y_pred u_pred v_pred etc;
    h_num = math.ceil(height/16)
    w_num = math.ceil(width/16)
    size = h_num*w_num
    MotionVector_arr = np.zeros((2, size)).astype(int)
    MotionVector_subarr = np.zeros((2, size)).astype(int)
    y_pred = np.zeros((height, width))
    u_pred = np.zeros((height // 2, width // 2))
    v_pred = np.zeros((height // 2, width // 2))
    CoMatrix = np.zeros((4,size))
    mv_row = mv_col = 0

    # Different location has different search size
    SearchWindow_dict = {
        576: 81,
        768: 153,
        1024: 289
    }

    # For each macroblock in the frame:
    mv_idx = 0
    for n, m in product(range(0, height, 16), range(0, width, 16)):
        MB_curr = y_curr[n:n + 16, m:m + 16]  # Current macroblock.

        # Identify search window parameters. For 8 px in each directions, we can have search windows of sizes 24x24,
        # 24x32, 32x24, or 32x32.
        SW_hmin = 0 if n - 8 < 0 else n - 8
        SW_wmin = 0 if m - 8 < 0 else m - 8
        SW_hmax = height if n + 16 + 8 > height else n + 16 + 8
        SW_wmax = width if m + 16 + 8 > width else m + 16 + 8

        SW_x = SW_wmax - SW_wmin
        SW_y = SW_hmax - SW_hmin
        SW_size = int(SW_x * SW_y)

        # Number of candidate blocks == SearchArea.
        SearchArea = 0
        for x, y in SearchWindow_dict.items():
            if x == SW_size:
                SearchArea = y
                break
        SA_vect = np.zeros(SearchArea)
        SA_arr = np.zeros((2, SearchArea)).astype(int)
        for i in range(SearchArea):
            SA_vect[i] = 99999.0
            SA_arr[0, i] = -1
            SA_arr[1, i] = -1

        # Go through the designated search window for the current macroblock.
        SW_tmp = 0
        for i, j in product(range(SW_hmin, SW_hmax - 15), range(SW_wmin, SW_wmax - 15)):
            MB_temp = y_ref[i:i + 16, j:j + 16]
            if MB_curr.shape[0] != 16 or MB_curr.shape[1] != 16:
                pass
            else:
                diff = np.float32(MB_curr) - np.float32(MB_temp)
                SA_vect[SW_tmp] = np.sum(np.abs(diff))
                SA_arr[0, SW_tmp] = i
                SA_arr[1, SW_tmp] = j
                SW_tmp += 1

        if MB_curr.shape[0] != 16 or MB_curr.shape[1] != 16:
            pass
        else:
        # Get minimum SAD (sum of absolute differences) and search for its corresponding microblock.
            SAD_min = min(SA_vect)
            for i in range(SearchArea):
                if SA_vect[i] == SAD_min:
                    mv_row = (SA_arr[0, i])
                    mv_col = (SA_arr[1, i])
                    break

            # The coordinates gives the the top left pixel + the motion vector coordinates dx and dy.
            MotionVector_arr[0, mv_idx] = mv_row - n
            MotionVector_arr[1, mv_idx] = mv_col - m

            # Do the same for u/v
            MotionVector_subarr[0, mv_idx] = int((mv_row - n) // 2)
            MotionVector_subarr[1, mv_idx] = int((mv_col - m) // 2)

        if MB_curr.shape[0] != 16 or MB_curr.shape[1] != 16:
            y_pred[n:n + 16, m:m + 16] = np.float32(y_ref[n:n + MB_curr.shape[0], m: m+ MB_curr.shape[1]])
        else:
            # Apply the motion vectors to the current block of the reference frame,
            y_pred[n:n + 16, m:m + 16] = np.float32(y_ref[mv_row:mv_row + 16, mv_col:mv_col + 16])

            # Get motion vector inputs for quiver().
            CoMatrix[0, mv_idx] = m
            CoMatrix[1, mv_idx] = n
            CoMatrix[2, mv_idx] = mv_col - m
            CoMatrix[3, mv_idx] = mv_row - n

            mv_idx += 1

    # Reconstruct  u/v.
    uv_idx = 0
    for i, j in product(range(0, (height // 2), 8), range(0, (width // 2), 8)):
        if i + 8 > (height//2):
            u_pred[i:height//2,j:j+8] = np.float32(u_ref[i:height//2,j:j+8])
            v_pred[i:height//2,j:j+8] = np.float32(v_ref[i:height//2,j:j+8])
        else:
            ref_row = i + (MotionVector_subarr[0, uv_idx])
            ref_col = j + (MotionVector_subarr[1, uv_idx])

            u_pred[i:i + 8, j:j + 8] = np.float32(u_ref[ref_row:ref_row + 8, ref_col:ref_col + 8])
            v_pred[i:i + 8, j:j + 8] = np.float32(v_ref[ref_row:ref_row + 8, ref_col:ref_col + 8])

            uv_idx += 1

    return CoMatrix, MotionVector_arr, MotionVector_subarr, y_pred, u_pred, v_pred

def getDC(CoeffMat):
    '''
    Computes DC coefficients for a given YUV component.
    :param CoeffMat: YUV component.
    :return: DC coefficients.
    '''
    dc_coeff = np.zeros(CoeffMat.shape[0])
    dc_coeff = CoeffMat[:, 0]
    dcdpcm = np.zeros(CoeffMat.shape[0])
    dcdpcm[0] = dc_coeff[0]
    dcdpcm[1:] = dc_coeff[1:] - dc_coeff[:-1]
    return dc_coeff, dcdpcm


def getAC(CoeffMat):
    '''
    Computes AC coefficients for a given YUV component using RLE
    :param CoeffMat: YUV component.
    :return: AC coefficients.
    '''
    ac_coeff = []
    for i in range(CoeffMat.shape[0]):
        "using the run length encoding algorithm"
        cnt = 0
        for x in CoeffMat[i, 1:]:
            if x == 0:
                cnt += 1
            if x != 0:
                ac_coeff.append((cnt, x))
                cnt = 0
        ac_coeff.append((0, 0))
    return ac_coeff


def huffmanCoding(data):
    '''
    Huffman coding for data.
    :param data:data.
    :return: Huffman coded and encode.
    '''
    codec = HuffmanCodec.from_data(data)
    encode = codec.encode(data)
    return codec, encode

def MatDecode(dc_codec, dc_encode, ac_codec, ac_encode, num):
    '''
    Decodes DC and AC coefficients.
    :param dc_codec: DC Huffman codec.
    :param dc_encode: DC Huffman encoded coefficients.
    :param ac_codec: AC Huffman codec.
    :param ac_encode: AC Huffman encoded coefficients.
    :return: Decoded DC and AC coefficients.
    '''
    dc_decode = HuffmanCodec.decode(dc_codec, dc_encode)
    dc = np.zeros((num, ))
    dc = dc_decode[:]
    dc = np.cumsum(dc)
    ac_decode = HuffmanCodec.decode(ac_codec, ac_encode)
    Mat = np.zeros((num, 64))
    Mat[:, 0] = dc
    block = 0
    cur = 1
    for ac in ac_decode:
        if ac == (0, 0):
            Mat[block, cur : 64] = 0
            block += 1
            cur = 1
        else:
            cnt = ac[0]
            Mat[block, cur : cur + cnt] = 0
            Mat[block, cur + cnt] = ac[1]
            cur += cnt + 1
        
    return Mat


def flatten_2d_array(arr):
    # Flatten a 2D array into a 1D sequence
    return [item for sublist in arr for item in sublist]

def reshape_1d_array(arr, shape):
    # Reshape a 1D array into the specified shape
    return np.reshape(arr, shape)


def encode(y, u, v, height, width,MV_arr, MV_sub_arr):
    yDCT, uDCT, vDCT = dctn(y), dctn(u), dctn(v)

    yQuant = quantize(yDCT, width, height)
    uQuant = quantize(uDCT, width // 2, height // 2)
    vQuant = quantize(vDCT, width // 2, height // 2)

    # Extract DC and AC coefficients; these would be transmitted to the decoder in a real MPEG
    # encoder/decoder framework.
    yCoeffMat = extractCoefficients(yQuant, width, height)
    
    dc_y, dpcm_y = getDC(yCoeffMat)
    ac_y = getAC(yCoeffMat)
    dccodec_y, dcencode_y = huffmanCoding(dpcm_y)
    accodec_y, acencode_y = huffmanCoding(ac_y)
    y_encoded = [dccodec_y,dcencode_y,accodec_y,acencode_y,yCoeffMat]
    
    uCoeffMat = extractCoefficients(uQuant, width // 2, height // 2)
    dc_u, dpcm_u = getDC(uCoeffMat)
    ac_u = getAC(uCoeffMat)
    dccodec_u, dcencode_u = huffmanCoding(dpcm_u)
    accodec_u, acencode_u = huffmanCoding(ac_u)
    u_encoded = [dccodec_u,dcencode_u,accodec_u,acencode_u,uCoeffMat]

    vCoeffMat = extractCoefficients(vQuant, width // 2, height // 2)
    dc_v, dpcm_v = getDC(vCoeffMat)
    ac_v = getAC(vCoeffMat)
    dccodec_v, dcencode_v = huffmanCoding(dpcm_v)
    accodec_v, acencode_v= huffmanCoding(ac_v)
    v_encoded = [dccodec_v,dcencode_v,accodec_v,acencode_v,vCoeffMat] 


    mvcodec, mvencode= huffmanCoding(MV_arr.flatten())
    mv_encoded = [mvcodec,mvencode]

    mvsubcodec, mvsubencode= huffmanCoding(MV_sub_arr.flatten())
    mv_sub_encoded = [mvsubcodec,mvsubencode]
    
    return  y_encoded,u_encoded,v_encoded,mv_encoded,mv_sub_encoded

def decode( height, width,y_encoded,u_encoded,v_encoded,mv_encoded,mv_sub_encoded):

# Calculate the size of each encoded data
    mvcodec,mvencode=mv_encoded[0],mv_encoded[1]
    mvsubcodec,mvsubencode=mv_sub_encoded[0],mv_sub_encoded[1]

    dccodec_y, dcencode_y, accodec_y, acencode_y, yCoeffMat = y_encoded[0],y_encoded[1],y_encoded[2],y_encoded[3],y_encoded[4]
    dccodec_v,dcencode_v,accodec_v,acencode_v,vCoeffMat = v_encoded[0],v_encoded[1],v_encoded[2],v_encoded[3],v_encoded[4]
    dccodec_u,dcencode_u,accodec_u,acencode_u,uCoeffMat = u_encoded[0],u_encoded[1],u_encoded[2],u_encoded[3],u_encoded[4]
    
    # Perform inverse quantization.
    # decoding
    mv_decode = mvcodec.decode(mvencode)
    mv_sub_decode = mvsubcodec.decode(mvsubencode)



    YMatRecon = MatDecode(dccodec_y, dcencode_y, accodec_y, acencode_y, yCoeffMat.shape[0])
    YQuantRecon = IextractCoefficients(YMatRecon, width, height)

    vMatRecon = MatDecode(dccodec_v,dcencode_v,accodec_v,acencode_v,vCoeffMat.shape[0])
    vQuantRecon = IextractCoefficients(vMatRecon,width//2,height//2)

    uMatRecon = MatDecode(dccodec_u,dcencode_u,accodec_u,acencode_u,uCoeffMat.shape[0])
    uQuantRecon = IextractCoefficients(uMatRecon,width//2,height//2)
    
    # perform inverse quantization
    yIQuant = quantize(YQuantRecon, width, height, isInv=True)
    uIQuant = quantize(uQuantRecon, width // 2, height // 2, isInv=True, isLum=False)
    vIQuant = quantize(vQuantRecon, width // 2, height // 2, isInv=True, isLum=False)

    #perform inverse DCT
    yIDCT = idctn(yIQuant)
    uIDCT = idctn(uIQuant)
    vIDCT = idctn(vIQuant)
    
    return yIDCT, uIDCT, vIDCT, mv_decode,mv_sub_decode

def encode_I(y, u, v, height, width):
    '''
    Encodes and decodes the YUV components.
    :param y: YUV component.
    :param u: YUV component.
    :param v: YUV component.
    :param height: Height of the image.
    :param width: Width of the image.
    :return: Encoded and decoded YUV components.
    '''

    yDCT, uDCT, vDCT = dctn(y), dctn(u), dctn(v)

    yQuant = quantize(yDCT, width, height)
    uQuant = quantize(uDCT, width // 2, height // 2)
    vQuant = quantize(vDCT, width // 2, height // 2)

    # Extract DC and AC coefficients; these would be transmitted to the decoder in a real MPEG
    # encoder/decoder framework.
    yCoeffMat = extractCoefficients(yQuant, width, height)
    
    dc_y, dpcm_y = getDC(yCoeffMat)
    ac_y = getAC(yCoeffMat)
    dccodec_y, dcencode_y = huffmanCoding(dpcm_y)
    accodec_y, acencode_y = huffmanCoding(ac_y)
    y_encoded = [dccodec_y,dcencode_y,accodec_y,acencode_y,yCoeffMat]

    uCoeffMat = extractCoefficients(uQuant, width // 2, height // 2)
    dc_u, dpcm_u = getDC(uCoeffMat)
    ac_u = getAC(uCoeffMat)
    dccodec_u, dcencode_u = huffmanCoding(dpcm_u)
    accodec_u, acencode_u = huffmanCoding(ac_u)
    u_encoded = [dccodec_u,dcencode_u,accodec_u,acencode_u,uCoeffMat]

    vCoeffMat = extractCoefficients(vQuant, width // 2, height // 2)
    dc_v, dpcm_v = getDC(vCoeffMat)
    ac_v = getAC(vCoeffMat)
    dccodec_v, dcencode_v = huffmanCoding(dpcm_v)
    accodec_v, acencode_v= huffmanCoding(ac_v)
    v_encoded = [dccodec_v,dcencode_v,accodec_v,acencode_v,vCoeffMat] 
    return y_encoded,u_encoded,v_encoded

def decode_I(height, width,y_encoded,u_encoded,v_encoded):

    # Perform inverse quantization.
    # decoding

    dccodec_y, dcencode_y, accodec_y, acencode_y, yCoeffMat = y_encoded[0],y_encoded[1],y_encoded[2],y_encoded[3],y_encoded[4]
    dccodec_v,dcencode_v,accodec_v,acencode_v,vCoeffMat = v_encoded[0],v_encoded[1],v_encoded[2],v_encoded[3],v_encoded[4]
    dccodec_u,dcencode_u,accodec_u,acencode_u,uCoeffMat = u_encoded[0],u_encoded[1],u_encoded[2],u_encoded[3],u_encoded[4]

    YMatRecon = MatDecode(dccodec_y, dcencode_y, accodec_y, acencode_y, yCoeffMat.shape[0])
    YQuantRecon = IextractCoefficients(YMatRecon, width, height)

    vMatRecon = MatDecode(dccodec_v,dcencode_v,accodec_v,acencode_v,vCoeffMat.shape[0])
    vQuantRecon = IextractCoefficients(vMatRecon,width//2,height//2)

    uMatRecon = MatDecode(dccodec_u,dcencode_u,accodec_u,acencode_u,uCoeffMat.shape[0])
    uQuantRecon = IextractCoefficients(uMatRecon,width//2,height//2)
    
    # perform inverse quantization
    yIQuant = quantize(YQuantRecon, width, height, isInv=True)
    uIQuant = quantize(uQuantRecon, width // 2, height // 2, isInv=True, isLum=False)
    vIQuant = quantize(vQuantRecon, width // 2, height // 2, isInv=True, isLum=False)

    #perform inverse DCT
    yIDCT = idctn(yIQuant)
    uIDCT = idctn(uIQuant)
    vIDCT = idctn(vIQuant)
    
    return yIDCT, uIDCT, vIDCT


def main():
    #desc = 'Showcase of image processing techniques in MPEG encoder/decoder framework.'
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', dest='src', required=True)
    parser.add_argument('--size', dest='size', required=True)
    parser.add_argument('--fps', dest='fps', required=True)
    parser.add_argument('--dst', dest='dst', required=True)

    args = parser.parse_args()

    # Get arguments
    filepath = args.src
    width, height = map(int, args.size.split('x'))
    fps = int(args.fps)
    start_frame = 0
    end_frame = 150
    dst = args.dst
    # print start time
    print('Start time: ' + str(datetime.datetime.now()))
    frames, num_frame = extractYUV(filepath, height, width)
    print('End time: ' + str(datetime.datetime.now()))

    video = cv.VideoWriter(dst, cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    k = 0
    i = 0
    PSNR = []
    encoded_file_size=0
    y_encoded_frames, u_encoded_frames, v_encoded_frames, MV_encoded_frames, MV_sub_array_encoded_frames = [], [], [], [], []


    for frame_num in range(len(frames)):
        curr = frames[frame_num]
        if curr is None:
            continue
        yCurr, uCurr, vCurr = curr['y'], curr['u'], curr['v']
        if frame_num % 2 == 0:
            print("[I] compressing frame " + str(frame_num))

            y_encoded, u_encoded, v_encoded = encode_I(yCurr, uCurr, vCurr, height, width)
            y_encoded_frames.append(y_encoded)
            u_encoded_frames.append(u_encoded)
            v_encoded_frames.append(v_encoded)
            MV_encoded_frames.append(0)
            MV_sub_array_encoded_frames.append(0)
            for i in range(len(y_encoded)):
                if i==1 or i==3:
                    encoded_file_size=encoded_file_size+len(y_encoded[i])+len(u_encoded[i])+len(v_encoded[i])

            # re_rgb = YUV2RGB(y_encoded.astype(np.uint8),u_encoded.astype(np.uint8), v_encoded.astype(np.uint8), height, width)
            y_ref = yCurr
            u_ref = uCurr
            v_ref = vCurr

        else:
            print("[P] compressing frame " + str(frame_num))

            # Do motion estimatation using the I-frame as the reference frame for the current frame in the loop.python mpeg.py --file 'walk_qcif.avi' --extract 6 10
            coordMat, MV_arr, MV_subarr, yPred, uPred, vPred = motionEstimation(yCurr,y_ref, u_ref, v_ref, width,height)

            yTmp, uTmp, vTmp = yPred, uPred, vPred

            # Get residual frame
            yDiff = yCurr.astype(np.uint8) - yTmp.astype(np.uint8)
            uDiff = uCurr.astype(np.uint8) - uTmp.astype(np.uint8)
            vDiff = vCurr.astype(np.uint8) - vTmp.astype(np.uint8)

            y_encoded, u_encoded, v_encoded ,MV_encoded, MV_subarr_encoded = encode(yDiff, uDiff, vDiff, height, width,MV_arr,MV_subarr)
            y_encoded_frames.append(y_encoded)
            u_encoded_frames.append(u_encoded)
            v_encoded_frames.append(v_encoded)
            MV_encoded_frames.append(MV_encoded)
            MV_sub_array_encoded_frames.append(MV_subarr_encoded)

            # encoded_file_size = encoded_file_size + len(MV_encoded[1]) + len(MV_subarr_encoded[1])
            for i in range(len(y_encoded)):
                if i==1 or i==3:
                    encoded_file_size=encoded_file_size+len(y_encoded[i])+len(u_encoded[i])+len(v_encoded[i])
            
            k += 1
            re_rgb = YUV2RGB(yCurr.astype(np.uint8),uCurr.astype(np.uint8),vCurr.astype(np.uint8), height, width)
            diffMat = YUV2RGB(yDiff,uDiff,vDiff,height,width)
            pred_rgb = YUV2RGB(yPred, uPred, vPred, height, width)
            # plot 
            plt.figure(figsize=(10, 10))
            curr = YUV2RGB(yCurr, uCurr, vCurr, height, width)
            curr_plt = cv.cvtColor(curr, cv.COLOR_BGR2RGB)
            re_rgb_plt = cv.cvtColor(re_rgb, cv.COLOR_BGR2RGB)
            pred_rgb_plt = cv.cvtColor(pred_rgb, cv.COLOR_BGR2RGB)
            diffMat_plt = cv.cvtColor(diffMat, cv.COLOR_BGR2RGB)
            plt.subplot(2, 2, 1).set_title('Current Image'), plt.imshow(curr_plt)
            plt.subplot(2, 2, 3).set_title('Differential Image'), plt.imshow(yDiff)
            plt.subplot(2, 2, 2).set_title('Predicted Image'), plt.imshow(pred_rgb_plt)
            plt.subplot(2, 2, 4).set_title('Motion Vectors'), plt.quiver(coordMat[0, :], coordMat[1, :], coordMat[2, :],
                                                                        coordMat[3, :])
            plt.savefig('result/train_'+str(k)+'.png')     
            plt.close()      

    print(encoded_file_size)
    i=0
    for frame_num in range(len(v_encoded_frames)):
        y_encoded, u_encoded, v_encoded, MV_encoded,MV_sub_array_encoded = y_encoded_frames[frame_num], u_encoded_frames[frame_num], v_encoded_frames[frame_num], MV_encoded_frames[frame_num],MV_sub_array_encoded_frames[frame_num] 
        if frame_num % 2 == 0:
            y_decoded, u_decoded, v_decoded = decode_I(height,width,y_encoded,u_encoded,v_encoded)
            re_rgb = YUV2RGB(y_decoded.astype(np.uint8),u_decoded.astype(np.uint8), v_decoded.astype(np.uint8), height, width)
            y_ref = y_decoded
            u_ref = u_decoded
            v_ref = v_decoded


        else:
            y_diff, u_diff, v_diff, MV_decoded,MV_sub_array_decoded = decode(height,width,y_encoded,u_encoded,v_encoded,MV_encoded,MV_sub_array_encoded)
            # yCurr, uCurr, vCurr = y_diff, u_diff, v_diff
            
            MV_decoded = np.reshape(MV_decoded, (2,int(len(MV_decoded)/2)))
            MV_sub_array_decoded = np.reshape(MV_sub_array_decoded, (2,int(len(MV_sub_array_decoded)/2)))
            mv_idx = 0
            y_Pred = np.zeros((height, width))
            u_Pred = np.zeros((height // 2, width // 2))
            v_Pred = np.zeros((height // 2, width // 2))
            for n, m in product(range(0, height, 16), range(0, width, 16)):
                y_Pred[n:n + 16, m:m + 16] = np.float32(y_ref[MV_decoded[0, mv_idx] +n:MV_decoded[0, mv_idx] +n + 16, MV_decoded[1, mv_idx] +m:MV_decoded[1, mv_idx] + m + 16])
                mv_idx += 1

            uv_idx = 0
            for i, j in product(range(0, (height // 2), 8), range(0, (width // 2), 8)):

                    ref_row = i + (MV_sub_array_decoded[0, uv_idx])
                    ref_col = j + (MV_sub_array_decoded[1, uv_idx])

                    u_Pred[i:i + 8, j:j + 8] = np.float32(u_ref[ref_row:ref_row + 8, ref_col:ref_col + 8])
                    v_Pred[i:i + 8, j:j + 8] = np.float32(v_ref[ref_row:ref_row + 8, ref_col:ref_col + 8])

                    uv_idx += 1


            yCurr = y_diff.astype(np.uint8) + y_Pred.astype(np.uint8)
            uCurr = u_diff.astype(np.uint8) + u_Pred.astype(np.uint8)
            vCurr = v_diff.astype(np.uint8) + v_Pred.astype(np.uint8)

            # i += 1
            re_rgb = YUV2RGB(yCurr.astype(np.uint8),uCurr.astype(np.uint8),vCurr.astype(np.uint8), height, width)


        video.write(re_rgb)  
    plt.title('PSNR per Frame')
    plt.ylim([50, 100])
    plt.plot(PSNR)
    plt.show()

if __name__ == '__main__':
    main()