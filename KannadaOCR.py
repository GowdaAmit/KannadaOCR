import cv2 as cv
import numpy as np
import logging
from PIL import Image
from scipy.signal import savgol_filter
import statistics as stats
from matplotlib import pyplot as plt
import itertools as iter
from multiprocessing import Pool, Queue
import multiprocessing as mp
import secrets
import numba

# module level variables ###
file_path = 'E:/AIProject/Python/CNN/data/'
_logger = None
#####

class KannadaScript(object):
    script_img = []
    script_attributes = []
    hu_points = None
    isSpace = False
    isfullStop = False
    isComma = False
    isScriptIdentified = False
    scriptUnicodes = []
    indexPos = 0

    def __call__(self):
        print('Space={}, fullstop={}, comma={}, index={}'.
              format(self.isSpace, self.isfullStop, self.isComma, self.indexPos))

    # _deerga1 = None
    # _deerga2 = None
    # _kombu = None
    # _ottu = None

    _unicodes = [['ಅ', '\u0c85'], ['ಆ', '\u0c86'], ['ಇ', '\u0c87'], ['ಈ', '\u0c88'], ['ಉ', '\u0c89'],
                 ['ಊ', '\u0c8A'], ['ಋ', '\u0c8B'], ['ೠ', '\u0cE0'], ['ಎ', '\u0c8E'], ['ಏ', '\u0c8F'],
                 ['ಒ', '\u0c92'], ['ಓ', '\u0c93'], ['ಐ', '\u0c90'], ['ಔ', '\u0c94'], ['ಕ', '\u0c95'],
                 ['ಖ', '\u0c96'], ['ಗ', '\u0c97'], ['ಘ', '\u0c98'], ['ಜ್ಞ', '\u0c99'], ['ಚ', '\u0c9A'],
                 ['ಛ', '\u0c9B'], ['ಜ', '\u0c9C'], ['ಝ', '\u0c9D'], ['ಞ', '\u0c9E'], ['ಟ', '\u0c9F'],
                 ['ಠ', '\u0cA0'], ['ಡ', '\u0cA1'], ['ಢ', '\u0cA2'], ['ಣ', '\u0cA3'], ['ತ', '\u0cA4'],
                 ['ಥ', '\u0cA5'], ['ದ', '\u0cA6'], ['ಧ', '\u0cA7'], ['ನ', '\u0cA8'], ['ಪ', '\u0cAA'],
                 ['ಫ', '\u0cAB'], ['ಬ', '\u0cAC'], ['ಭ', '\u0cAD'], ['ಮ', '\u0cAE'], ['ಯ', '\u0cAF'],
                 ['ರ', '\u0cB0'], ['ಲ', '\u0cB2'], ['ವ', '\u0cB5'], ['ಶ', '\u0cB6'], ['ಷ', '\u0cB7'],
                 ['ಸ', '\u0cB8'], ['ಹ', '\u0cB9'], ['ಳ', '\u0cB3'], ['ೀ', '\u0cc0'], ['ು', '\u0cc1'],
                 ['ೂ', '\u0cc2'], ['ೃ', '\u0cc3'], ['ೄ', '\u0cc4'], ['ೆ', '\u0cc6'], ['ೇ', '\u0cc7'],
                 ['ೈ', '\u0cc8'], ['ೕ', '\u0cD5'], ['ೖ', '\u0cD6'], ['ೊ', '\u0cCA'], ['ೋ', '\u0cCB'],
                 ['ೌ', '\u0cCC'], ['್', '\u0cCD'], ['ಾ', '\u0cBE'], ['ಿ', '\u0cBF'], ['಼,', '\u0cBC'],
                 ['ಂ', '\u0c82'], ['ಃ', '\u0c83'], ['೦', '\u0ce6'], ['೧', '\u0ce7'], ['೨', '\u0ce8'],
                 ['೩', '\u0ce9'], ['೪', '\u0ceA'], ['೫', '\u0ceB'], ['೬', '\u0ceC'], ['೭', '\u0ceD'],
                 ['೮', '\u0ceE'], ['೯', '\u0ceF']]

# end class

class KannadaScriptLine(object):
    _line_image = None
    _line_index = 0

    _sub_images = []

    def __init__(self, text_image=None, txtIdx=0):
        self._image = text_image
        self._pos_index = txtIdx

    # end def
# end class

def _initiatelogger():
    if _logger is not None:
        return _logger
    # end if

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('KannadaOCR')
    hdlr = logging.FileHandler(file_path + 'Kannada.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.WARNING)

    return logger

# end def


class KannadaFont(object):

    # predefined optimized functions to calculate img row and col means
    lambda_row_mean = lambda self, img: [np.mean(row) for row in img]
    lambda_col_mean = lambda self, img: [np.mean(img[:, i]) for i in range(img.shape[1])]
    lambda_rowpixcount_nz = lambda self, img: [cv.countNonZero(row) for row in img]

    _logger = None
    # store meanwidth of individual lines to create cutoff for splitting texts
    _lineMeanWidthcutOff = {}
    logging.basicConfig(level=logging.DEBUG)
    IDX, IMG = 0, 1

    def __init__(self, imageinput):
        self._imgfile = file_path + imageinput
        print('Initiating...')
        _logger = _initiatelogger()

    # end __init__

    # proc to smoothen jagged graph lines
    def smooth(self, x, window_len=11, window='hanning'):

        if window_len < 3:
            return x

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    # end def smooth

    # proc to find the maximum graph points to identify text
    def _findtextendpoints(self, peaks=[]):
        _elevation = peaks
        _xlength = len(_elevation)
        _point1 = _elevation[0]
        for _pt in _elevation:
            if _point1 > _pt: break
            _point1 = _pt
        # end for
        return _point1

    # end def

    def _saveimage2file(self, imgdet:[])-> None:
        _txtlineno, idx, _new_image = imgdet
        _outpath = file_path + 'outline/'

        _num = 0
        if type(_new_image) is dict:
            for _key in _new_image.keys():
                if type(_new_image[_key]) is list:
                    for _img in _new_image[_key]:
                        Image.fromarray(_img). \
                            save(_outpath + 'img_{}_{}_{}.jpg'.format(_txtlineno, idx, _num))
                        _num += 1
                    # end for
                else:
                    Image.fromarray(_new_image[_key]). \
                        save(_outpath + 'img_{}_{}_{}.jpg'.format(_txtlineno, idx, _num))
                    _num += 1
                # end if
            # end for
        elif type(_new_image) is list:
            for _img in _new_image:
                Image.fromarray(_img). \
                    save(_outpath + 'img_{}_{}_{}.jpg'.format(_txtlineno, idx, _num))
                _num += 1
            # end for
        else:
            Image.fromarray(_new_image).save(_outpath + 'img_{}_{}.jpg'.format(_txtlineno, idx))
        # end if

    # end def

    def _topLeftRightMargin(self, bwimg, rightmargin=False):

        img_ext = bwimg.copy()

        h, w = bwimg.shape
        _iPos = 0

        ##### left top margin and right margin
        if not rightmargin:
            x = np.nonzero(img_ext[0, :])[0]
        else:
            # reverse the array to handle right margin
            x = np.nonzero(img_ext[0, :][::-1])[0]
            _iPos = w - 1
        # end if

        _index = 0
        while _index < len(x) - 1:
            if np.abs(x[_index + 1] - x[_index]) > 1: break
            _index += 1
        # end while
        # print(_index)

        for i in np.arange(_index + 1):
            _loc = _iPos - i if _iPos > 0 else i
            x = np.nonzero(img_ext[:, _loc])[0]
            _index = 0
            while _index < len(x) - 1:
                if np.abs(x[_index + 1] - x[_index]) > 1: break
                _index += 1
            # end while
            # print(_index)
            img_ext[0:_index + 1, _loc] = 0
        # end for
        ##### left top margin end

        return img_ext

    # end def

    def _bottomleftrightmargins(self, bwimg, bottomright=False):

        # find the band width at the bottom

        img_ext = bwimg.copy()

        h, w = bwimg.shape
        _iPos = 0

        ##### left top margin and right margin
        if not bottomright:
            _r = np.nonzero(img_ext[h - 1, :])[0]
        else:
            _r = np.nonzero(img_ext[h - 1, :][::-1])[0]
            _iPos = w - 1
        # end if

        _index = 0
        while _index < len(_r) - 1:
            if np.abs(_r[_index + 1] - _r[_index]) > 1: break
            _index += 1
        # end while
        # print(_index)

        _colCount = _index
        # paint bottom border white
        for _col in np.arange(_colCount + 1):
            _loc = _iPos - _col if _iPos > 0 else _col

            _r = np.nonzero(img_ext[:, _loc])[0][::-1]

            _index = 0
            while _index < len(_r) - 1:
                if np.abs(_r[_index + 1] - _r[_index]) > 1: break
                _index += 1
            # end while
            # print(_loc, _index - 1, h - _index)
            img_ext[h - _index - 1:, _loc] = 0
        # end for
        return img_ext

    # end def _bottomleftrightmargins

    def _paintImageBorders(self, bwimg):

        img_ext = bwimg

        img_ext = self._topLeftRightMargin(img_ext)
        img_ext = self._topLeftRightMargin(img_ext, True)

        img_ext = self._bottomleftrightmargins(img_ext)
        img_ext = self._bottomleftrightmargins(img_ext, True)

        Image.fromarray(img_ext).save(file_path + 'new_ext1.jpg')

        return img_ext

    # end def _paintImageBorders

    def _rotateTiltedTextDocument(self, inputimage=[],
                                  displayImage=False,
                                  smoothOutImage=False,
                                  expandImage=False):

        img = inputimage

        if img is None:
            print('File Error!')
            return
        # end if

        edges = cv.Canny(img.copy(), 50, 150, apertureSize=3)

        lines = cv.HoughLines(edges, 1, np.pi / 360, 200)

        if lines is None:
            return inputimage, False
        # end if

        rotation_angle = 0

        img_t = img.copy()

        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # cv.line(img_t, (x1, y1), (x2, y2), (0, 0, 255), 2)
            rotation_angle = (np.arctan2(np.abs(y2 - y1), np.abs(x2 - x1)) * 180 / np.pi * np.sign(a) * -1)
        # end for

        # skip when the roatation angles are perpendicular or -90
        _rotated = False

        if np.abs(rotation_angle) >= 80:
            rotation_angle = (90 + rotation_angle) if np.sign(rotation_angle) == -1 else (90 - rotation_angle) * -1
        elif np.abs(rotation_angle) > 10 :
            rotation_angle = 0
        # end if

        print(rotation_angle)

        if rotation_angle != 0:

            img_t = np.bitwise_not(img)
            _, img_t = cv.threshold(img_t, 127, 255, cv.THRESH_BINARY)

            row, col = img.shape
            image_center = tuple(np.array((row, col)) // 2)
            rot_mat = cv.getRotationMatrix2D(image_center, rotation_angle, 1.0)
            img_t = cv.warpAffine(img_t, rot_mat, (col, row), flags=cv.INTER_NEAREST)
            img_t = np.bitwise_not(img_t)

            if (displayImage):
                cv.imshow("Rotated", img_t)
                cv.waitKey(-1)
                cv.destroyAllWindows()
            # end if
            _rotated = True
        # end for

        imgExpand = img_t

        if _rotated:

            outPutType = cv.INTER_LINEAR
            if smoothOutImage:
                outPutType = cv.INTER_CUBIC
            # end if

            if expandImage:
                imgExpand = cv.resize(img_t, None, fx=2, fy=2, interpolation=outPutType)  # increase image size
            # end if

            imgExpand = self._paintImageBorders(imgExpand)

        # end if

        return imgExpand, _rotated

    # end of _rotateTiltedTextDocument

    def _getCutOffIndex(self, _spoints=[]):

        _indexPos = 0
        _minPos = 0
        _prevMin = _spoints[0]

        while True:

            _minIndex = np.argmin(_spoints[_indexPos:_indexPos + 5])
            _minValue = _spoints[_indexPos:_indexPos + 5][_minIndex]

            if _minValue > _prevMin: break
            _prevMin = _minValue

            _indexPos += 5

            if _indexPos > len(_spoints) // 2: break
        # end while

        # check for empty space margin
        while stats.variance(_spoints[_indexPos:_indexPos + 20]) < 0.75:
            _indexPos += 20
        # end while
        # print(_indexPos)

        return _indexPos

    # end def _getCutOffIndex

    def _cropImage(self, img_ext):

        # crop top and bottom margin
        # _margins = [np.mean(row) for row in img_ext]
        _margins = self.lambda_row_mean(img_ext)
        _margins = savgol_filter(_margins, 51, 3)

        _topMargin = self._getCutOffIndex(_margins)
        _bottomMargin = self._getCutOffIndex(_margins[::-1])
        _bottomMargin = img_ext.shape[0] - _bottomMargin
        img_ext = img_ext[_topMargin:_bottomMargin, :]

        # ext_points = []
        # for i in np.arange(img_ext.shape[1]):
        #     ext_points.append(img_ext[:, i].mean())
        # # end for
        ext_points = self.lambda_col_mean(img_ext)
        ext_points = savgol_filter(ext_points, 51, 3)

        # crop left and right margin
        _leftMargin = self._getCutOffIndex(ext_points)
        _rightMargin = self._getCutOffIndex(ext_points[::-1])
        _rightMargin = len(ext_points) - _rightMargin
        img_ext = img_ext[:, _leftMargin:_rightMargin]

        # print(img_ext.shape)

        return img_ext

    # end def _cropImage(self, img_ext)

    def _gettextstrip(self, bwimg):

        _indexList = self._extractTextLines(bwimg)
        img_ext = bwimg

        _linePos = 0
        # take 1/4 extra height to cover the entire gamut
        _margin = (_indexList[3] - _indexList[2]) // 3
        _textLines = []
        _prevlinepos = 0
        for _indexpos in _indexList:
            _text_line = img_ext[_prevlinepos:_indexpos + _margin, :]
            _, _text_line = cv.threshold(_text_line, 127, 255, cv.THRESH_BINARY)
            _textLines.append(_text_line)
            _prevlinepos = _indexpos
            Image.fromarray(_text_line).save(file_path + 'line_{}.jpg'.format(_linePos))
            _linePos += 1
        # endfor

        # extract the last line of the text document
        if _indexpos < img_ext.shape[0]:
            _text_line = img_ext[_indexpos:, :]
            _, _text_line = cv.threshold(_text_line, 127, 255, cv.THRESH_BINARY)
            _textLines.append(_text_line)
            Image.fromarray(_text_line).save(file_path + 'line_{}.jpg'.format(_linePos))
        # end if


        return _textLines

    # end def _getTextPos

    # after image cropping and rotation, extract text lines to process them independently
    def _extractTextLines(self, bwimg):

        img_ext = bwimg

        # ext_points = [cv.countNonZero(row) for row in img_ext]
        ext_points = self.lambda_rowpixcount_nz(img_ext)
        ext_points = self.smooth(ext_points, 11, 'hanning')

        # reverse the data array to find the space between text lines
        _apex = ext_points[np.argmax(ext_points)]
        _low_points = np.array([int(_apex - row) for row in ext_points])
        _low_points[_low_points < max(_low_points) * 0.50] = 0

        # plt.plot(_low_points)
        # plt.show()

        # _low_points = ext_points
        _zerosList = list(_low_points)
        _gapList = []
        _peakValue = 0

        try:
            _startIndex = 0
            _lastIndex = 0
            while True:
                _startIndex = _zerosList[_startIndex:].index(0)
                _indexpos = _startIndex + _lastIndex
                _lastIndex = _startIndex + _lastIndex
                while (_zerosList[_lastIndex:_lastIndex + 1][0] == 0):
                    _lastIndex += 1
                    if _lastIndex >= len(_low_points): break
                # end while
                # print(_lastIndex - _indexpos)
                _gapList.append([_indexpos, _lastIndex])
                _startIndex = _lastIndex
            # end while

        except ValueError:
            print("Exception")

        _rowindex = 0
        _spaceWidth = []
        # calculate the gap between lines to remove redundant lines where the gap is less than average
        for _rowindex in np.arange(1, len(_gapList)):
            _spaceWidth.append([_rowindex,
                                _gapList[_rowindex][1],
                                _gapList[_rowindex - 1][1],
                                _gapList[_rowindex][1] - _gapList[_rowindex - 1][1]])
        # end for

        _temp = _spaceWidth.copy()
        _temp.sort(key=lambda _textGap: _textGap[3], reverse=True)
        # remove top skewed entries that may represent large space between text lines
        _startPos, _endPos = len(_temp) // 4, len(_temp) // 2
        _textWidth = np.mean(np.array(_temp[_startPos: _endPos])[:, 3])

        _newList = []
        _index = 0
        _temp = _temp[::-1]
        for _pt in _temp:
            if _pt[3] / _textWidth < 0.50:
                _index += 1
                continue
            # end if
            break
        # end for

        _spaceWidth = _temp[_index:]
        # restore the line order by row index
        _spaceWidth.sort(key=lambda row: row[0])
        del _temp

        marked_text = img_ext.copy()
        for _pt in _spaceWidth:
            cv.line(marked_text, (0, _pt[2]), (img_ext.shape[1], _pt[2]), 255, 1)
        # end for
        Image.fromarray(marked_text).save(file_path + 'marked_text.jpg')
        del marked_text
        del _gapList

        # return only the index positions to text line extraction
        return [_row[2] for _row in _spaceWidth]

    # end def

    def _repaintborderlinetext(self, _lineimg):

        # top border clearing
        # ext_points = [np.mean(row) for row in _lineimg]
        ext_points = self.lambda_row_mean(_lineimg)
        _lowpoint = np.argmin(ext_points[:int(len(ext_points) * 0.3)])
        _index = 0
        for _col in range(_lineimg.shape[1]):
            _lineimg[0:_lowpoint, _col] = 0
            _index += 1
        # end for

        return _lineimg

    # end def

    def _scrapetextline(self, _txtLines):
        # find most common/maximum occcurances textline height
        _hist = np.histogram([row.shape[0] for row in _txtLines])
        _maxCountIndx = np.argmax(_hist[0])
        _meanHeight = _hist[1][_maxCountIndx]

        _index = 0
        _newtxtlines = []
        for _line in _txtLines:
            # if text height more than 1.5 times the average height
            _newsplitimages = []
            self._splitline(_line, _meanHeight, _index, _newsplitimages)
            for _img in _newsplitimages:
                _newtxtlines.append(_img)
            # end if
            _index += 1
        # end for

        _txtLines = _newtxtlines

        _index = 0
        for _line in _newtxtlines:

            _line_image = _line

            # code block to remove excessive blank lines at the bottom margin
            # that does not exceed beyond bottom 20% of space
            _startIndex = _line_image.shape[0] - 1
            while np.count_nonzero(_line_image[_startIndex, :]) != 0:
                _startIndex -= 1
                if _startIndex == 0: break
            # end while

            if _startIndex >= _line_image.shape[0] * 0.8:
                _line_image = _line_image[:_startIndex + 1, :]
            # end if
            logging.debug('removed {} lines from bottom of {}'.
                          format(_line_image.shape[0] - _startIndex, _index))

            # code block to remove excessive blank lines at the top margin
            _lineIndex = 0
            for _row in range(_line_image.shape[1]):
                if _line_image[_row, :].mean() == 0:
                    _lineIndex += 1
                    continue
                # end if
                break
            # end for

            if _lineIndex > 0:
                _line_image = _line_image[_lineIndex:, :]
            # end if

            self._repaintborderlinetext(_line_image)

            # remove blank lines at the left margin
            _margin_index = self._removeblanklines(_line_image)
            if _margin_index > 0:
                _line_image = _line_image[:, _margin_index:]
            # end if

            _newtxtlines[_index] = _line_image
            Image.fromarray(_line_image).save(file_path + 'outline_{}.jpg'.format(_index))
            _index += 1
        # end for

        return _newtxtlines

    # end def

    # a recursive code to split joined lines
    def _splitline(self, _line, _meanHeight, _index, _newImages):
        if _line.shape[0] / _meanHeight > 1.5:
            # ext_points = [np.mean(row) for row in _line]
            ext_points = self.lambda_row_mean(_line)
            _height = len(ext_points)
            _cutOffPoint = np.argmin(ext_points[int(_height * .25):int(_height * .75)]) + int(_height * .25)
            img1 = _line[:_cutOffPoint, :]
            img2 = _line[_cutOffPoint:, :]
            self._splitline(img1, _meanHeight, _index, _newImages)
            self._splitline(img2, _meanHeight, _index, _newImages)
        else:
            # ext_points = [np.mean(_line[:, i]) for i in range(_line.shape[1])]
            ext_points = self.lambda_col_mean(_line)
            if stats.variance(ext_points) > 50:
                _newImages.append(_line)
        # end if

    # end def

    def _removeblanklines(self, img):
        # code block to remove excessive blank lines at the left margin
        _lines = 0

        for _col in range(img.shape[1]):
            if cv.countNonZero(img[:, _col]) == 0:
                _lines += 1
                continue
            # end if
            break
        # end for

        return _lines

    # end def
    def _splitscript(self, _imgdet, _lineno=0, _reprocess=False) -> {}:
        """
        This function helps to extract individual texts from a line of text

        :param _imgdet:
        :return:
        """

        _parentid, lineimg = _imgdet

        img = lineimg
        # print(np.histogram(img))
        # ignore the right end point >90% in order to get the lowest points in the midrange
        # start from 10% onwards to remove outlier effect
        ext_points = self.lambda_col_mean(img[:, :int(img.shape[1] * 0.9)])

        idx = 0
        _maxpt = max(ext_points[5:]) // 2
        while ext_points[idx] == 0:
            ext_points[idx] = _maxpt
            idx +=1

        ext_points = ext_points - min(ext_points[5:])
        # if _reprocess:
        #     ext_points = ext_points[5:]
        # # end if

        _minimas = []
        _indexPos = 0
        while _indexPos < len(ext_points) - 5:
            _mins = np.argmin(ext_points[_indexPos:_indexPos + 10])
            if 0 <= ext_points[_indexPos:_indexPos + 10][_mins] < 2:
                # print(stats.variance(ext_points[_indexPos:_indexPos+10]))
                _minimas.append(_indexPos + _mins)
            _indexPos += 5
        # end for

        # print(_minimas)
        _uniquepts = sorted(set(_minimas))
        # print(_uniquepts)

        _startIndex = 0
        _textHit = False
        _scripts = []
        _last_img_idx = 0
        _prev_img_width = 0

        _imagesDict = {}

        for _last_idx, _cell in enumerate(_uniquepts):

            # process this only for the top level
            if not _reprocess:
                if cv.countNonZero(img[:, _startIndex:_cell]) == 0:
                    if _textHit:
                        _script = KannadaScript()
                        _script.isSpace = True
                        _script.indexPos = _last_idx
                        _scripts.append(_script)
                    # end if
                    _textHit = False
                    continue
                # end if
            # end if

            _newimg = img[:, _startIndex:_cell + 1]
            _index = self._removeblanklines(_newimg)

            # allow one pix space between border and text
            # when two texts are connected there should be no space between connecting
            # points, otherwise the space between texts indicate disjointed texts
            if _index - 1 > 0:
                _newimg = _newimg[:, _index - 1:]
            # end if

            if np.count_nonzero(_newimg) > 0:
                # generate a unique ID for child text
                _imagesDict[secrets.token_hex(2)] = [_last_idx, _newimg]
            # end if
            _last_img_idx = _last_idx
            _startIndex = _cell + 1
            _textHit = True
        # end for

        # store the last piece
        if _startIndex > 0:
            _newimg = img[:, _startIndex:]
            _index = self._removeblanklines(_newimg)
            if _index - 1 > 0:
                _newimg = _newimg[:, _index - 1:]
            # end if

            # if not an empty array , append in the end of the list
            if np.count_nonzero(_newimg) > 0:
                _imagesDict[secrets.token_hex(2)] = [_last_img_idx + 1, _newimg]
            # end if
        #end if

        # process only for first pass
        if not _reprocess:
            _holdBay = []
            # remove iamges with noise when the data variance is < 25% of max variance
            _m = max([np.var(_imagesDict[_key][1]) for _key in _imagesDict.keys()])
            _remKeys = [_key for _key in _imagesDict.keys() if np.var(_imagesDict[_key][1]) < _m * .25]

            _temp = [(idx, _key) for idx, _key in enumerate(_imagesDict.keys()) if _key in _remKeys]
            for i in range(1, len(_temp)):
                if _temp[i][0] - _temp[i - 1][0] > 1:
                    _holdBay.append(_temp[i][1])
                # end if
            # end for

            [_imagesDict.__delitem__(key) for key in _remKeys if key not in _holdBay]
        # end if

        self._reprocessText(_parentid, _lineno, _imagesDict)

        return _imagesDict

    # end def

    # find the oversize and unprocessed text that are clumped > 1 text
    def _reprocessText(self, _parentid='', _lineno=0, imgsDict={}):

        # try:

        _mwCutOff = 1.0
        _outbasket = []
        _meantextsize = sorted([img.shape[1] for _, img in imgsDict.values()])

        _halflength = len(_meantextsize)
        # the truncated mean is calculated only there are more than 4 text elements
        if _halflength > 4:
            _halflength = int(_halflength * 0.75)
            _mwCutOff = 1.75
        # end if

        if _halflength == 0:
            return  # False, imgsDict
        # end if

        _meanwidth = np.mean(_meantextsize[0:_halflength])
        del _meantextsize, _halflength

        logging.debug("Meanwidth<---->{}".format(_meanwidth))

        # store meanwidth to refer in deeper iterations
        if _lineno not in self._lineMeanWidthcutOff.keys():
            self._lineMeanWidthcutOff[_lineno] = _meanwidth
        # end ifimgs[0][1]

        # this code is useful in deeper nested processing that prevents
        # text from splitting too thin
        if _meanwidth < self._lineMeanWidthcutOff[_lineno]:
            logging.debug("too small size of image to process....{}, {}".format(
                int(self._lineMeanWidthcutOff[_lineno]), _meanwidth))
            return  # False, imgsDict
        # end if

        IMGIDX, IMG, WIDTH = 1, 1, 1
        # store the location of the large image that to be replaced with broken images
        # generate parentID in order to recognize children
        _largeimges = [[_key, imgsDict[_key][IMG]]
                       for _key in imgsDict.keys() if imgsDict[_key][IMG].shape[WIDTH] > _meanwidth * _mwCutOff]

        if len(_largeimges) == 0:
            logging.debug('No large images found...')
            return  # False, imgsDict
        # end if

        _parents2remove = []
        # store the index the image to delete and sub images
        for _key, img in _largeimges:

            # skip blank images
            ext_points = self.lambda_col_mean(img[:int(img.shape[0] * 0.9), :])
            if sum(ext_points) == 0:
                continue
            # end if

            logging.debug('Processing.. -->{}, {} images'.format(_key, img.shape))

            _splitMidDict = self._splitscript([_key, img], _lineno, _reprocess=True)
            if _key in imgsDict.keys():
                imgsDict[_key].append(_splitMidDict)
            else:
                imgsDict[_key] = _splitMidDict
            # end if

        # end for
        del _largeimges

        # except ValueError:
        #     logging.debug('Value Error...')

    # end def _reprocessText

    def _procTextTree(self, _itemsDict:{}) -> {}:
        """
        Extract images to patchup and put line text in a proper order

        :param _itemsDict:
        :return:
        """

        _parentImges = {}
        _lineIndexNo = 0

        for _itemKey in _itemsDict.keys():
            _children = _itemsDict[_itemKey]
            _foundChildren = False

            for _splititem in _children:
                if type(_splititem) is dict:
                    _foundChildren = True
                    _outImges = self._procTextTree(_splititem)
                    del (_parentImges[_lineIndexNo])
                else:
                    _outImges = _splititem
                    if type(_splititem) is int:
                        _lineIndexNo = _splititem
                        continue
                    # end if
                # end if
                if _lineIndexNo in _parentImges.keys():
                    _parentImges[_lineIndexNo].append(_outImges)
                else:
                    # store the image as a list element
                    _parentImges[_lineIndexNo] = _outImges
                # end if

            # end for
        # end for
        return _parentImges

    # end def

    # procedure to flatten hierarchical dictionaries and fuse fragmented text
    # fuse the texts if they are connected originally
    def _reorgTextList(self, _semiFinishedTextImgs:{} )-> {}:

        _imageList = {}
        _dicFound = False
        _keyUpdated = 0

        for _key in _semiFinishedTextImgs.keys():
            _items = _semiFinishedTextImgs[_key]
            if type(_items) is dict:

                _fusedText = []
                _index = 0

                for _subkey in _items.keys():
                    if type(_items[_subkey]) is dict:
                        _dicFound = True
                        _imagePatch = self._reorgTextList(_items[_subkey])
                        if len(_imagePatch) > 0:
                            _fusedText = self._patchText(_imagePatch)
                        # end if
                        if len(_fusedText) > 0:
                            # del (_semiFinishedTextImgs[_key])
                            _items[_subkey] = _fusedText[0] if len(_fusedText) == 1 else _fusedText
                        # end if
                    # elif type(_items[_subkey]) is list:
                    #     _fusedText = self._patchText(_items[_subkey])
                    #     _items[_subkey] = _fusedText
                    else:
                        _imageList[_index] = _items[_subkey]
                        _index += 1
                        _keyUpdated = _key
                    # end if
                # end for
            elif type(_semiFinishedTextImgs[_key]) is list:
                _imageList = _items
            # end if
        # end for

        if not _dicFound and len(_imageList) > 0:
            _fusedText = self._patchText(_imageList)
            _semiFinishedTextImgs[_keyUpdated] = \
                _fusedText[0] if len(_fusedText) == 1 else _fusedText
        # end if

        return _semiFinishedTextImgs

    # end def

    def _flattenDict(self, _topDict={}):

        _newList = []
        for _key in _topDict.keys():
            if type(_topDict[_key]) is dict:
                _images = self._flattenDict(_topDict[_key])
                if len(_images) > 1:
                    [_newList.append(_img) for _img in _images]
                # end if
            elif type(_topDict[_key]) is list:
                [_newList.append(_img) for _img in _topDict[_key]]
            else:
                _newList.append(_topDict[_key])
            # end if
        # end for
        return _newList

    # end def

    def _patchText(self, _patches=[]):
        """
        if there found a bridge connecting two images join those images
        """
        # Execute this function only if there are dictionary items in the data
        _imgList = _patches
        _objTypes = [type(_item) for _item in _patches.values()]
        if dict in _objTypes or list in _objTypes :
            _imgList = self._flattenDict(_patches)
        # end if

        _tmpimages = []

        _firstImage = _imgList[0]

        _, _pimg = cv.threshold(_firstImage, 127, 255, cv.THRESH_BINARY)
        for _key in range(1, len(_imgList)):

            if type(_imgList[_key]) is list:
                _patchedImg = self._patchText(_imgList[_key])
            # end if

            _, _cimg = cv.threshold(_imgList[_key], 127, 255, cv.THRESH_BINARY)
            if np.count_nonzero(_pimg[:, -1] & _cimg[:, 0]) > 0:
                _pimg = np.hstack((_pimg, _cimg))
                _ispadded = True
            else:
                # store the joined image
                _tmpimages.append(_pimg)
                _, _pimg = cv.threshold(_imgList[_key], 127, 255, cv.THRESH_BINARY)
                _ispadded = False
            # end if
        # end for
        _tmpimages.append(_pimg)

        return _tmpimages

    # end def

    ###################
    # the func to find similar letters and number in a document
    ###################

    def _similarImage(self, pImg, cImg):
        _dist = lambda src, dst: sum((src - dst) ** 2) / len(src)
        _mom1 = cv.HuMoments(cv.moments(pImg))
        _mom2 = cv.HuMoments(cv.moments(cImg))
        _mom1 = -1 * np.sign(_mom1) * np.log(np.abs(_mom1))
        _mom2 = -1 * np.sign(_mom2) * np.log(np.abs(_mom2))

        _dist = _dist(_mom1, _mom2)
        return True if _dist < 100 else False

    # end def


    def processImage(self, _pool):

        img = cv.imread(self._imgfile, 0)
        img = np.bitwise_not(img)

        bw, _rotated = self._rotateTiltedTextDocument(img, displayImage=False)

        # the image is thresholded if rotated, so do not perform another threshold
        if not _rotated:
            _, bw = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        # end if rotated

        part_img = self._cropImage(bw)

        # Image.fromarray(part_img).save(file_path + 'cropped.jpg')
        # rotated_img = part_img

        rotated_img, _rotated = self._rotateTiltedTextDocument(part_img)

        _textlinesList = self._gettextstrip(rotated_img)

        # Image.fromarray(rotated_img).save(file_path + 'rotated.jpg')
        # imgShrunk = cv.resize(rotated_img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)

        _textlinesList = self._scrapetextline(_textlinesList)

        # reset if there are remenents from earlier processing
        self._lineMeanWidthcutOff.clear()

        # _lineno = 4
        # img = _textlinesList[_lineno]
        for _lineno, img in enumerate(_textlinesList):

            # if _lineno !=19: continue

            _parentid = secrets.token_hex(3)
            _textDictList = self._splitscript([_parentid, img], _lineno)
            # _textobjectList = _pool.map(self._splitscript, enumerate(_textlinesList))

            _semiFinishedTextImgs = self._procTextTree(_textDictList)
            self._reorgTextList(_semiFinishedTextImgs)

            for _key in _semiFinishedTextImgs.keys():
                self._saveimage2file([_lineno, _key, _semiFinishedTextImgs[_key]])
            # end for
        # end for

        return _semiFinishedTextImgs

    # end processImage def

# end class

if __name__ == "__main__":
    _pool = Pool(processes=mp.cpu_count())
    kannadaClass = KannadaFont('sai_391.jpg')
    kannadaClass.processImage(_pool)
# end if