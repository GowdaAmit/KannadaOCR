import KannadaOCR as kannada
import pickle
import numpy as np
import cv2 as cv
from PIL import Image
import ExtractScriptData

X, Y, W, H, IMG_IDX = 0, 1, 2, 3, 4


class PreProcess(object):

    def __init__(self, errorlogger=None):
        _logger = errorlogger
        print('Initiating')

    # Variables
    DISP_GAP = 10  # pixels
    IMAGE_WIDTH = ExtractScriptData.RESIZED_IMAGE_WIDTH
    IMAGE_HEIGHT = ExtractScriptData.RESIZED_IMAGE_HEIGHT
    TOP_GUTTER = 5  # Pixels
    MIN_OVERLAP = 5  # the minimum overall gap
    _SPACE_WIDTH = 8

    SAVE_IMAGE_FLAG = True
    IMAGE_FILE_SAVE_PATH = 'E:/AIProject/Python/CNN/data/'

    _logger = None
    # _reversedtext = []
    _childrentoremove = []
    _spacesPos = []

    ####

    # TODO: to workout logic in this block
    def trial():

        img_bgr = cv.imread("/home/anmol/Downloads/tWuTW.jpg")
        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV_FULL)

        # Filter out low saturation values, which means gray-scale pixels(majorly in background)
        bgd_mask = cv.inRange(img_hsv, np.array([0, 0, 0]), np.array([255, 30, 255]))

        # Get a mask for pitch black pixel values
        black_pixels_mask = cv.inRange(img_bgr, np.array([0, 0, 0]), np.array([70, 70, 70]))

        # Get the mask for extreme white pixels.
        white_pixels_mask = cv.inRange(img_bgr, np.array([230, 230, 230]), np.array([255, 255, 255]))

        final_mask = cv.max(bgd_mask, black_pixels_mask)
        final_mask = cv.min(final_mask, ~white_pixels_mask)
        final_mask = ~final_mask

        final_mask = cv.erode(final_mask, np.ones((3, 3), dtype=np.uint8))
        final_mask = cv.dilate(final_mask, np.ones((5, 5), dtype=np.uint8))

        # Now you can finally find contours.
        im, contours, hierarchy = cv.findContours(final_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        final_contours = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 2000:
                final_contours.append(contour)

        for i in xrange(len(final_contours)):
            img_bgr = cv.drawContours(img_bgr, final_contours, i, np.array([50, 250, 50]), 4)

        debug_img = img_bgr
        debug_img = cv.resize(debug_img, None, fx=0.3, fy=0.3)
        cv.imwrite("./out.png", debug_img)

    def _iskMirrorImage(self, img1, img2):
        # check and chose the template which is smaller of the two images
        _temp = img1
        _parent = img2

        if img1.shape > img2.shape:
            _temp = img2
            _parent = img1
        # end if

        match = cv.matchTemplate(_parent, np.flip(_temp, 1), cv.TM_CCOEFF_NORMED)
        _, _confidence, _, _ = cv.minMaxLoc(match)
        if _confidence >= 70:
            print('Mirror Image found {}'.format(_confidence))
            return True
        # end if

        return False

    # end def

    def _split_glued_chars(self, tpts=[], troi=[]):

        print("Images to skip...")

        if len(troi) == 0:
            raise Exception('The Image List is Empty!')

        _stats = np.array([img_row.shape[1] for img_row in troi])
        minimum_image_width = np.int(_stats.mean() - _stats.std() // 2)
        del _stats

        # minimum_image_width = np.int(np.mean(img_to_process, axis=0)[0] - np.std(img_to_process, axis=0)[0] // 2.0)
        print("Image mininmum width criteria {}".format(minimum_image_width))

        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        x_pos = []
        img_idx_pos = 0
        for row_img in troi:

            image_width = row_img.shape[1]

            if image_width < minimum_image_width:
                img_idx_pos += 1
                continue

            eroded = cv.erode(row_img, kernel)
            # _saveImage2File(eroded, img_idx_pos, 0, 9)

            # prev_column = 0
            _startpos = np.int(eroded.shape[1] * 0.20)
            _endpos = np.int(eroded.shape[1] * 0.80)
            # consider only columns that are falling in the range between 25% and 75% of the image
            for column in np.arange(_startpos, _endpos + 1):
                if cv.countNonZero(eroded[:, column]) == 0:
                    # break column continuity, capture only one of continuous columns
                    # if the image width falls below minimum criterial, do not consider for splitting
                    # print("Split at Index {} with column {} != width {} for {} ".
                    #      format(img_to_process[i][2], column, image_width, column != image_width and column != 0))
                    x_pos.append(column)
                # end if
            # end for

            if len(x_pos) > 0:

                # find gaps between sub images
                _prev_x = x_pos[0]
                _step = []
                _master = []

                for x in x_pos:
                    if x - _prev_x <= 1:
                        _step.append(x)
                    else:
                        if (len(_step)) == 0:
                            _step.append(x)
                        _master.append(_step)
                        _step = []
                        _step.append(x)
                    # end if
                    _prev_x = x
                # end for

                if (len(_step)) > 0:
                    _master.append(_step)
                # end if

                # get the starting point of image boundary
                _imageList = []
                _image_strtpos = 0

                for _idx in np.arange(0, len(_master)):
                    _image_endpos = _master[_idx][0]

                    # if the image width is less than 10 units combine that portion with next image
                    if (_image_endpos - _image_strtpos) < 10:
                        continue

                    _tmp_img = row_img[:, _image_strtpos:_image_endpos]
                    if cv.countNonZero(_tmp_img) != 0:
                        _imageList.append(_tmp_img)
                    # end if
                    _image_strtpos = _master[_idx][-1]
                # end for

                if row_img.shape[1] > _image_strtpos:
                    _tmp_img = row_img[:, _image_strtpos:]
                    if cv.countNonZero(_tmp_img) != 0:
                        _imageList.append(_tmp_img)
                    # end if
                # end if

                _skipsplit = False
                if len(_imageList) == 2:
                    _img1 = _imageList[0]
                    _img2 = _imageList[1]
                    _skipsplit = self._iskMirrorImage(_img1, _img2)
                elif len(_imageList) == 1:
                    _skipsplit = True
                # end if

                # overwrite the fused image with list of divided images

                if _skipsplit == False:
                    troi[img_idx_pos] = _imageList
                    # print("Split Image {} at {}".format(image_width, img_idx_pos))
                # endif
            # end if
            img_idx_pos += 1
            x_pos = []
        # end for

        _tpts, _troi = self._reoorglineaftersplit(tpts, troi)

        return _tpts, _troi

    # end def

    def _reoorglineaftersplit(self, tpts=[], troi=[]):
        _n_troi = []
        _n_tpts = []
        _img_idx_pos = 0
        _org_idx = 0

        for _row_img in troi:
            if type(_row_img) is list:
                _start_x = tpts[_org_idx][X]
                for _img in _row_img:
                    _tpts_pos = [_start_x, tpts[_org_idx][Y], _img.shape[1], tpts[_org_idx][H], _img_idx_pos]

                    _n_tpts.append(_tpts_pos)
                    _n_troi.append(_img)
                    _img_idx_pos += 1
                    _start_x += _img.shape[1]
                # end for
            else:
                tpts[_org_idx][IMG_IDX] = _img_idx_pos
                _n_tpts.append(tpts[_org_idx])
                _n_troi.append(troi[_org_idx])
                _img_idx_pos += 1
            # end if list
            _org_idx += 1
        # end for

        _n_tpts.sort(key=lambda x: x[0])
        return _n_tpts, _n_troi

    # end def

    def _checkParentChildPosition(self, tpts=[], troi=[]):
        row_idx = 0
        _cond_found = False
        for tpts_row in tpts:
            # check if child image is larger than parent image and both

            if row_idx + 1 == len(tpts):
                break
            # endif

            intX1, intY1, intW1, intH1, img_idx1 = tpts_row
            intX2, intY2, intW2, intH2, img_idx2 = tpts[row_idx + 1]

            # starts with same -x-position, then interchange parent and child
            if intX1 == intX2 and intW2 > intW1:
                # store the parent-child index to reverse the sequence after pre-processing
                # _reversedtext.append([row_idx, row_idx + 1])
                tpts[row_idx] = [intX2, intY2, intW2, intH2, img_idx2]
                tpts[row_idx + 1] = [intX1, intY1, intW1, intH1, img_idx1]
                _cond_found = True
                print('Exchanged {} and {} '.format(row_idx, row_idx + 1))
            # end if
            row_idx += 1
        # end for

        # recursive to find no more parent child anamoly
        if _cond_found:
            tpts, troi = self._checkParentChildPosition(tpts, troi)
        # end if

        return tpts, troi

    # end def

    def _check_pip(self, tpts=[], troi=[]):

        if len(tpts) == 0:
            raise Exception("Check pip: invalid tpts list")
        # end if

        # changes tpts
        tpts, troi = self._checkParentChildPosition(tpts, troi)

        # fused_list = []
        index_dict = {}
        parent_idx = 0

        textlength = len(tpts)
        while parent_idx < (textlength - 1):

            intX1, intY1, intW1, intH1, _ = tpts[parent_idx]
            child_idx = parent_idx + 1
            intX2, intY2, intW2, intH2, _ = tpts[child_idx]

            # check if the boundaries are lying within the larger roi
            if np.abs(intX2 - intX1) < intW1:
                # print("pip found in Log 1..@ {} ".format(parent_idx))

                parent_size = intX1 + intW1  # parent width
                child_size = intX2 + intW2  # child width

                # check whether the smaller roi falls falls completely within the bigger image
                # and differs < 5 unints on both the ends
                if np.abs(intX2 - intX1) < 5 and np.abs(parent_size - child_size) < 5:
                    self._childrentoremove.append(child_idx)
                    parent_idx += 2
                    continue

                # find how many othher roi's are found linked with bigger roi
                sub_list = []
                _start_idx_child = child_idx
                while True:
                    child_bound = tpts[child_idx]

                    if parent_size - child_bound[X] > 0:
                        # print(" @ child idx {} overlap Size {}".format(child_idx, parent_size - child_bound[X]))
                        # if the overlap width is less than 10 units , do not store the child image index
                        if parent_size - child_bound[X] > 10:
                            # store both parent and child index ids
                            sub_list.append(child_idx)
                        # else:
                        #    print("----Skipped image index @ {} due to <= 10 width".format(child_idx))
                        # endif
                        child_idx += 1
                        if child_idx >= textlength:
                            break
                        # endif
                    else:
                        if _start_idx_child < child_idx:
                            child_idx -= 1
                        # end if
                        break
                    # endif
                # end while

                # add to list if there are overlaps
                if len(sub_list) > 0:
                    index_dict[parent_idx] = sub_list
                # end if
                parent_idx = child_idx
            else:
                parent_idx += 1
            # end if
        # end while

        print('Duplicate Images to remove {}'.format(self._childrentoremove))

        return tpts, troi, index_dict

    # enf def

    def _get_coords(self, x, y, _image, idx, ischild=True):
        _new_X, _new_Y = x, y
        _new_W, _new_H = _image.shape[1], _image.shape[0]
        _new_cords = [_new_X, _new_Y, _new_W, _new_H, idx, 'C' if ischild else 'P']
        return _new_cords

    # end def

    def _saveImage2File(self, _new_image, parent, child, proid):
        Image.fromarray(_new_image).save(self.IMAGE_FILE_SAVE_PATH + 'IMG_{}_{}_{}.jpg'.format(parent, child, proid))
    # end def

    def _process_pip(self, tpts=[], troi=[], index_dict={}):
        print('* * * * Process PIP....')

        _parentstochange = {}

        if len(index_dict) > 0:
            for p_pos in index_dict:
                new_child_img = []

                parent_bound = tpts[p_pos]
                parent_img_idx = parent_bound[IMG_IDX]
                parent_image = troi[parent_img_idx]
                remaining_p_image = parent_image
                parent_extension = parent_bound[X] + parent_bound[W]

                new_x, new_width = 0, 0

                children = index_dict[p_pos]
                print('children are {} @ {}'.format(children, p_pos))
                for child in children:

                    print('Child is {}'.format(child))

                    child_bound = tpts[child]
                    child_img_idx = child_bound[IMG_IDX]
                    child_extension = child_bound[X] + child_bound[W]
                    child_image = troi[child_img_idx]

                    # due to multiple overlaps over if parent is emptied store the child directly
                    if cv.countNonZero(remaining_p_image) == 0:
                        _new_coords = self._get_coords(child_bound[X], parent_bound[Y], child_image, p_pos, True)
                        print('@02 New Imaqe:', child_image.shape, _new_coords)
                        new_child_img.append([_new_coords, child_image.copy()])
                        if self.SAVE_IMAGE_FLAG:
                            self._saveImage2File(child_image, p_pos, child, 5)
                        # end if
                        continue
                    # end if

                    # child bounds extending beyond parent's
                    if np.abs(child_bound[X] - parent_bound[X]) < parent_bound[
                        W] and child_extension > parent_extension:

                        # if the parent image and child image are not different much at the end
                        _last_mile_gap = child_extension - parent_extension
                        # print('Last mile gap is:', _last_mile_gap)

                        # first part of parent Image
                        new_width = child_bound[X] - parent_bound[X]
                        _last_mile_gap = child_extension - parent_extension

                        # seperate the duplicate portion and disassociate parent and child
                        if new_width < 5 and _last_mile_gap > 5:

                            new_x = parent_bound[W]
                            _new_image = parent_image[:, :new_x]

                            _new_coords = self._get_coords(parent_bound[X], parent_bound[Y], _new_image, p_pos, False)
                            print('@05 New Imaqe:', _new_image.shape, _new_coords)
                            new_child_img.append([_new_coords, _new_image.copy()])
                            if self.SAVE_IMAGE_FLAG:
                                self._saveImage2File(_new_image, p_pos, child, 5)

                            new_x = new_x - new_width

                            _new_image = child_image[:, new_x:]

                            _new_coords = self._get_coords(parent_extension, parent_bound[Y], _new_image, p_pos, True)
                            print('@06 New Imaqe:', _new_image.shape, _new_coords)
                            new_child_img.append([_new_coords, _new_image.copy()])
                            if self.SAVE_IMAGE_FLAG:
                                self._saveImage2File(_new_image, p_pos, child, 6)

                            continue
                        # end if

                        _t_shade = child_bound[W] - _last_mile_gap
                        _last_mile_image = child_image[:, _t_shade:]
                        _size_share = _last_mile_image.shape[1] < parent_image.shape[1] * 0.25
                        print('****Size share', parent_image.shape[1] - _last_mile_image.shape[1], _size_share)

                        _skip_next = False
                        if _last_mile_gap < 7:
                            print('Merging child with parent...@ 10')
                            _new_image = np.hstack((parent_image, _last_mile_image))
                            new_width = parent_bound[W]
                            # since parent is combined with fragment , empty the remaining image of parent
                            remaining_p_image[:] = 0
                            _bound = parent_bound.copy()
                            _bound.append('P')
                            new_child_img.append([_bound, remaining_p_image.copy()])

                            _skip_next = True
                        else:
                            _new_image = parent_image[:, :new_width]
                        # end if

                        _last_X_pos = parent_bound[X]

                        _new_coords = self._get_coords(_last_X_pos, parent_bound[Y], _new_image, p_pos, True)
                        if self.SAVE_IMAGE_FLAG:
                            self._saveImage2File(_new_image, p_pos, child, 10)
                        # end if

                        print('@10 New Imaqe:', _new_image.shape, _new_coords)
                        new_child_img.append([_new_coords, _new_image.copy()])

                        # skip to the next child, if any, to process since
                        if _skip_next:
                            continue
                        # end if

                        remaining_p_image[:, :new_width] = 0

                        # overlap region between parent and child
                        _last_X_pos = _last_X_pos + new_width
                        new_x = new_width
                        new_width = parent_bound[W] - new_width

                        _new_image = parent_image[:, new_x: new_x + new_width]

                        _new_coords = self._get_coords(_last_X_pos, parent_bound[Y], _new_image, p_pos, True)
                        if self.SAVE_IMAGE_FLAG:
                            self._saveImage2File(_new_image, p_pos, child, 11)
                        # end if

                        print('@11 New Imaqe:', _new_image.shape, _new_coords)
                        new_child_img.append([_new_coords, _new_image.copy()])
                        remaining_p_image[:, new_x: new_x + new_width] = 0

                        # capture the remanining part of child image
                        # if the remaining image is wider than minimum threshold 3 units
                        if child_bound[W] - new_width > 3:

                            # push the pointer
                            _last_X_pos += new_width

                            new_x = new_width
                            _last_mile_width = child_bound[W] - new_width
                            print('Last mile width {}'.format(_last_mile_width))

                            _new_image = child_image[:, new_x:]
                            # with this condition merge the last vestibule to the previous image
                            if _last_mile_width <= 10:
                                _new_coords, _prev_image = new_child_img.pop()
                                _new_image = np.hstack((_prev_image, _new_image))
                                _new_coords[W] = _new_image.shape[1]
                                print('Image POPPED! Please ignore prev image...')
                            else:
                                # _new_image = child_image[:, new_x:]
                                _new_coords = self._get_coords(_last_X_pos, parent_bound[Y], _new_image, child, True)
                            # end if

                            print('@12 New Imaqe:', _new_image.shape, _new_coords)
                            new_child_img.append([_new_coords, _new_image.copy()])

                            if self.SAVE_IMAGE_FLAG:
                                self._saveImage2File(_new_image, p_pos, child, 12)
                            # end if

                        # end if

                    # if child area falls within the first 40% of parent width
                    elif child_extension <= parent_bound[X] + (parent_bound[W] * 0.40):
                        # get the starting point at parent X and include the width of the child
                        child_width = child_bound[X] - parent_bound[X] + child_bound[W]

                        _new_image = parent_image[:, :child_width]

                        _new_coords = self._get_coords(parent_bound[X], parent_bound[Y], _new_image, child, True)
                        print('@13 New Imaqe:', _new_image.shape, _new_coords)
                        new_child_img.append([_new_coords, _new_image.copy()])
                        if self.SAVE_IMAGE_FLAG:
                            self._saveImage2File(_new_image, p_pos, child, 13)

                        remaining_p_image[:, :child_width] = 0
                        _bound = parent_bound.copy()
                        _bound[X] = _bound[X] + child_width
                        _bound[W] = _bound[W] - child_width
                        _bound.append('P')
                        new_child_img.append([_bound, remaining_p_image.copy()])

                    # if child area falls within the last 30% of parent width
                    elif child_extension >= parent_bound[X] + \
                            (parent_bound[W] * 0.70) and child_extension <= parent_extension:

                        new_x = child_bound[X] - parent_bound[X]
                        _new_p_width = parent_bound[W] - new_x
                        _new_image = parent_image[:, new_x:]

                        _new_coords = self._get_coords(parent_bound[X] + new_x, parent_bound[Y], _new_image, child,
                                                       True)
                        new_child_img.append([_new_coords, _new_image.copy()])
                        print('@14 New Imaqe:', _new_image.shape, _new_coords)
                        if self.SAVE_IMAGE_FLAG:
                            self._saveImage2File(_new_image, p_pos, child, 14)
                        # end if

                        remaining_p_image[:, new_x:] = 0
                        _bound = parent_bound.copy()
                        _bound[W] = _new_p_width
                        _bound.append('P')
                        new_child_img.append([_bound, remaining_p_image.copy()])

                    elif child_extension < parent_extension:
                        # process other boundary conditions where child lies within parent
                        new_x = child_bound[X] - parent_bound[X]
                        _left_margin = 0

                        # when small distance exist between child and parent in the start position
                        if new_x > 5:
                            _new_image = parent_image[:, : new_x]

                            _new_coords = self._get_coords(parent_bound[X], parent_bound[Y], _new_image, child, True)
                            new_child_img.append([_new_coords, _new_image.copy()])
                            print('@16 New Imaqe:', _new_image.shape, _new_coords)
                            if self.SAVE_IMAGE_FLAG:
                                self._saveImage2File(_new_image, p_pos, child, 16)
                            # end if

                            remaining_p_image[:, : new_x] = 0
                            _bound = parent_bound.copy()
                            _bound.append('P')
                            new_child_img.append([_bound, remaining_p_image.copy()])
                            new_width = new_x

                        else:
                            _left_margin = new_x
                            new_x = 0
                        # end if

                        # this scenario is when the child is positioned at the center of the parent
                        if parent_bound[X] + new_x <= child_bound[X]:
                            _new_image = parent_image[:, new_x: new_x + child_bound[W]]

                            _new_coords = self._get_coords(parent_bound[X], child_bound[Y], _new_image, child, True)
                            new_child_img.append([_new_coords, _new_image.copy()])
                            print('@17 New Imaqe:', _new_image.shape, _new_coords)
                            if self.SAVE_IMAGE_FLAG:
                                self._saveImage2File(_new_image, p_pos, child, 17)

                            remaining_p_image[:, new_x: new_x + child_bound[W]] = 0
                            _bound = parent_bound.copy()
                            _bound.append('P')
                            new_child_img.append([_bound, remaining_p_image.copy()])
                            new_x = new_width = _left_margin + new_x + child_bound[W]
                        # end if

                        if parent_bound[W] > new_x:
                            _new_image = parent_image[:, new_x:]

                            _new_coords = self._get_coords(new_x + parent_bound[X], child_bound[Y], _new_image, child,
                                                           True)
                            new_child_img.append([_new_coords, _new_image.copy()])
                            if self.SAVE_IMAGE_FLAG:
                                self._saveImage2File(_new_image, p_pos, child, 18)

                            print('@18 New Imaqe:', _new_image.shape, _new_coords)

                            _bound = parent_bound.copy()
                            _bound.append('P')
                            remaining_p_image[:, new_x:] = 0
                            new_child_img.append([_bound, remaining_p_image.copy()])
                        # end if
                    # end if
                # end loop for children
                _parentstochange[p_pos] = new_child_img
            # end for loop for parents

        # end if

        # save children indexes that need to be removed
        for _parent in index_dict.keys():
            _childlist = index_dict[_parent]
            for _childid in _childlist:
                self._childrentoremove.append(_childid)
            # end for
        # end for

        self._childrentoremove.sort()
        print("Children to remove {}".format(self._childrentoremove))

        tpts, troi = self._redo_line_images(tpts, troi, _parentstochange, self._childrentoremove)

        return tpts, troi

    # end def

    def _insertSpacing(self, tpts=[], troi=[]):
        _prev = tpts[0][X] + tpts[0][W]

        self._spacesAfter = []
        _idx = 0
        for row in tpts:
            if row[X] - _prev > self._SPACE_WIDTH:
                self._spacesAfter.append(_idx)
                # self._saveImage2File(troi[_idx], 0, _idx, 20)
            # end for
            _idx += 1
            _prev = row[X] + row[W]
        # end for

        print('Insert space after {}'.format(self._spacesAfter))
        return self._spacesAfter

    # end for

    # end def

    # identify and stitch image/char fragments
    def _fuse_chars(self, tpts=[], troi=[]):

        if len(tpts) == 0 | len(troi) == 0:
            raise ValueError("@Fuse_chars: Invalid input!, expected image and position array!")

        self._spacesPos = self._insertSpacing(tpts, troi)

        tpts, troi, _index_dict = self._check_pip(tpts, troi)
        tpts, troi = self._process_pip(tpts, troi, _index_dict)

        tpts, troi = self._split_glued_chars(tpts, troi)

        # find the descrepency between tpts height width and troi image H/W
        for row in tpts:
            if (row[H], row[W]) != troi[row[IMG_IDX]].shape:
                print('Image Size mismatch! @ {} and {}'.format(row, troi[row[IMG_IDX]].shape))
            # end if
        # end if

        return tpts, troi

    # end def

    def _scrapblankspace(self, coords, rawimage):
        # try:
        _imagelength = rawimage.shape[1]
        _new_image = rawimage

        row_id = -1
        # scrap all blank lines to the right of the image
        for row in np.arange(_imagelength - 1, -1, -1):
            if cv.countNonZero(rawimage[:, row]) != 0:
                break
            # end if
            row_id = row
        # end for

        # crop the image
        if row_id > 0:
            _new_image = rawimage[:, :row_id]
            coords[W] = _new_image.shape[1]
        # end if

        # left side of parent image to be scrapped if any blanks are found
        _imagelength = _new_image.shape[1]
        row_id = -1
        # scrap all blank lines to the right of the image
        for row in np.arange(_imagelength):
            if cv.countNonZero(_new_image[:, row]) != 0:
                break
            # end if
            row_id = row
        # end for

        # crop the image
        if row_id > 0:
            _new_image = rawimage[:, row_id:]
            coords[W] = _new_image.shape[1]
            coords[X] += row_id
        # end if

        return coords, _new_image

        # except:
        #    _logger.error('Invalid data at function scrapblankspace')

    # end def

    def _redo_line_images(self, tpts=[], troi=[], parentstochange={}, _children2remove=[]):
        n_tpts = []
        n_troi = []

        new_idx = 0
        for row_tpts in tpts:
            row_idx = row_tpts[IMG_IDX]
            img_temp = troi[row_idx]

            if row_idx in _children2remove:
                continue

            # find the split images
            _getChildren = parentstochange.get(row_idx, None)
            if _getChildren is not None:
                # extract ONLY the last parent image
                for row in _getChildren[::-1]:
                    _coords = row[0]
                    if _coords[-1] == 'P':
                        # check if the parent image is blank
                        if cv.countNonZero(row[1]) == 0:
                            print('Blank parent found...')
                            break
                        # end if

                        _coords, _new_image = self._scrapblankspace(_coords, row[1])

                        n_troi.append(_new_image)
                        _coords = _coords[:-2]
                        _coords.append(new_idx)
                        n_tpts.append(_coords)
                        new_idx += 1
                        break
                    # end if
                # end for

                # extract children images
                for row in _getChildren:
                    _coords = row[0]  # _getChildren[row][0]
                    if _coords[-1] == 'C':
                        # check if the image is blank
                        if cv.countNonZero(row[1]) == 0:  # _getChildren[row][1]
                            print('Blank child found...')
                            continue
                        # end if
                        n_troi.append(row[1])
                        _coords = _coords[:-2]
                        _coords.append(new_idx)
                        n_tpts.append(_coords)
                        new_idx += 1
                    # end if
                # end for
            else:
                # print('Single Image index {}'.format(new_idx))
                row_tpts[-1] = new_idx
                n_tpts.append(row_tpts)
                n_troi.append(img_temp)
                new_idx += 1
            # end if
        # end for

        # empty childrentoremove list
        self._childrentoremove.clear()

        return n_tpts, n_troi

    # end def

    def display_char_line(self, tpts=[], troi=[]):
        a_vars = np.amax(tpts, axis=0)
        num_elements = a_vars[-1] + 1
        unit_image_width = a_vars[2]
        unit_image_height = a_vars[3]

        img_strip_width = (unit_image_width + self.DISP_GAP) * num_elements

        img_strip = np.zeros((unit_image_height + self.TOP_GUTTER, img_strip_width), dtype=np.uint8)

        ### This block creates Image strip to display the extracted chars for display
        s_idx = 0
        img_index = 0

        for row in tpts:
            img_strip[: row[H], s_idx: s_idx + row[W]] = troi[tpts[img_index][IMG_IDX]]

            s_idx = s_idx + row[W] + self.DISP_GAP
            img_index += 1
        # for loop
        #######

        imgShrunk = cv.resize(img_strip, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)  # decrease image size

        cv.imshow('Char Strip', imgShrunk)
        cv.waitKey(-1)
        cv.destroyAllWindows()

    def process_char(self, lineDetails=[], _logger=None):
        file_path = kannada.file_path
        with open(file_path + "mySavedDict.txt", "rb") as myFile:
            lineDetails = pickle.load(myFile)

        tpts = []
        troi = []

        # extract only last line for sampling , ****break command TO BE REMOVED***(
        new_line = 0
        for key in lineDetails.keys():
            if new_line != 10:
                new_line += 1
                continue
            # end if

            for pts, roi in lineDetails[key]:
                tpts.append(pts)
                troi.append(roi)
            # for loop
            break
        # for loop

        # print(pts)

        for idx in np.arange(len(tpts)):
            tpts[idx].append(idx)  # add row index to remember the char location after sorting
        # end for
        tpts.sort(key=lambda tpts: tpts[0])

        print('Stage 1: total chars {}'.format(len(tpts)))
        tpts, troi = self._fuse_chars(tpts, troi)

        print('Stage 2: total chars {}'.format(len(tpts)))

        return tpts, troi
    # end class

if __name__ == "__main__":
    _logger = kannada._initiatelogger()

    _process = PreProcess(_logger)
    tpts, troi = _process.process_char()
    tpts.sort(key=lambda tpts: tpts[0])
    _process.display_char_line(tpts, troi)
