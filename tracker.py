import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import sys

class Tracker(object):
    def __init__(self):
        self.msk_ = None
        self.rect_ = None
        self.m_init_ = False
        self.m_bg_ = np.zeros((1,65), np.float64)
        self.m_fg_ = np.zeros((1,65), np.float64)

    def __call__(self, img, img_t):
        if self.m_init_:
            _, _, _ = cv2.grabCut(img,
                    self.msk_, None,
                    self.m_bg_,self.m_fg_,
                    5, cv2.GC_EVAL_FREEZE_MODEL)
                    #5, cv2.GC_EVAL)
            #msk = np.logical_not( (msk==2) | (msk==0) )
            msk = np.logical_or(self.msk_ == cv2.GC_FGD, self.msk_ == cv2.GC_PR_FGD)
            img_t.fill(0)
            np.copyto(img_t, img, where=msk[..., None])

        #img = cv.imread('messi5.jpg')
        #mask = np.zeros(img.shape[:2],np.uint8)
        #bgdModel = np.zeros((1,65),np.float64)
        #fgdModel = np.zeros((1,65),np.float64)
        #rect = (50,50,450,290)
        #cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
        #mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        #img = img*mask2[:,:,np.newaxis]
        #plt.imshow(img),plt.colorbar(),plt.show()
        pass

    def need_init(self):
        return (not self.m_init_)

    def initialize(self, img, rect, aux_msk=None):
        self.rect_ = rect
        self.m_init_ = True
        self.msk_ = np.full(img.shape[:2],
                cv2.GC_PR_BGD,
                dtype=np.uint8)

        if aux_msk is not None:
            np.copyto(self.msk_, aux_msk, where=np.logical_or(aux_msk == cv2.GC_FGD, aux_msk == cv2.GC_BGD))
            cv2.grabCut(img, self.msk_, None,
                    self.m_bg_, self.m_fg_,
                    iterCount=5, # itercount
                    mode=cv2.GC_INIT_WITH_MASK)
        else:
            cv2.grabCut(img, self.msk_, rect,
                    self.m_bg_, self.m_fg_,
                    iterCount=5, # itercount
                    mode=cv2.GC_INIT_WITH_RECT)

        # construct foreground visualization
        mask2 = np.where(
                (self.msk_==cv2.GC_FGD)|(self.msk_==cv2.GC_PR_FGD),1,0).astype('uint8')
        img = img*mask2[..., np.newaxis]
        return img

    def update(self):
        pass

class Runner(object):
    S_RINIT = 1 # initializing rectangle
    S_MODEL = 2 # refining model

    def __init__(self):
        # Data
        self.img_ = {
                'raw' : None,
                'trk' : None,
                'ann' : None,
                'msk' : None
                }
        self.rect_ = None

        # GUI Prep
        self.fig_ = plt.figure()
        self.ax_  = self.fig_.add_subplot(2,2,1) # ax0 for raw image
        self.ax_t_ = self.fig_.add_subplot(2,2,2) # ax1 for tracking results

        self.ax_a_ = self.fig_.add_subplot(2,2,3) # ax2 for grabcut annotations
        self.ax_m_ = self.fig_.add_subplot(2,2,4) # ax2 for grabcut model visualization

        self.cam_ = cv2.VideoCapture(0)
        self.cam_.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cam_.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # GUI Handles
        self.rpatch_  = patches.Rectangle(
                (0,0),
                1,1,
                linewidth=1,
                edgecolor='r',
                facecolor='none')
        self.rpatch_.set_visible( False )
        self.ax_a_.add_patch( self.rpatch_ )

        # Proc Handle
        self.trk_ = Tracker()

        # GUI State
        self.state_ = Runner.S_RINIT

        # GUI data
        self.m0_ = None
        self.m1_ = None

        self.quit_ = False

    def read(self):
        print 'reading ...'
        if self.img_['raw'] is not None:
            res, img = self.cam_.read(self.img_['raw'])
        else:
            res, img = self.cam_.read()

        if self.img_['raw'] is None:
            # initialize image data cache
            self.img_['raw'] = img
            self.img_['ann'] = img.copy()
            self.img_['trk'] = img.copy()
            self.img_['msk'] = np.full(img.shape[:2],
                    cv2.GC_PR_BGD,
                    dtype=np.uint8)

        return res

    def key_cb(self, event):
        if event.key in ['space', ' ']:
            # copy current image to editing queue
            np.copyto(self.img_['ann'], self.img_['raw'])
        if event.key in ['q', 'escape']:
            # terminate application
            self.quit_ = True

        if event.key in ['r']:
            self.state_ = Runner.S_RINIT

        if event.key in ['m']:
            self.state_ = Runner.S_MODEL

    def release_cb(self, event):
        if self.trk_.need_init():
            # construct rectangle
            (x0,y0), (x1,y1) = self.m0_, self.m1_
            rect = tuple(int(np.round(x)) for x in (
                    min(x0,x1), min(y0,y1),
                    max(x0,x1), max(y0,y1)
                    ))
            self.rect_ = rect
            ann = self.trk_.initialize(self.img_['ann'], self.rect_)
            self.ax_m_.imshow(ann[...,::-1])
        else:
            self.trk_.initialize(self.img_['ann'],
                    self.rect_, self.img_['msk'])
            #self.trk_.update(self.img_['ann'], self.img_['msk'])

        # reset data
        self.m0_ = None
        self.m1_ = None

    def press_cb(self, event):
        self.m0_ = (event.xdata, event.ydata)
        if self.trk_.need_init():
            self.rpatch_.set_visible(True)
            self.rpatch_.set_xy( self.m0_ )
            self.rpatch_.set_width(1)
            self.rpatch_.set_height(1)
            # initialize rectangle
        else:
            # delete rectangle
            # self.rpatch_.set_visible(False)
            # handle manual annotations
            pass
        pass

    def move_cb(self, event):
        self.m1_ = event.xdata, event.ydata
        if (self.m0_ is not None):
            if self.trk_.need_init():
                # construct initial rectangle
                w = self.m1_[0] - self.m0_[0]
                h = self.m1_[1] - self.m0_[1]
                self.rpatch_.set_width(w)
                self.rpatch_.set_height(h)
            else:
                # construct refinement mask
                if event.button is not None:
                    bmap = {
                            1 : cv2.GC_FGD,
                            2 : cv2.GC_BGD,
                            3 : cv2.GC_BGD
                            #3 : cv2.GC_PR_FGD
                            }
                    cv2.circle(self.img_['msk'],
                            tuple(int(e) for e in self.m1_), 3,
                            bmap[event.button],
                            thickness=-1)

    def run(self):
        # register callbacks
        self.fig_.canvas.mpl_connect('key_press_event', self.key_cb)
        self.fig_.canvas.mpl_connect('button_press_event', self.press_cb)
        self.fig_.canvas.mpl_connect('button_release_event', self.release_cb)
        self.fig_.canvas.mpl_connect('motion_notify_event', self.move_cb)
        self.fig_.canvas.mpl_connect('close_event', sys.exit)

        while not self.quit_:
            # grab image
            res = self.read()
            if not res:
                print('ERROR : Camera read failure')
                break

            # process image
            self.trk_(self.img_['raw'], self.img_['trk'])

            self.ax_.imshow(self.img_['raw'][...,::-1] )

            self.ax_a_.imshow(self.img_['ann'][...,::-1] )
            self.ax_a_.imshow(self.img_['msk'], alpha=0.5, cmap='gray',
                    vmin = 0, vmax = 2
                    )


            self.ax_t_.imshow(self.img_['trk'][...,::-1] )

            plt.pause(0.001)

def main():
    app = Runner()
    app.run()

if __name__ == "__main__":
    main()
