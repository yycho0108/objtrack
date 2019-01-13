import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import sys

def gc_msk_to_img(msk, img=None):
    if img is None:
        h, w = msk.shape[:2]
        img = np.empty((h,w,3), dtype=np.uint8)

    img[msk == cv2.GC_BGD] = 0
    img[msk == cv2.GC_PR_BGD] = 64
    img[msk == cv2.GC_PR_FGD] = 128
    img[msk == cv2.GC_FGD] = 255

    return img

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
    S_RECT  = 1 # initializing rectangle
    S_MODEL = 2 # refining model
    S_EVAL  = 3 # evaluate grabcut

    def __init__(self):
        # Data
        self.img_ = {
                'raw' : None,
                'trk' : None,
                'ann' : None,
                'msk' : None,
                'tmp' : None
                }
        #self.tmp_ = defaultdict(lambda: np.empty_like(self.img_['raw']))
        self.rect_ = None

        # GUI Prep
        self.win_ = {cv2.namedWindow(k) for k in ['raw','trk','ann','mod']}
        cv2.moveWindow('raw', 0, 0)
        cv2.moveWindow('trk', 320, 0)
        cv2.moveWindow('ann', 0, 240)
        cv2.moveWindow('mod', 320, 240)

        cv2.setMouseCallback('ann', self.mouse_cb)

        #self.ax_  = self.fig_.add_subplot(2,2,1) # ax0 for raw image
        #self.ax_t_ = self.fig_.add_subplot(2,2,2) # ax1 for tracking results
        #self.ax_a_ = self.fig_.add_subplot(2,2,3) # ax2 for grabcut annotations
        #self.ax_m_ = self.fig_.add_subplot(2,2,4) # ax2 for grabcut model visualization

        self.cam_ = cv2.VideoCapture(0)
        self.cam_.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cam_.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # GUI Handles
        # self.rpatch_  = patches.Rectangle(
        #         (0,0),
        #         1,1,
        #         linewidth=1,
        #         edgecolor='r',
        #         facecolor='none')
        # self.rpatch_.set_visible( False )
        # self.ax_a_.add_patch( self.rpatch_ )

        # Proc Handle
        self.trk_ = Tracker()

        # GUI State
        self.state_ = Runner.S_RECT

        # GUI data
        self.m0_ = None
        self.m1_ = None
        self.btn_ = None

        self.quit_ = False

    def read(self):
        print '<reading>'
        if self.img_['raw'] is not None:
            res, img = self.cam_.read(self.img_['raw'])
        else:
            res, img = self.cam_.read()

        if self.img_['raw'] is None:
            # initialize image data cache
            self.img_['raw'] = img
            self.img_['ann'] = img.copy()
            self.img_['trk'] = img.copy()
            self.img_['tmp'] = np.empty_like(img)
            self.img_['msk'] = np.full(img.shape[:2],
                    cv2.GC_PR_BGD,
                    dtype=np.uint8)
        else:
            # copy to annotations image
            np.copyto(self.img_['ann'], self.img_['raw'])
        print '</reading>'

        return res

    def key_cb(self, key):
        if key in [ord('q'), 27]:
            # terminate application
            self.quit_ = True

        # GUI state control
        if key in [ord('r')]:
            self.state_ = Runner.S_RECT
        if key in [ord('m')]:
            self.state_ = Runner.S_MODEL
        if key in [ord('e')]:
            self.state_ = Runner.S_EVAL

        if key in [ord('c')]:
            # clear
            self.img_['msk'].fill(cv2.GC_PR_BGD)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.m0_ = (x, y)
            self.btn_ = 'l'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.m0_ = (x, y)
            self.btn_ = 'r'
        elif event == cv2.EVENT_MOUSEMOVE:
            self.m1_ = (x, y)
            if self.state_ == Runner.S_MODEL:
                if self.btn_ is not None:
                    bmap = {
                            'l' : cv2.GC_FGD,
                            'r' : cv2.GC_BGD, # << ?? middle mouse?
                            }
                    cv2.circle(self.img_['msk'],
                            tuple(int(e) for e in self.m1_), 3,
                            bmap[self.btn_],
                            thickness=-1)

        elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
            if self.trk_.need_init():
                (x0,y0), (x1,y1) = self.m0_, self.m1_
                rect = tuple(int(np.round(x)) for x in (
                        min(x0,x1), min(y0,y1),
                        max(x0,x1), max(y0,y1)
                        ))
                self.rect_ = rect
                ann = self.trk_.initialize(self.img_['ann'], self.rect_)
                cv2.imshow('mod', ann)
            else:
                self.trk_.initialize(self.img_['ann'],
                        self.rect_, self.img_['msk'])

            # clear
            self.m0_ = None
            self.m1_ = None
            self.btn_ = None
            pass

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

        if self.state_ == Runner.S_RECT:
            #self.rpatch_.set_visible(True)
            #self.rpatch_.set_xy( self.m0_ )
            #self.rpatch_.set_width(1)
            #self.rpatch_.set_height(1)

            # initialize rectangle
            pass
        else:
            # handle manual annotations
            pass

    def move_cb(self, event):
        self.m1_ = event.xdata, event.ydata
        if (self.m0_ is not None):
            if self.state_ == Runner.S_RECT:
                # construct initial rectangle
                #w = self.m1_[0] - self.m0_[0]
                #h = self.m1_[1] - self.m0_[1]
                #self.rpatch_.set_width(w)
                #self.rpatch_.set_height(h)
                pass
            else:
                # construct refinement mask
                if event.button is not None:
                    bmap = {
                            1 : cv2.GC_FGD,
                            2 : cv2.GC_BGD, # << ?? middle mouse?
                            3 : cv2.GC_BGD
                            #3 : cv2.GC_PR_FGD
                            }
                    cv2.circle(self.img_['msk'],
                            tuple(int(e) for e in self.m1_), 3,
                            bmap[event.button],
                            thickness=-1)

    def run(self):
        # register callbacks

        # self.fig_.canvas.mpl_connect('key_press_event', self.key_cb)
        # self.fig_.canvas.mpl_connect('button_press_event', self.press_cb)
        # self.fig_.canvas.mpl_connect('button_release_event', self.release_cb)
        # self.fig_.canvas.mpl_connect('motion_notify_event', self.move_cb)
        # self.fig_.canvas.mpl_connect('close_event', sys.exit)

        while not self.quit_:
            # grab image
            res = self.read()
            if not res:
                print('ERROR : Camera read failure')
                break

            cv2.imshow('raw', self.img_['raw'])

            # visualization
            np.copyto(self.img_['tmp'], self.img_['raw'])

            # process image
            if self.state_ == Runner.S_EVAL:
                self.trk_(self.img_['raw'], self.img_['trk'])
                #self.ax_t_.imshow(self.img_['trk'][...,::-1] )
                cv2.imshow('trk', self.img_['trk'])
            elif self.state_ == Runner.S_RECT:
                if self.m0_ is not None and self.m1_ is not None:
                    # draw rectangle
                    cv2.rectangle(self.img_['tmp'],
                            tuple(int(e) for e in self.m0_),
                            tuple(int(e) for e in self.m1_),
                            (255,0,0))
                cv2.imshow('ann', self.img_['tmp'])
            elif self.state_ == Runner.S_MODEL:
                # ann + msk
                viz = cv2.addWeighted(self.img_['ann'], 0.5, gc_msk_to_img(self.img_['msk']), 0.5, 0.0)
                #viz = np.add(0.5 * self.img_['ann'], 127.0 * self.img_['msk'][..., None]).astype(np.uint8)
                #self.win_['ann'].imshow( viz )
                cv2.imshow('ann', viz)

            k = cv2.waitKey( 1 )
            self.key_cb( k )

            #self.ax_.imshow(self.img_['raw'][...,::-1] )
            #self.ax_a_.imshow(self.img_['ann'][...,::-1] )
            #self.ax_a_.imshow(self.img_['msk'], alpha=0.5, cmap='gray',
            #        vmin = 0, vmax = 2
            #        )


            #plt.pause(0.001)

def main():
    app = Runner()
    app.run()

if __name__ == "__main__":
    main()
