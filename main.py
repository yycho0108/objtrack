class Tracker(object):
    def __init__(self):
        self.trk   = [] # currently tracked 2D Points
        self.trk_i = [] # self.pts[self.trk_i] are currently tracked
        self.pts   = [] # list of all known object feature point positions
        self.kpt   = [] # corresponding image coordinates for all points
        self.des   = [] # corresponding descriptors for all points

    def detect(self, img):
        pass

    def solve(self, img, pt0, pt1, li_c):
        #R1, t1, o1, i_u, i_i = self.solve(img, pt0, pt1, li_c)
        E, _ = findEssentialMat( ... )

    def track(self,
            img0, kpt0, des0,
            img1):
        # initialize containers
        o_pt0  = []
        o_pt1  = []

        # grab landmark points
        kpt_l = self.kpt[self.trk_i]
        #des_l = self.des[self.trk_i]

        # compute cross-frame match results
        # i_l0l, i_l00 = self.match(des_l, des0)
        # i_l1l, i_l11 = self.match(des_l, des1)
        # i_010, i_011 = self.match(des0,  des1)

        # detect already registered feature points from kpt0
        i_ll, i_l0 = self.match(self.des, des0) # TODO : global match (i.e. bounded search but not constrained to current track)

        # compute inverted match index
        m_l0  = np.ones(len(des0), dtype=np.bool)
        m_l0[i_l0] = False
        ni_l0 = np.where(m_l0)[0]

        # handle track recovery ...
        ril, ri0 = zip(*[(il, i0) for (il,i0) in zip(i_ll,i_l0) if il not in self.trk_i]) # track recovery
        # TODO : better way to get ^^
        pt1_lr, idx_tr = self.flow(img0, img1, pt0[ri0])
        self.kpt[ril[idx_tr]] = ptl_lr

        o_pt0.extend( pt0[ri0][idx_tr] )
        o_pt1.extend( pt1_lr[idx_tr]   )

        # track previous landmark/keypoints ...
        ptl1, idx_tl = self.flow(img0, img1, kpt_l)
        self.kpt[idx_tl] = ptl1

        o_pt0.extend( kpt_l[idx_tl] )
        o_pt1.extend( ptl1[idx_tl]  )

        # track 'new' points ...
        pt1, idx_t = self.flow(img0, img1, pt0[ni_l0])
        o_pt0.extend( pt0[ni_l0][idx_t] )
        o_pt1.extend( pt1[idx_t]  )
        # does not insert to tracking cache yet

        # compute output indices
        o_li   = np.r_[ril[idx_tr], self.trk_i[idx_tl]]
        o_li_c = np.s_[:len(idx_tr) + len(idx_tl)]
        # slice corresponding new descriptors
        o_des1 = des0[ni_l0][idx_t]

        # update tracking points and indices
        # NOTE : does not ADD points here.
        self.trk_i = np.intersect1d(self.trk_i, idx_tl)  # subtract lost points
        self.trk_i = np.union1d(self.trk_i, ril[idx_tr]) # add recovered points << TODO : validate

        return o_pt0, o_pt1, o_li, o_li_c, o_des1

    def track_v2(self, args):
        # detect already registered feature points from kpt0
        i_ml, i_m0 = self.match(self.des, des0) # TODO : global match (i.e. bounded search but not constrained to current track)

        # invert index
        m_m0 = np.zeros(len(des0), dtype=np.bool)
        m_m0[i_m0] = True
        m_ml = np.zeros(len(self.des), dtype=np.bool)
        m_ml[i_ml] = True

        ni_m0 = np.where(~m_m0)[0]
        ni_ml = np.where(~m_ml)[0]

        # index partition
        i0 = np.arange(len(des0))
        il = np.arange(len(self.des))
        ril, ri0 = zip(*[(il, i0) for (il,i0) in zip(i_ll, i_l0) if il not in self.trk_i]) # track recovery

        i0a = i0[ ni_m0 ] # case a : new; unmatched
        i0b = i0[ ri0   ] # case b : recovery; matched & untracked landmark

        ilc = il[ i_ml  ] # case c : suppress; matched & tracked landmark
        ild = il[ ni_ml ] # case d : old ; unmatched & tracked landmark

        # compute flows
        pt1a, idx_ta = self.flow(img0, img1, pt0[i0a])
        pt1b, idx_tb = self.flow(img0, img1, pt0[i0b])
        pt1c, idx_tc = self.flow(img0, img1, ptl[ilc])
        pt1d, idx_td = self.flow(img0, img1, ptl[ild])

        # case a : new
        pt0_n = pt0[i0a][idx_ta]
        pt1_n = pt1a[idx_ta]

        # case b : recovery
        pt0_r = pt0[i0b][idx_tb]
        pt1_r = pt1b[idx_tb]

        # case c : suppress
        pt0_s = ptl[ilc][idx_tc]
        pt1_s = pt1c[idx_tc]

        # case d : old
        pt0_o = ptl[ild][idx_td]
        pt1_o = pt1d[idx_td]

        # TODO : untrack failures for c/d
        self.trk

        # format output
        o_pt0 = np.concatenate([pt0_n, pt0_r, pt0_s, pt0_o], axis=0)
        o_pt1 = np.concatenate([pt1_n, pt1_r, pt1_s, pt1_o], axis=0)
        # TODO : forge o_li/ o_li_c from case b,c,d
        # TODO : forge des_new from case a

        # compute inverted match index
        m_l0  = np.ones(len(des0), dtype=np.bool)
        m_l0[i_l0] = False
        ni_l0 = np.where(m_l0)[0]

        # handle track recovery ...
        # TODO : better way to get ^^
        pt1_lr, idx_tr = self.flow(img0, img1, pt0[ri0])
        self.kpt[ril[idx_tr]] = ptl_lr

        o_pt0.extend( pt0[ri0][idx_tr] )
        o_pt1.extend( pt1_lr[idx_tr]   )

        # track previous landmark/keypoints ...
        ptl1, idx_tl = self.flow(img0, img1, kpt_l)
        self.kpt[idx_tl] = ptl1

        o_pt0.extend( kpt_l[idx_tl] )
        o_pt1.extend( ptl1[idx_tl]  )

        # track 'new' points ...
        pt1, idx_t = self.flow(img0, img1, pt0[ni_l0])
        o_pt0.extend( pt0[ni_l0][idx_t] )
        o_pt1.extend( pt1[idx_t]  )
        # does not insert to tracking cache yet

        # compute output indices
        o_li   = np.r_[ril[idx_tr], self.trk_i[idx_tl]]
        o_li_c = np.s_[:len(idx_tr) + len(idx_tl)]
        # slice corresponding new descriptors
        o_des1 = des0[ni_l0][idx_t]

        # update tracking points and indices
        # NOTE : does not ADD points here.
        self.trk_i = np.intersect1d(self.trk_i, idx_tl)  # subtract lost points
        self.trk_i = np.union1d(self.trk_i, ril[idx_tr]) # add recovered points << TODO : validate

        return o_pt0, o_pt1, o_li, o_li_c, o_des1


    def __call__(self, img):
        img1 = undistort(img)
        kpt1, des1 = self.detect(img1)
        self.hist.append( [img1, kpt1, des1] )

        # query history
        img0, kpt0, des0 = self.hist[-2]

        # track old + new points
        # such that self.pts[li] = pt0[li_c] = pt1[li_c]
        pt0, pt1, li, li_c, des_new = self.track(
                img0, kpt0, des0,
                img1, kpt1, des1)

        # compute current pose and new object points
        # i_u = indices that are good to update
        # i_i = indices that are good to insert
        R1, t1, o1, i_u, i_i = self.solve(img, pt0, pt1, li_c)

        # update old points with new position
        self.pts[li].update( o1[i_u] )

        # insert new points
        msk_i = np.ones(len(self.pts), dtype=np.bool) # insert mask
        msk_i[li_c] = False # do not insert previous landmarks again
        prv_size = len(self.pts)
        self.pts.extend(o1[i_i])
        self.kpt.extend(pt1[msk_i][i_i])
        self.des.extend(des_new[i_i])
        new_size = len(self.pts)
        self.trk_i.extend( np.arange(prv_size, new_size) )
